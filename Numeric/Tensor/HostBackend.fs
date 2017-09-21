namespace Tensor

open System
open System.Reflection
open System.Numerics
open System.Threading.Tasks
open System.Linq.Expressions
open System.Collections.Generic
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

open Tensor.Utils
open Tensor.Backend

/// pinned .NET managed memory (wraps a GCHandle)
type PinnedMemory (gcHnd: GCHandle, size: int64) =       
    let mutable disposed = false

    /// pointer to storage array 
    member this.Ptr = gcHnd.AddrOfPinnedObject()

    /// size of storage array in bytes
    member this.Size = size

    interface IDisposable with
        member this.Dispose() = 
            if not disposed then
                gcHnd.Free()
                disposed <- true

    override this.Finalize() = (this :> IDisposable).Dispose()

/// BLAS / LAPACK library imports
module internal BLAS =
    type lapack_int = int64
    type MKL_INT = int64

    type CBLAS_LAYOUT =
        | CblasRowMajor = 101
        | CblasColMajor = 102
    type CBLAS_TRANSPOSE =
        | CblasNoTrans = 111
        | CblasTrans = 112
        | CblasConjTrans = 113
    type CBLAS_UPLO = 
        | CblasUpper = 121
        | CblasLower = 122
    
    let LAPACK_ROW_MAJOR = 101
    let LAPACK_COL_MAJOR = 102

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern single cblas_sdot (MKL_INT n, nativeint x, MKL_INT incx, nativeint y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern double cblas_ddot (MKL_INT n, nativeint x, MKL_INT incx, nativeint y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_sgemv (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                             MKL_INT M, MKL_INT N, single alpha,
                             nativeint A, MKL_INT lda, nativeint X, MKL_INT incx,
                             single beta, nativeint Y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_dgemv (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                             MKL_INT M, MKL_INT N, double alpha,
                             nativeint A, MKL_INT lda, nativeint X, MKL_INT incx,
                             double beta, nativeint Y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_sgemm (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                             MKL_INT M, MKL_INT N, MKL_INT K, single alpha,
                             nativeint A, MKL_INT lda, nativeint B, MKL_INT ldb,
                             single beta, nativeint C, MKL_INT ldc)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_dgemm (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                             MKL_INT M, MKL_INT N, MKL_INT K, double alpha,
                             nativeint A, MKL_INT lda, nativeint B, MKL_INT ldb,
                             double beta, nativeint C, MKL_INT ldc)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_sgemm_batch (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE[] TransA, CBLAS_TRANSPOSE[] TransB,
                                   MKL_INT[] M, MKL_INT[] N, MKL_INT[] K, single[] alpha,
                                   nativeint[] A, MKL_INT[] lda, nativeint[] B, MKL_INT[] ldb,
                                   single[] beta, nativeint[] C, MKL_INT[] ldc,
                                   MKL_INT group_count, MKL_INT[] group_size)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_dgemm_batch (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE[] TransA, CBLAS_TRANSPOSE[] TransB,
                                   MKL_INT[] M, MKL_INT[] N, MKL_INT[] K, double[] alpha,
                                   nativeint[] A, MKL_INT[] lda, nativeint[] B, MKL_INT[] ldb,
                                   double[] beta, nativeint[] C, MKL_INT[] ldc,
                                   MKL_INT group_count, MKL_INT[] group_size)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_ssyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dsyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgesdd (int matrix_layout, char jobz, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      nativeint s, 
                                      nativeint u, lapack_int ldu, 
                                      nativeint vt, lapack_int ldvt)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgesdd (int matrix_layout, char jobz, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      nativeint s, 
                                      nativeint u, lapack_int ldu, 
                                      nativeint vt, lapack_int ldvt)


module internal HostBLASExtensions = 
    type Tensor.Backend.BLAS.MatrixInfo with
        member this.CTrans = 
            match this.Trans with
            | BLAS.NoTrans   -> BLAS.CBLAS_TRANSPOSE.CblasNoTrans
            | BLAS.Trans     -> BLAS.CBLAS_TRANSPOSE.CblasTrans
            | BLAS.ConjTrans -> BLAS.CBLAS_TRANSPOSE.CblasConjTrans

open HostBLASExtensions

module private Tools =
    let inline checkedInt layout (x: int64) =
        if int64 FSharp.Core.int.MinValue <= x && x <= int64 FSharp.Core.int.MaxValue then
            int x
        else failwithf "cannot convert tensor layout %A to 32-bit integer" layout

/// Fast layout operations.
[<Struct>]
[<StructuredFormatDisplay("FastLayout32 (Shape={Shape} Offset={Offset} Stride={Stride})")>]
type internal FastLayout32 = 
    val NDims   : int
    val NElems  : int
    val Offset  : int
    val Shape   : int []
    val Stride  : int []

    new (layout: TensorLayout) = {
        NDims   = TensorLayout.nDims layout
        NElems  = TensorLayout.nElems layout |> Tools.checkedInt layout
        Offset  = TensorLayout.offset layout |> Tools.checkedInt layout
        Shape   = TensorLayout.shape layout |> List.toArray |> Array.map (Tools.checkedInt layout)
        Stride  = TensorLayout.stride layout |> List.toArray |> Array.map (Tools.checkedInt layout)
    }

    member inline this.IsPosValid (pos: int[]) =
        if pos.Length = this.NDims then
            Array.forall2 (fun i size -> 0 <= i && i < size) pos this.Shape
        else false

    member inline this.UncheckedAddr (pos: int[]) =
        let mutable addr = this.Offset
        for d=0 to this.NDims-1 do
            addr <- addr + pos.[d] * this.Stride.[d]
        addr

    member inline this.Addr (pos: int64[]) =
        if pos.Length <> this.NDims then
            let msg = 
                sprintf "position %A has wrong dimensionality for tensor of shape %A"
                        pos this.Shape
            raise (IndexOutOfRange msg)                
        let mutable addr = this.Offset           
        for d=0 to this.NDims-1 do
            let p = int pos.[d]
            if (0 <= p && p < this.Shape.[d]) then
                addr <- addr + p * this.Stride.[d]
            else
                let msg = 
                    sprintf "position %A is out of range for tensor of shape %A"
                            pos this.Shape
                raise (IndexOutOfRange msg)
        addr

/// Fast index operations.
[<Struct>]
type internal PosIter32 = 
    val Pos             : int []
    val mutable Addr    : int
    val mutable Active  : bool
    val Shape           : int []
    val Stride          : int []               
    val FromDim         : int
    val ToDim           : int

    new (fl: FastLayout32, ?startPos, ?fromDim, ?toDim) = 
        let startPos = defaultArg startPos (Array.zeroCreate fl.NDims)
        let fromDim = defaultArg fromDim 0
        let toDim = defaultArg toDim (fl.NDims - 1)
        #if DEBUG
        if not (0 <= fromDim) then
            failwithf "fromDim=%d out of range for shape %A" fromDim fl.Shape
        if not (toDim < fl.NDims) then
            failwithf "toDim=%d out of range for shape %A" toDim fl.Shape
        #endif
        let active = 
            [0 .. fl.NDims]
            |> List.forall (fun d -> 
                if fromDim <= d && d <= toDim then 
                    0 <= startPos.[d] && startPos.[d] < fl.Shape.[d] 
                else true)
        {
            Pos     = Array.copy startPos
            Addr    = fl.UncheckedAddr startPos
            Active  = active
            Shape   = fl.Shape
            Stride  = fl.Stride
            FromDim = fromDim
            ToDim   = toDim
        }

    member inline this.MoveNext () =
        #if DEBUG
        if not this.Active then
            failwith "iteration past end attempted"
        #endif      

        // try incrementing starting from last axis
        let mutable increment = true
        let mutable d = this.ToDim
        while increment && d >= this.FromDim do
            if this.Pos.[d] = this.Shape.[d] - 1 then
                // was last element of that axis
                this.Addr <- this.Addr - this.Pos.[d] * this.Stride.[d]
                this.Pos.[d] <- 0
                d <- d - 1
            else
                // can increment this axis
                this.Addr <- this.Addr + this.Stride.[d]
                this.Pos.[d] <- this.Pos.[d] + 1
                increment <- false  
        // if we tried to increment past first axis, then iteration finished                            
        if d < this.FromDim then this.Active <- false                  

/// Data and fast layout of a host tensor.
type internal DataAndLayout<'T> = {
    Data:       'T[]
    FastLayout: FastLayout32
}


/// Generic scalar operation primitives.
type internal ScalarPrimitives<'T, 'TC> () =
    static let fscAsm = Assembly.GetAssembly(typeof<unit>)
    static let myAsm = Assembly.GetExecutingAssembly()
    static let fso = fscAsm.GetType("Microsoft.FSharp.Core.Operators", true)
    static let tso = myAsm.GetType("Tensor.Operators", true)

    static let a = Expression.Parameter(typeof<'T>, "a")
    static let b = Expression.Parameter(typeof<'T>, "b")
    static let c = Expression.Parameter(typeof<'TC>, "c")
    static let cond = Expression.Parameter(typeof<bool>, "cond")

    static let compileAny (fns: (unit -> Expression<_>) list) =        
        match fns |> List.tryPick (fun fn ->
                try Some (fn().Compile())
                with :? InvalidOperationException -> None) with
        | Some expr -> expr
        | None -> 
            failwithf "cannot compile scalar primitive for type %s" typeof<'T>.Name

    static let tryUnary op fns =
        let errExpr () =
            let msg = sprintf "the type %s does not implemented %s" typeof<'T>.Name op
            let thrw = Expression.Throw(Expression.Constant(InvalidOperationException msg))
            Expression.Lambda<Func<'T, 'T>>(Expression.Block(thrw, a), a)
        compileAny (fns @ [errExpr])

    static let tryBinary op fns =
        let errExpr () =
            let msg = sprintf "the type %s does not implemented %s" typeof<'T>.Name op
            let thrw = Expression.Throw(Expression.Constant(InvalidOperationException msg))
            Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Block(thrw, a), a, b)
        compileAny (fns @ [errExpr])

    static let tryCompare op fns =
        let errExpr () =
            let msg = sprintf "the type %s does not implemented %s" typeof<'T>.Name op
            let thrw = Expression.Throw(Expression.Constant(InvalidOperationException msg))
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.Block(thrw, Expression.Constant(false)), a, b)
        compileAny (fns @ [errExpr])

    member val ConvertFunc = 
        Expression.Lambda<Func<'TC, 'T>>(Expression.Convert(c, typeof<'T>), c).Compile()
    member inline this.Convert cv = this.ConvertFunc.Invoke(cv)

    member val UnaryPlusFunc = 
        tryUnary "~+" [fun () -> Expression.Lambda<Func<'T, 'T>>(Expression.UnaryPlus(a), a)]
    member inline this.UnaryPlus av = this.UnaryPlusFunc.Invoke(av)

    member val UnaryMinusFunc = 
        tryUnary "~-" [fun () -> Expression.Lambda<Func<'T, 'T>>(Expression.Negate(a), a)]
    member inline this.UnaryMinus av = this.UnaryMinusFunc.Invoke(av)

    member val AbsFunc = 
        let m = fso.GetMethod("Abs").MakeGenericMethod (typeof<'T>)   
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()   
    member inline this.Abs av = this.AbsFunc.Invoke(av)

    member val SgnFunc = 
        let m = tso.GetMethod("Sgn").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sgn av = this.SgnFunc.Invoke(av)

    member val LogFunc = 
        let m = fso.GetMethod("Log").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Log av = this.LogFunc.Invoke(av)

    member val Log10Func = 
        let m = fso.GetMethod("Log10").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Log10 av = this.Log10Func.Invoke(av)

    member val ExpFunc = 
        let m = fso.GetMethod("Exp").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Exp av = this.ExpFunc.Invoke(av)

    member val SinFunc = 
        let m = fso.GetMethod("Sin").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sin av = this.SinFunc.Invoke(av)

    member val CosFunc = 
        let m = fso.GetMethod("Cos").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Cos av = this.CosFunc.Invoke(av)

    member val TanFunc = 
        let m = fso.GetMethod("Tan").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Tan av = this.TanFunc.Invoke(av)

    member val AsinFunc = 
        let m = fso.GetMethod("Asin").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Asin av = this.AsinFunc.Invoke(av)

    member val AcosFunc = 
        let m = fso.GetMethod("Acos").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Acos av = this.AcosFunc.Invoke(av)

    member val AtanFunc = 
        let m = fso.GetMethod("Atan").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Atan av = this.AtanFunc.Invoke(av)

    member val SinhFunc = 
        let m = fso.GetMethod("Sinh").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sinh av = this.SinhFunc.Invoke(av)

    member val CoshFunc = 
        let m = fso.GetMethod("Cosh").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Cosh av = this.CoshFunc.Invoke(av)

    member val TanhFunc = 
        let m = fso.GetMethod("Tanh").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Tanh av = this.TanhFunc.Invoke(av)

    member val SqrtFunc = 
        let m = fso.GetMethod("Sqrt").MakeGenericMethod (typeof<'T>, typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sqrt av = this.SqrtFunc.Invoke(av)

    member val CeilingFunc = 
        let m = fso.GetMethod("Ceiling").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Ceiling av = this.CeilingFunc.Invoke(av)

    member val FloorFunc = 
        let m = fso.GetMethod("Floor").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Floor av = this.FloorFunc.Invoke(av)

    member val RoundFunc = 
        let m = fso.GetMethod("Round").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Round av = this.RoundFunc.Invoke(av)

    member val TruncateFunc = 
        let m = fso.GetMethod("Truncate").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Truncate av = this.TruncateFunc.Invoke(av)

    member val AddFunc = 
        tryBinary "+" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Add(a, b), a, b)]
    member inline this.Add av bv = this.AddFunc.Invoke(av, bv)
        
    member val SubtractFunc = 
        tryBinary "-" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Subtract(a, b), a, b)]
    member inline this.Subtract av bv = this.SubtractFunc.Invoke(av, bv)

    member val MultiplyFunc = 
        tryBinary "*" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Multiply(a, b), a, b)]
    member inline this.Multiply av bv = this.MultiplyFunc.Invoke(av, bv)

    member val DivideFunc = 
        tryBinary "/" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Divide(a, b), a, b)]
    member inline this.Divide av bv = this.DivideFunc.Invoke(av, bv)

    member val ModuloFunc = 
        tryBinary "%" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Modulo(a, b), a, b)]
    member inline this.Modulo av bv = this.ModuloFunc.Invoke(av, bv)

    member val PowerFunc = 
        // note: power is currently significantly slower than other operations
        let m = fso.GetMethod("op_Exponentiation").MakeGenericMethod (typeof<'T>, typeof<'T>)        
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Call(m, a, b), a, b).Compile()
    member inline this.Power av bv = this.PowerFunc.Invoke(av, bv)

    member val IsFiniteFunc : ('T -> bool) =
        match typeof<'T> with
        | t when t=typeof<single> -> 
            unbox (fun (v: single) -> not (System.Single.IsInfinity v || System.Single.IsNaN v))
        | t when t=typeof<double> -> 
            unbox (fun (v: double) -> not (System.Double.IsInfinity v || System.Double.IsNaN v))
        | _ -> (fun _ -> true)
    member inline this.IsFinite av = this.IsFiniteFunc av

    member val EqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.Equal(a, b), a, b)
        let fnInterface () =
            let m = typeof<IEquatable<'T>>.GetMethod("Equals") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.Call(a, m, b), a, b)
        tryCompare "=" [fnOp; fnInterface]          
    member inline this.Equal av bv = this.EqualFunc.Invoke(av, bv)

    member val NotEqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.NotEqual(a, b), a, b)
        let fnInterface () =
            let m = typeof<IEquatable<'T>>.GetMethod("Equals") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.IsFalse(Expression.Call(a, m, b)), a, b)
        tryCompare "!=" [fnOp; fnInterface]      
    member inline this.NotEqual av bv = this.NotEqualFunc.Invoke(av, bv)

    member val LessFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThan(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThan(Expression.Call(a, m, b), 
                                                                      Expression.Constant(0)), a, b)
        tryCompare "<" [fnOp; fnInterface]
    member inline this.Less av bv = this.LessFunc.Invoke(av, bv)

    member val LessOrEqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThanOrEqual(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThanOrEqual(Expression.Call(a, m, b), 
                                                                            Expression.Constant(0)), a, b)
        tryCompare "<=" [fnOp; fnInterface]
    member inline this.LessOrEqual av bv = this.LessOrEqualFunc.Invoke(av, bv)

    member val GreaterFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThan(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThan(Expression.Call(a, m, b), 
                                                                         Expression.Constant(0)), a, b)
        tryCompare ">" [fnOp; fnInterface]
    member inline this.Greater av bv = this.GreaterFunc.Invoke(av, bv)

    member val GreaterOrEqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThanOrEqual(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThanOrEqual(Expression.Call(a, m, b), 
                                                                                Expression.Constant(0)), a, b)
        tryCompare ">=" [fnOp; fnInterface]
    member inline this.GreaterOrEqual av bv = this.GreaterOrEqualFunc.Invoke(av, bv)

    member val IfThenElseFunc =
        Expression.Lambda<Func<bool, 'T, 'T, 'T>>(Expression.Condition(cond, a, b), cond, a, b).Compile()
    member inline this.IfThenElse condv ifTrue ifFalse = this.IfThenElseFunc.Invoke(condv, ifTrue, ifFalse)


/// Generic scalar operation primitives.
module internal ScalarPrimitives = 
    let private instances = Dictionary<Type * Type, obj>()
    let For<'T, 'TC> () =
        lock instances (fun () ->
            let types = typeof<'T>, typeof<'TC>
            match instances.TryFind types with
            | Some inst -> inst :?> ScalarPrimitives<'T, 'TC>
            | None ->
                let inst = ScalarPrimitives<'T, 'TC> ()
                instances.Add (types, inst)
                inst
        )
        

/// Scalar operations on host tensors.
type internal ScalarOps =

    static member inline ApplyNoaryOp (scalarOp: int64[] -> 'T, 
                                       trgt: DataAndLayout<'T>,
                                       isIndexed: bool, useThreads: bool) =        
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
                     
        let inline loops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let pos64 = Array.zeroCreate trgtPosIter.Pos.Length
            while trgtPosIter.Active do
                let mutable trgtAddr = trgtPosIter.Addr
                if nd = 0 then
                    trgt.Data.[trgtPosIter.Addr] <- scalarOp [||]
                elif isIndexed then
                    for d in 0 .. nd - 1 do
                        pos64.[d] <- int64 trgtPosIter.Pos.[d]
                    for i in 0 .. shape.[nd-1] - 1 do
                        //printfn "shape: %A pos64: %A" trgt.FastLayout.Shape pos64
                        trgt.Data.[trgtAddr] <- scalarOp pos64
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                else
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp [||]
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                trgtPosIter.MoveNext()
                    
        if useThreads && nd > 1 then
            Parallel.For (0, shape.[0], fun dim0Pos -> loops true dim0Pos) |> ignore
        else
            loops false 0

    static member inline ApplyUnaryOp (scalarOp: int64[] -> 'T1 -> 'T, 
                                       trgt: DataAndLayout<'T>, src1: DataAndLayout<'T1>, 
                                       isIndexed: bool, useThreads: bool) =        
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
                      
        let inline loops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let pos64 = Array.zeroCreate trgtPosIter.Pos.Length
            while trgtPosIter.Active do
                let mutable trgtAddr, src1Addr = trgtPosIter.Addr, src1PosIter.Addr
                if nd = 0 then
                    trgt.Data.[trgtPosIter.Addr] <- scalarOp [||] src1.Data.[src1PosIter.Addr] 
                elif isIndexed then
                    for d in 0 .. nd - 1 do
                        pos64.[d] <- int64 trgtPosIter.Pos.[d]
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp pos64 src1.Data.[src1Addr] 
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                else
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp [||] src1.Data.[src1Addr] 
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                trgtPosIter.MoveNext()
                src1PosIter.MoveNext()
                    
        if useThreads && nd > 1 then
            Parallel.For (0, shape.[0], fun dim0Pos -> loops true dim0Pos) |> ignore
        else
            loops false 0

    static member inline ApplyUnaryMethod (scalarOp: int64[] -> 'T1 -> unit, 
                                           src1: DataAndLayout<'T1>, 
                                           isIndexed: bool, useThreads: bool) =        
        let nd = src1.FastLayout.NDims
        let shape = src1.FastLayout.Shape
                      
        let inline loops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let pos64 = Array.zeroCreate src1PosIter.Pos.Length
            while src1PosIter.Active do
                let mutable src1Addr = src1PosIter.Addr
                if nd = 0 then
                    scalarOp [||] src1.Data.[src1PosIter.Addr] 
                elif isIndexed then
                    for d in 0 .. nd - 1 do
                        pos64.[d] <- int64 src1PosIter.Pos.[d]
                    for i in 0 .. shape.[nd-1] - 1 do
                        scalarOp pos64 src1.Data.[src1Addr] 
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                else
                    for i in 0 .. shape.[nd-1] - 1 do
                        scalarOp [||] src1.Data.[src1Addr] 
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                src1PosIter.MoveNext()
                    
        if useThreads && nd > 1 then
            Parallel.For (0, shape.[0], fun dim0Pos -> loops true dim0Pos) |> ignore
        else
            loops false 0

    static member inline ApplyBinaryOp (scalarOp: int64[] -> 'T1 -> 'T2 -> 'T, 
                                        trgt: DataAndLayout<'T>,
                                        src1: DataAndLayout<'T1>, src2: DataAndLayout<'T2>,
                                        isIndexed: bool, useThreads: bool) =        
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
                              
        let inline loops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src2PosIter = 
                PosIter32 (src2.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let pos64 = Array.zeroCreate trgtPosIter.Pos.Length
            while trgtPosIter.Active do
                let mutable trgtAddr, src1Addr, src2Addr = 
                    trgtPosIter.Addr, src1PosIter.Addr, src2PosIter.Addr
                if nd = 0 then
                    trgt.Data.[trgtPosIter.Addr] <- 
                        scalarOp [||] src1.Data.[src1PosIter.Addr] src2.Data.[src2PosIter.Addr]
                elif isIndexed then
                    for d in 0 .. nd - 1 do
                        pos64.[d] <- int64 trgtPosIter.Pos.[d]
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp pos64 src1.Data.[src1Addr] src2.Data.[src2Addr]
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        src2Addr <- src2Addr + src2.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                else
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp [||] src1.Data.[src1Addr] src2.Data.[src2Addr]
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        src2Addr <- src2Addr + src2.FastLayout.Stride.[nd-1]
                trgtPosIter.MoveNext()
                src1PosIter.MoveNext()
                src2PosIter.MoveNext()
                    
        if useThreads && nd > 1 then
            Parallel.For (0, shape.[0], fun dim0Pos -> loops true dim0Pos) |> ignore
        else
            loops false 0

    static member inline ApplyTernaryOp (scalarOp: int64[] -> 'T1 -> 'T2 -> 'T3 -> 'T, 
                                         trgt: DataAndLayout<'T>,
                                         src1: DataAndLayout<'T1>, src2: DataAndLayout<'T2>, src3: DataAndLayout<'T3>,
                                         isIndexed: bool, useThreads: bool) =        
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
                              
        let inline loops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src2PosIter = 
                PosIter32 (src2.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src3PosIter = 
                PosIter32 (src3.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let pos64 = Array.zeroCreate trgtPosIter.Pos.Length
            while trgtPosIter.Active do
                let mutable trgtAddr, src1Addr, src2Addr, src3Addr = 
                    trgtPosIter.Addr, src1PosIter.Addr, src2PosIter.Addr, src3PosIter.Addr
                if nd = 0 then
                    trgt.Data.[trgtPosIter.Addr] <- 
                        scalarOp [||] src1.Data.[src1PosIter.Addr] src2.Data.[src2PosIter.Addr] src3.Data.[src3PosIter.Addr]
                elif isIndexed then
                    for d in 0 .. nd - 1 do
                        pos64.[d] <- int64 trgtPosIter.Pos.[d]
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp pos64 src1.Data.[src1Addr] src2.Data.[src2Addr] src3.Data.[src3Addr]
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        src2Addr <- src2Addr + src2.FastLayout.Stride.[nd-1]
                        src3Addr <- src3Addr + src3.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                else
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp [||] src1.Data.[src1Addr] src2.Data.[src2Addr] src3.Data.[src3Addr]
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        src2Addr <- src2Addr + src2.FastLayout.Stride.[nd-1]
                        src3Addr <- src3Addr + src3.FastLayout.Stride.[nd-1]
                trgtPosIter.MoveNext()
                src1PosIter.MoveNext()
                src2PosIter.MoveNext()
                src3PosIter.MoveNext()
                    
        if useThreads && nd > 1 then
            Parallel.For (0, shape.[0], fun dim0Pos -> loops true dim0Pos) |> ignore
        else
            loops false 0

    static member inline ApplyNAryOp (scalarOp: int64[] -> 'T[] -> 'T, 
                                      trgt: DataAndLayout<'T>, srcs: DataAndLayout<'T>[],
                                      isIndexed: bool, useThreads: bool) =      
        if not (srcs |> Array.forall (fun src -> 
                List.ofArray trgt.FastLayout.Shape = List.ofArray src.FastLayout.Shape)) then
            invalidArg "srcs" "sources must have same shape as target"

        let nSrcs = srcs.Length
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
                        
        let inline genericInnerLoop (pos: int[]) (trgtAddr: int) (srcsAddr: int[]) =
            let mutable trgtAddr = trgtAddr
            let srcVals : 'T[] = Array.zeroCreate nSrcs
            if isIndexed then
                let pos64 = Array.zeroCreate pos.Length
                for d in 0 .. nd - 1 do
                    pos64.[d] <- int64 pos.[d]
                for pos in 0 .. shape.[nd-1] - 1 do
                    for s in 0 .. nSrcs-1 do
                        srcVals.[s] <- srcs.[s].Data.[srcsAddr.[s]]
                    trgt.Data.[trgtAddr] <- scalarOp pos64 srcVals
                    trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                    for s in 0 .. nSrcs-1 do
                        srcsAddr.[s] <- srcsAddr.[s] + srcs.[s].FastLayout.Stride.[nd-1]         
                    pos64.[nd-1] <- pos64.[nd-1] + 1L                        
            else
                for pos in 0 .. shape.[nd-1] - 1 do
                    for s in 0 .. nSrcs-1 do
                        srcVals.[s] <- srcs.[s].Data.[srcsAddr.[s]]
                    trgt.Data.[trgtAddr] <- scalarOp [||] srcVals
                    trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                    for s in 0 .. nSrcs-1 do
                        srcsAddr.[s] <- srcsAddr.[s] + srcs.[s].FastLayout.Stride.[nd-1]
         
        let inline scalarInnerLoop (trgtAddr: int) (srcsAddr: int[]) =    
            let srcsValue = 
                (srcs, srcsAddr) 
                ||> Array.map2 (fun (src: DataAndLayout<'T>) addr -> src.Data.[addr])
            trgt.Data.[trgtAddr] <- scalarOp [||] srcsValue

        let inline outerLoops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let srcsPosIter =
                srcs |> Array.map (fun src -> 
                    PosIter32 (src.FastLayout, startPos, fromDim=fromDim, toDim=nd-2))

            while trgtPosIter.Active do
                let srcsAddr = srcsPosIter |> Array.map (fun pi -> pi.Addr)
                if nd = 0 then
                    scalarInnerLoop trgtPosIter.Addr srcsAddr
                else
                    genericInnerLoop trgtPosIter.Pos trgtPosIter.Addr srcsAddr
                trgtPosIter.MoveNext()
                for s in 0 .. nSrcs-1 do
                    srcsPosIter.[s].MoveNext()
                    
        if useThreads && nd > 1 then
            Parallel.For (0, shape.[0], fun dim0Pos -> outerLoops true dim0Pos) |> ignore
        else
            outerLoops false 0

    static member inline ApplyAxisFold (foldOp: int64[] -> 'TS -> 'T1 -> 'TS, 
                                        extractOp: 'TS -> 'T, 
                                        trgt: DataAndLayout<'T>, src1: DataAndLayout<'T1>, 
                                        initial: Choice<'TS, DataAndLayout<'TS>>,
                                        isIndexed: bool, useThreads: bool) =        
        let nd = src1.FastLayout.NDims
        let shape = src1.FastLayout.Shape

        #if DEBUG
        if trgt.FastLayout.NDims <> nd-1 ||
           List.ofArray trgt.FastLayout.Shape <> List.ofArray src1.FastLayout.Shape.[0 .. nd-2] then
            failwithf "target of shape %A is incompatible with source shape %A" 
                      trgt.FastLayout.Shape src1.FastLayout.Shape
        #endif
                              
        let inline loops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos.[0 .. nd-2], fromDim=fromDim, toDim=nd-2)
            let initialPosIter =
                match initial with
                | Choice1Of2 initialVal -> None 
                | Choice2Of2 initialTensor -> 
                    Some (ref (PosIter32 (initialTensor.FastLayout, startPos.[0 .. nd-2], fromDim=fromDim, toDim=nd-2)))
            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let pos64 = Array.zeroCreate nd
            while trgtPosIter.Active do
                let mutable src1Addr = src1PosIter.Addr
                let mutable state =
                    match initial with
                    | Choice1Of2 initialVal -> initialVal
                    | Choice2Of2 initialTensor ->                                   
                        initialTensor.Data.[initialPosIter.Value.contents.Addr]
                if nd = 0 then
                    trgt.Data.[trgtPosIter.Addr] <- foldOp [||] state src1.Data.[src1Addr] |> extractOp
                elif isIndexed then
                    for d in 0 .. nd-2 do
                        pos64.[d] <- int64 trgtPosIter.Pos.[d]
                    pos64.[nd-1] <- 0L
                    for i in 0 .. shape.[nd-1] - 1 do
                        state <- foldOp pos64 state src1.Data.[src1Addr]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                    trgt.Data.[trgtPosIter.Addr] <- extractOp state
                else
                    for i in 0 .. shape.[nd-1] - 1 do
                        state <- foldOp [||] state src1.Data.[src1Addr]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                    trgt.Data.[trgtPosIter.Addr] <- extractOp state
                trgtPosIter.MoveNext()
                match initial with
                | Choice1Of2 _ -> () 
                | Choice2Of2 _ -> initialPosIter.Value.contents.MoveNext()
                src1PosIter.MoveNext()
                    
        if useThreads && nd > 1 then
            Parallel.For (0, shape.[0], fun dim0Pos -> loops true dim0Pos) |> ignore
        else
            loops false 0

    static member Fill (value: 'T, trgt: DataAndLayout<'T>) =
        let inline op pos = value
        ScalarOps.ApplyNoaryOp (op, trgt, isIndexed=false, useThreads=true)

    static member Copy (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let inline op pos a = a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Convert (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T1>) =
        let p = ScalarPrimitives.For<'T, 'T1>()
        let inline op pos (a: 'T1) = p.Convert a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member UnaryPlus (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.UnaryPlus a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member UnaryMinus (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.UnaryMinus a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Abs (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Abs a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Sgn (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Sgn a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Log (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Log a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Log10 (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Log10 a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Exp (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Exp a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Sin (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Sin a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Cos (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Cos a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Tan (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Tan a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Asin (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Asin a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Acos (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Acos a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Atan (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Atan a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Sinh (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Sinh a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Cosh (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Cosh a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Tanh (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Tanh a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Sqrt (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Sqrt a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Ceiling (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Ceiling a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Floor (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Floor a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Round (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Round a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Truncate (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.Truncate a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member IsFinite (trgt: DataAndLayout<bool>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a = p.IsFinite a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Negate (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>) =
        let inline op pos a = not a
        ScalarOps.ApplyUnaryOp (op, trgt, src1, isIndexed=false, useThreads=true)

    static member Add (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Add a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Subtract (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Subtract a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Multiply (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Multiply a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Divide (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Divide a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Modulo (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Modulo a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Power (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Power a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member MaxElemwise (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = if p.Greater a b then a else b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member MinElemwise (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = if p.Less a b then a else b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Equal (trgt: DataAndLayout<bool>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Equal a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member NotEqual (trgt: DataAndLayout<bool>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.NotEqual a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Less (trgt: DataAndLayout<bool>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Less a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member LessOrEqual (trgt: DataAndLayout<bool>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.LessOrEqual a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Greater (trgt: DataAndLayout<bool>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Greater a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member GreaterOrEqual (trgt: DataAndLayout<bool>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.GreaterOrEqual a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member And (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>, src2: DataAndLayout<bool>) =
        let inline op pos a b = a && b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member Or (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>, src2: DataAndLayout<bool>) =
        let inline op pos a b = a || b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)    

    static member Xor (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>, src2: DataAndLayout<bool>) =
        let inline op pos a b = a <> b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)    

    static member IfThenElse (trgt: DataAndLayout<'T>, cond: DataAndLayout<bool>, 
                              ifTrue: DataAndLayout<'T>, ifFalse: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos c t f = p.IfThenElse c t f
        ScalarOps.ApplyTernaryOp (op, trgt, cond, ifTrue, ifFalse, isIndexed=false, useThreads=true)    

    static member Gather (trgt: DataAndLayout<'T>, srcIndices: DataAndLayout<int64> option [],
                          src: DataAndLayout<'T>) =
        let inline op (trgtIdx: int64[]) = 
            let srcIdx = Array.init src.FastLayout.NDims (fun dim ->
                match srcIndices.[dim] with
                | Some i -> i.Data.[i.FastLayout.Addr trgtIdx]
                | None -> trgtIdx.[dim])
            src.Data.[src.FastLayout.Addr srcIdx]                                      
        ScalarOps.ApplyNoaryOp (op, trgt, isIndexed=true, useThreads=true)         

    static member Scatter (trgt: DataAndLayout<'T>, trgtIndices: DataAndLayout<int64> option [],
                           src: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (srcVal: 'T) = 
            let trgtIdx = Array.init trgt.FastLayout.NDims (fun dim ->
                match trgtIndices.[dim] with
                | Some i -> i.Data.[i.FastLayout.Addr srcIdx]
                | None -> srcIdx.[dim])
            let prvVal = trgt.Data.[trgt.FastLayout.Addr trgtIdx]
            trgt.Data.[trgt.FastLayout.Addr trgtIdx] <- p.Add prvVal srcVal
        // currently cannot use threads, because we have no interlocked addition
        ScalarOps.ApplyUnaryMethod (op, src, isIndexed=true, useThreads=false)     

    static member SumLastAxis (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (res: 'T) (v: 'T) = p.Add res v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 zero<'T>, isIndexed=false, useThreads=true)     

    static member ProductLastAxis (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (res: 'T) (v: 'T) = p.Multiply res v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 one<'T>, isIndexed=false, useThreads=true)  

    static member MaxLastAxis (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (res: 'T) (v: 'T) = if p.Greater res v then res else v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 minValue<'T>, isIndexed=false, useThreads=true)  

    static member MinLastAxis (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (res: 'T) (v: 'T) = if p.Less res v then res else v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 maxValue<'T>, isIndexed=false, useThreads=true)  

    static member AllLastAxis (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>) =
        let inline op (srcIdx: int64[]) (res: bool) (v: bool) = res && v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 true, isIndexed=false, useThreads=true)  

    static member AnyLastAxis (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>) =
        let inline op (srcIdx: int64[]) (res: bool) (v: bool) = res || v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 false, isIndexed=false, useThreads=true)  

    static member ArgMaxLastAxis (trgt: DataAndLayout<int64>, src1: DataAndLayout<'T>) =
        let nd = src1.FastLayout.NDims
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (maxPos, maxVal) (v: 'T) = 
            if p.Greater v maxVal then srcIdx.[nd-1], v
            else maxPos, maxVal
        ScalarOps.ApplyAxisFold (op, fst, trgt, src1, initial=Choice1Of2 (-1L, minValue<'T>), 
                                 isIndexed=true, useThreads=true)     

    static member ArgMinLastAxis (trgt: DataAndLayout<int64>, src1: DataAndLayout<'T>) =
        let nd = src1.FastLayout.NDims
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (minPos, minVal) (v: 'T) = 
            if p.Less v minVal then srcIdx.[nd-1], v
            else minPos, minVal
        ScalarOps.ApplyAxisFold (op, fst, trgt, src1, initial=Choice1Of2 (-1L, maxValue<'T>), 
                                 isIndexed=true, useThreads=true)     


// delegates for VectorOps
type internal FillDelegate<'T>   = delegate of 'T * DataAndLayout<'T> -> unit
type internal UnaryDelegate<'T>  = delegate of DataAndLayout<'T> * DataAndLayout<'T> -> unit
type internal BinaryDelegate<'T> = delegate of DataAndLayout<'T> * DataAndLayout<'T> * DataAndLayout<'T> -> unit
type internal CopyDelegate<'T>   = delegate of DataAndLayout<'T> * DataAndLayout<'T> -> unit

/// Vectorized (SIMD) operations on host tensors.
type internal VectorOps() =
    static let MethodDelegates = Dictionary<string * Type list, Delegate> ()

    static let vecTypes = [|typeof<byte>; typeof<sbyte>; typeof<int16>; typeof<uint16>;
                            typeof<int32>; typeof<uint32>; typeof<int64>; typeof<uint64>;
                            typeof<nativeint>; typeof<single>; typeof<double>|]

    static member private FillImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                  (value: 'T, trgt: DataAndLayout<'T>) = 
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape

        let inline vectorInnerLoop (trgtAddr: int) =                   
            let mutable trgtAddr = trgtAddr               
            let trgtVec = Vector<'T> value
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            for vecIter in 0 .. vecIters-1 do
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count 
            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- trgtVec.[restPos]
                trgtAddr <- trgtAddr + 1
                       
        let mutable trgtPosIter = PosIter32 (trgt.FastLayout, toDim=nd-2)
        while trgtPosIter.Active do
            match trgt.FastLayout.Stride.[nd-1] with
            | 1 -> vectorInnerLoop trgtPosIter.Addr 
            | _ -> failwith "vector operation to applicable to the given tensor"
            trgtPosIter.MoveNext()      
            
    static member inline private ApplyUnary (vectorOp: Vector<'T1> -> Vector<'T>,
                                             trgt: DataAndLayout<'T>, src1: DataAndLayout<'T1>) =        
        assert (Vector<'T>.Count = Vector<'T1>.Count)
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
        let src1Buf : 'T1[] = Array.zeroCreate Vector<'T>.Count

        let inline stride11InnerLoop (trgtAddr: int) (src1Addr: int) =                   
            let mutable trgtAddr, src1Addr = trgtAddr, src1Addr                
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            for vecIter in 0 .. vecIters-1 do
                let trgtVec = vectorOp (Vector (src1.Data, src1Addr)) 
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
                src1Addr <- src1Addr + Vector<'T>.Count 
            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                src1Buf.[restPos] <- src1.Data.[src1Addr]
                src1Addr <- src1Addr + 1
            let trgtVec = vectorOp (Vector src1Buf)
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- trgtVec.[restPos]
                trgtAddr <- trgtAddr + 1

        let inline stride10InnerLoop (trgtAddr: int) (src1Addr: int) =                   
            let mutable trgtAddr = trgtAddr                
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            let src1Vec = Vector (src1.Data.[src1Addr])
            let trgtVec = vectorOp src1Vec
            for vecIter in 0 .. vecIters-1 do
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count 
            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- trgtVec.[restPos]
                trgtAddr <- trgtAddr + 1
                     
        let mutable trgtPosIter = PosIter32 (trgt.FastLayout, toDim=nd-2)
        let mutable src1PosIter = PosIter32 (src1.FastLayout, toDim=nd-2)
        while trgtPosIter.Active do
            match trgt.FastLayout.Stride.[nd-1], src1.FastLayout.Stride.[nd-1] with
            | 1, 1 -> stride11InnerLoop trgtPosIter.Addr src1PosIter.Addr 
            | 1, 0 -> stride10InnerLoop trgtPosIter.Addr src1PosIter.Addr 
            | _ -> failwith "vector operation to applicable to the given tensor"
            trgtPosIter.MoveNext()
            src1PosIter.MoveNext()
                    
    static member inline private ApplyBinary (vectorOp: (Vector<'T1> * Vector<'T2>) -> Vector<'T>,
                                              trgt: DataAndLayout<'T>, 
                                              src1: DataAndLayout<'T1>, src2: DataAndLayout<'T2>) =
        assert (Vector<'T>.Count = Vector<'T1>.Count && Vector<'T1>.Count = Vector<'T2>.Count)
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
        let src1Buf : 'T1[] = Array.zeroCreate Vector<'T>.Count
        let src2Buf : 'T2[] = Array.zeroCreate Vector<'T>.Count
        
        let inline stride111InnerLoop (trgtAddr: int) (src1Addr: int) (src2Addr: int) =                   
            let mutable trgtAddr, src1Addr, src2Addr = trgtAddr, src1Addr, src2Addr                
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            for vecIter in 0 .. vecIters-1 do
                let trgtVec = vectorOp (Vector (src1.Data, src1Addr), Vector (src2.Data, src2Addr))
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
                src1Addr <- src1Addr + Vector<'T>.Count
                src2Addr <- src2Addr + Vector<'T>.Count

            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                src1Buf.[restPos] <- src1.Data.[src1Addr]
                src2Buf.[restPos] <- src2.Data.[src2Addr]
                src1Addr <- src1Addr + 1
                src2Addr <- src2Addr + 1
            let trgtVec = vectorOp (Vector src1Buf, Vector src2Buf)
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- trgtVec.[restPos]
                trgtAddr <- trgtAddr + 1

        let inline stride110InnerLoop (trgtAddr: int) (src1Addr: int) (src2Addr: int) =                   
            let mutable trgtAddr, src1Addr = trgtAddr, src1Addr               
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            let src2Vec = Vector (src2.Data.[src2Addr])
            for vecIter in 0 .. vecIters-1 do
                let trgtVec = vectorOp (Vector (src1.Data, src1Addr), src2Vec)
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
                src1Addr <- src1Addr + Vector<'T>.Count 

            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                src1Buf.[restPos] <- src1.Data.[src1Addr]
                src1Addr <- src1Addr + 1
            let trgtVec = vectorOp (Vector src1Buf, src2Vec)
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- trgtVec.[restPos]
                trgtAddr <- trgtAddr + 1
                       
        let inline stride101InnerLoop (trgtAddr: int) (src1Addr: int) (src2Addr: int) =                   
            let mutable trgtAddr, src2Addr = trgtAddr, src2Addr               
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            let src1Vec = Vector (src1.Data.[src1Addr])
            for vecIter in 0 .. vecIters-1 do
                let trgtVec = vectorOp (src1Vec, Vector (src2.Data, src2Addr))
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
                src2Addr <- src2Addr + Vector<'T>.Count 

            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                src2Buf.[restPos] <- src2.Data.[src2Addr]
                src2Addr <- src2Addr + 1
            let trgtVec = vectorOp (src1Vec, Vector src2Buf)
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- trgtVec.[restPos]
                trgtAddr <- trgtAddr + 1

        let inline stride100InnerLoop (trgtAddr: int) (src1Addr: int) (src2Addr: int) =                   
            let mutable trgtAddr = trgtAddr
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            let trgtVec = vectorOp (Vector src1.Data.[src1Addr], Vector src2.Data.[src2Addr])
            for vecIter in 0 .. vecIters-1 do
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count

            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- trgtVec.[restPos]
                trgtAddr <- trgtAddr + 1
                      
        let mutable trgtPosIter = PosIter32 (trgt.FastLayout, toDim=nd-2)
        let mutable src1PosIter = PosIter32 (src1.FastLayout, toDim=nd-2)
        let mutable src2PosIter = PosIter32 (src2.FastLayout, toDim=nd-2)
        while trgtPosIter.Active do
            match trgt.FastLayout.Stride.[nd-1], 
                  src1.FastLayout.Stride.[nd-1], src2.FastLayout.Stride.[nd-1] with
            | 1, 1, 1 -> stride111InnerLoop trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
            | 1, 1, 0 -> stride110InnerLoop trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
            | 1, 0, 1 -> stride101InnerLoop trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
            | 1, 0, 0 -> stride100InnerLoop trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
            | _ -> failwith "vector operation to applicable to the given tensor"
            trgtPosIter.MoveNext()
            src1PosIter.MoveNext()
            src2PosIter.MoveNext()                   

    static member private CopyImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.ApplyUnary (id, trgt, src1)

    static member private UnaryMinusImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.ApplyUnary (Vector.Negate, trgt, src1)

    static member private AbsImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.ApplyUnary (Vector.Abs, trgt, src1)

    static member private SqrtImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.ApplyUnary (Vector.SquareRoot, trgt, src1)

    static member private AddImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.ApplyBinary (Vector.Add, trgt, src1, src2)

    static member private SubtractImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.ApplyBinary (Vector.Subtract, trgt, src1, src2)

    static member private MultiplyImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let inline vecOp (a: Vector<'T>, b: Vector<'T>) = Vector.Multiply (a, b)
        VectorOps.ApplyBinary (vecOp, trgt, src1, src2)

    static member private DivideImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.ApplyBinary (Vector.Divide, trgt, src1, src2)

    static member private MaxElemwiseImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.ApplyBinary (Vector.Max, trgt, src1, src2)

    static member private MinElemwiseImpl<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                      (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.ApplyBinary (Vector.Min, trgt, src1, src2)

    static member inline private Method<'D when 'D :> Delegate> (name: string) : 'D = 
        let dt = typeof<'D>.GenericTypeArguments
        let dtl = dt |> List.ofArray
        match MethodDelegates.TryFind (name, dtl) with
        | Some del -> del :?> 'D
        | None -> 
            let mi = typeof<VectorOps>.GetMethod (name, BindingFlags.Static ||| BindingFlags.NonPublic)
            let mi = mi.MakeGenericMethod(dt)
            let del = mi.CreateDelegate(typeof<'D>) 
            MethodDelegates.[(name, dtl)] <- del
            del :?> 'D

    static member Fill (value: 'T, trgt: DataAndLayout<'T>) =
        VectorOps.Method<FillDelegate<'T>>("FillImpl").Invoke (value, trgt) 

    static member Copy (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.Method<CopyDelegate<'T>>("CopyImpl").Invoke (trgt, src1) 

    static member UnaryMinus (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.Method<UnaryDelegate<'T>>("UnaryMinusImpl").Invoke (trgt, src1) 

    static member Abs (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.Method<UnaryDelegate<'T>>("AbsImpl").Invoke (trgt, src1) 

    static member Sqrt (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        VectorOps.Method<UnaryDelegate<'T>>("SqrtImpl").Invoke (trgt, src1) 

    static member Add (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("AddImpl").Invoke (trgt, src1, src2) 

    static member Subtract (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("SubtractImpl").Invoke (trgt, src1, src2) 

    static member Multiply (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("MultiplyImpl").Invoke (trgt, src1, src2) 

    static member Divide (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("DivideImpl").Invoke (trgt, src1, src2) 

    static member MaxElemwise (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("MaxElemwiseImpl").Invoke (trgt, src1, src2) 

    static member MinElemwise (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("MinElemwiseImpl").Invoke (trgt, src1, src2) 

    static member CanUse (trgt: DataAndLayout<'T>, ?src1: DataAndLayout<'T1>, ?src2: DataAndLayout<'T2>) =
        match trgt.FastLayout.NDims with
        | 0 -> false
        | nd ->
            let canUseType =
                vecTypes |> Array.contains typeof<'T>
            let canUseTrgt = 
                trgt.FastLayout.Stride.[nd-1] = 1
            let canUseSrc src = 
                match src with
                | Some src -> 
                    let str = src.FastLayout.Stride 
                    str.[nd-1] = 1 || str.[nd-1] = 0
                | None -> true
            canUseType && canUseTrgt && canUseSrc src1 && canUseSrc src2


/// type-neutral interface to TensorHostStorage<'T>
type ITensorHostStorage =
    /// the underlying data array
    abstract Data: Array
    /// size of underlying data array in elements
    abstract DataSize: int64
    /// size of underlying data array in bytes
    abstract DataSizeInBytes: int64
    /// pins the underlying data array and returns the corresponding pinned memory
    abstract Pin: unit -> PinnedMemory


/// Storage (using a .NET array) for host tensors.
type TensorHostStorage<'T> (data: 'T []) =

    /// allocates a new data array with the given number of elements
    new (nElems: int64) =
        if nElems > int64 FSharp.Core.int32.MaxValue then
            failwithf "Cannot create host tensor storage for %d elements, the current
                       limit is %d elements." nElems FSharp.Core.int32.MaxValue
        TensorHostStorage<'T> (Array.zeroCreate (int32 nElems))        

    /// the underlying data array
    member this.Data = data

    /// pins the underlying data array and returns the corresponding pinned memory
    member this.Pin () =
        let gcHnd = GCHandle.Alloc (data, GCHandleType.Pinned)
        new PinnedMemory (gcHnd, data.LongLength * sizeof64<'T>) 

    /// size of underlying data array in elements
    member this.DataSize = data.LongLength

    /// size of underlying data array in bytes
    member this.DataSizeInBytes = data.LongLength * sizeof64<'T>

    interface ITensorStorage<'T> with
        member this.Backend layout =
            TensorHostBackend<'T> (layout, this) :> ITensorBackend<_>
        member this.Dev = 
            TensorHostDevice.Instance :> ITensorDevice

    interface ITensorHostStorage with
        member this.Data = this.Data :> Array
        member this.DataSize = this.DataSize
        member this.DataSizeInBytes = this.DataSizeInBytes
        member this.Pin () = this.Pin ()

    interface BLAS.IBLASStorage with
        member this.Pin () =
            let pinHnd = this.Pin ()
            pinHnd :> IDisposable, pinHnd.Ptr

    override this.Equals other =
        match other with
        | :? TensorHostStorage<'T> as os ->
            LanguagePrimitives.PhysicalEquality this.Data os.Data
        | _ -> false            

    override this.GetHashCode () =
        RuntimeHelpers.GetHashCode data
        

/// Backend for host tensors.
and TensorHostBackend<'T> (layout: TensorLayout, storage: TensorHostStorage<'T>) =

    /// true if BLAS operations support type 'T 
    let isBlasSupported =
        let blasSupportedTypes = [typeof<single>; typeof<double>]
        blasSupportedTypes |> List.contains typeof<'T> 

    /// fast layout
    member val internal FastLayout = FastLayout32 layout

    /// underlying TensorHostStorate<'T>
    member this.Storage = storage

    /// underlying data array
    member val Data = storage.Data

    /// data array and fast layout
    member inline internal this.DataAndLayout = 
        {Data=this.Data; FastLayout=this.FastLayout}
              
    /// gets DataAndLayout for specified tensors
    static member internal GetDataAndLayout (t: Tensor<'T>) =
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout

    /// gets DataAndLayout for specified tensors
    static member internal GetDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>) =
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Backend :?> TensorHostBackend<'TA>).DataAndLayout 

    /// gets DataAndLayout for specified tensors
    static member internal GetDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>, b: Tensor<'TB>) =
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Backend :?> TensorHostBackend<'TA>).DataAndLayout,
        (b.Backend :?> TensorHostBackend<'TB>).DataAndLayout 

    /// gets layouts for specified targets and sources, optimized for an element-wise operation
    static member internal ElemwiseLayouts (trgt: TensorLayout, srcs: TensorLayout list) =
        let dimGood = 
            [0 .. trgt.NDims-1]
            |> List.map (fun d ->
                trgt.Stride.[d] = 1L &&
                srcs |> List.forall (fun src -> src.Stride.[d]=1L || src.Stride.[d]=0L))
        if dimGood |> List.exists id then
            let bestLastDim =
                [0 .. trgt.NDims-1]
                |> List.maxBy (fun d ->
                    if dimGood.[d] then trgt.Shape.[d] else -1L)
            let swap = TensorLayout.swapDim bestLastDim (trgt.NDims-1)
            swap trgt, List.map swap srcs
        else
            trgt, srcs

    /// gets DataAndLayout for specified tensors, optimized for an element-wise operation
    static member internal ElemwiseDataAndLayout (t: Tensor<'T>) =        
        let tl, ls = TensorHostBackend<_>.ElemwiseLayouts (t.Layout, [])
        (t.Relayout(tl).Backend :?> TensorHostBackend<'T>).DataAndLayout        

    /// gets DataAndLayout for specified tensors, optimized for an element-wise operation
    static member internal ElemwiseDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>) =
        let tl, ls = TensorHostBackend<_>.ElemwiseLayouts (t.Layout, [a.Layout])
        (t.Relayout(tl).Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Relayout(ls.[0]).Backend :?> TensorHostBackend<'TA>).DataAndLayout 

    /// gets DataAndLayout for specified tensors, optimized for an element-wise operation
    static member internal ElemwiseDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>, b: Tensor<'TB>) =
        let tl, ls = TensorHostBackend<_>.ElemwiseLayouts (t.Layout, [a.Layout; b.Layout])
        (t.Relayout(tl).Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Relayout(ls.[0]).Backend :?> TensorHostBackend<'TA>).DataAndLayout,
        (b.Relayout(ls.[1]).Backend :?> TensorHostBackend<'TB>).DataAndLayout 

    /// gets DataAndLayout for specified tensors, optimized for an element-wise operation
    static member internal ElemwiseDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) =
        let tl, ls = TensorHostBackend<_>.ElemwiseLayouts (t.Layout, [a.Layout; b.Layout; c.Layout])
        (t.Relayout(tl).Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Relayout(ls.[0]).Backend :?> TensorHostBackend<'TA>).DataAndLayout,
        (b.Relayout(ls.[1]).Backend :?> TensorHostBackend<'TB>).DataAndLayout,
        (c.Relayout(ls.[2]).Backend :?> TensorHostBackend<'TC>).DataAndLayout 


    interface ITensorBackend<'T> with
        member this.Item 
            with get idx = this.Data.[this.FastLayout.Addr idx]
            and set idx value = this.Data.[this.FastLayout.Addr idx] <- value

        member this.FillConst (value, trgt) =
            let trgt = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt)
            if VectorOps.CanUse (trgt) then VectorOps.Fill (value, trgt)
            else ScalarOps.Fill (value, trgt)

        member this.Copy (trgt, src) =
            if TensorLayout.hasContiguousMemory trgt.Layout &&
               TensorLayout.hasContiguousMemory src.Layout &&
               trgt.Layout.Stride = src.Layout.Stride then
                // use array block copy for contiguous memory block
                let trgt, src = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, src)
                if trgt.FastLayout.NElems > 0 then
                    Array.Copy (src.Data, src.FastLayout.Offset, 
                                trgt.Data, trgt.FastLayout.Offset, trgt.FastLayout.NElems)
            else 
                let trgt, src = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, src)
                if VectorOps.CanUse (trgt, src) then VectorOps.Copy (trgt, src)
                else ScalarOps.Copy (trgt, src)

        member this.Transfer (trgt, src) =
            false

        member this.Convert (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Convert (trgt, a)

        member this.Fill (fn, trgt, useThreads) = 
            let trgt = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt)
            let inline scalarOp idx = fn ()
            ScalarOps.ApplyNoaryOp (scalarOp, trgt, isIndexed=false, useThreads=useThreads)

        member this.FillIndexed (fn, trgt, useThreads) = 
            let trgt = TensorHostBackend<_>.GetDataAndLayout (trgt)
            ScalarOps.ApplyNoaryOp (fn, trgt, isIndexed=true, useThreads=useThreads)

        member this.Map (fn, trgt, a, useThreads) = 
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            let inline scalarOp idx av = fn av
            ScalarOps.ApplyUnaryOp (scalarOp, trgt, a, isIndexed=false, useThreads=useThreads)

        member this.MapIndexed (fn, trgt, a, useThreads) = 
            let trgt, a = TensorHostBackend<_>.GetDataAndLayout (trgt, a)
            ScalarOps.ApplyUnaryOp (fn, trgt, a, isIndexed=true, useThreads=useThreads)

        member this.Map2 (fn, trgt, a, b, useThreads) = 
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            let inline scalarOp idx av bv = fn av bv
            ScalarOps.ApplyBinaryOp (scalarOp, trgt, a, b, isIndexed=false, useThreads=useThreads)

        member this.MapIndexed2 (fn, trgt, a, b, useThreads) =
            let trgt, a, b = TensorHostBackend<_>.GetDataAndLayout (trgt, a, b)
            ScalarOps.ApplyBinaryOp (fn, trgt, a, b, isIndexed=true, useThreads=useThreads)
      
        member this.UnaryPlus (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.UnaryPlus (trgt, a)

        member this.UnaryMinus (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            if VectorOps.CanUse (trgt, a) then VectorOps.UnaryMinus (trgt, a)
            else ScalarOps.UnaryMinus (trgt, a)

        member this.Abs (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            if VectorOps.CanUse (trgt, a) then VectorOps.Abs (trgt, a)
            else ScalarOps.Abs (trgt, a)

        member this.Sgn (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Sgn (trgt, a)

        member this.Log (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Log (trgt, a)

        member this.Log10 (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Log10 (trgt, a)

        member this.Exp (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Exp (trgt, a)

        member this.Sin (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Sin (trgt, a)

        member this.Cos (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Cos (trgt, a)

        member this.Tan (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Tan (trgt, a)

        member this.Asin (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Asin (trgt, a)

        member this.Acos (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Acos (trgt, a)

        member this.Atan (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Atan (trgt, a)

        member this.Sinh (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Sinh (trgt, a)

        member this.Cosh (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Cosh (trgt, a)

        member this.Tanh (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Tanh (trgt, a)

        member this.Sqrt (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            if VectorOps.CanUse (trgt, a) then VectorOps.Sqrt (trgt, a)
            else ScalarOps.Sqrt (trgt, a)

        member this.Ceiling (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Ceiling (trgt, a)

        member this.Floor (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Floor (trgt, a)

        member this.Round (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Round (trgt, a)

        member this.Truncate (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Truncate (trgt, a)

        member this.IsFinite (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.IsFinite (trgt, a)

        member this.Negate (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Negate (trgt, a)

        member this.Add (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            if VectorOps.CanUse (trgt, a, b) then VectorOps.Add (trgt, a, b)
            else ScalarOps.Add (trgt, a, b)

        member this.Subtract (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            if VectorOps.CanUse (trgt, a, b) then VectorOps.Subtract (trgt, a, b)
            else ScalarOps.Subtract (trgt, a, b)

        member this.Multiply (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            if VectorOps.CanUse (trgt, a, b) then VectorOps.Multiply (trgt, a, b)
            else ScalarOps.Multiply (trgt, a, b)

        member this.Divide (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            if VectorOps.CanUse (trgt, a, b) then VectorOps.Divide (trgt, a, b)
            else ScalarOps.Divide (trgt, a, b)

        member this.Modulo (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.Modulo (trgt, a, b)

        member this.Power (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.Power (trgt, a, b)

        member this.MaxElemwise (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            if VectorOps.CanUse (trgt, a, b) then VectorOps.MaxElemwise (trgt, a, b)
            else ScalarOps.MaxElemwise (trgt, a, b)

        member this.MinElemwise (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            if VectorOps.CanUse (trgt, a, b) then VectorOps.MinElemwise (trgt, a, b)
            else ScalarOps.MinElemwise (trgt, a, b)

        member this.Equal (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.Equal (trgt, a, b)

        member this.NotEqual (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.NotEqual (trgt, a, b)

        member this.Less (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.Less (trgt, a, b)

        member this.LessOrEqual (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.LessOrEqual (trgt, a, b)

        member this.Greater (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.Greater (trgt, a, b)

        member this.GreaterOrEqual (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.GreaterOrEqual (trgt, a, b)

        member this.And (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.And (trgt, a, b)

        member this.Or (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.Or (trgt, a, b)

        member this.Xor (trgt, a, b) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            ScalarOps.Xor (trgt, a, b)

        member this.IfThenElse (trgt, cond, ifTrue, ifFalse) =
            let trgt, cond, ifTrue, ifFalse = 
                TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, cond, ifTrue, ifFalse)
            ScalarOps.IfThenElse (trgt, cond, ifTrue, ifFalse)

        member this.Gather (trgt, srcIndices, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            let srcIndices = 
                srcIndices 
                |> List.map (Option.map (fun i -> (i.Backend :?> TensorHostBackend<int64>).DataAndLayout))
                |> Array.ofList
            ScalarOps.Gather (trgt, srcIndices, src)

        member this.Scatter (trgt, trgtIndices, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            let trgtIndices = 
                trgtIndices 
                |> List.map (Option.map (fun i -> (i.Backend :?> TensorHostBackend<int64>).DataAndLayout))
                |> Array.ofList
            ScalarOps.Scatter (trgt, trgtIndices, src)

        member this.FoldLastAxis (fn, initial, trgt, a, useThreads) = 
            let initial, trgt, a = TensorHostBackend<_>.GetDataAndLayout (initial, trgt, a)
            let inline foldOp idx state xv = fn state xv
            ScalarOps.ApplyAxisFold (foldOp, id, trgt, a, Choice2Of2 initial, isIndexed=false, useThreads=useThreads)

        member this.FoldLastAxisIndexed (fn, initial, trgt, a, useThreads) = 
            let initial, trgt, a = TensorHostBackend<_>.GetDataAndLayout (initial, trgt, a)
            ScalarOps.ApplyAxisFold (fn, id, trgt, a, Choice2Of2 initial, isIndexed=true, useThreads=useThreads)

        member this.SumLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.SumLastAxis (trgt, src)
            
        member this.ProductLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.ProductLastAxis (trgt, src)

        member this.MinLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.MinLastAxis (trgt, src)

        member this.MaxLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.MaxLastAxis (trgt, src)

        member this.AllLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.AllLastAxis (trgt, src)

        member this.AnyLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.AnyLastAxis (trgt, src)

        member this.ArgMinLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.ArgMinLastAxis (trgt, src)

        member this.ArgMaxLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.ArgMaxLastAxis (trgt, src)

        member this.VecVecDot (trgt, a, b) =
            if isBlasSupported then
                use x = BLAS.GetVector (a, isSource=true, isTarget=false)
                use y = BLAS.GetVector (b, isSource=true, isTarget=false)
                BLAS.Invoke<'T, unit>
                    (singleFn=(fun () -> 
                        let trgt = trgt |> box :?> Tensor<single>
                        trgt.Value <- BLAS.cblas_sdot (x.Size, x.Ptr, x.Inc, y.Ptr, y.Inc)),
                        doubleFn=(fun () -> 
                        let trgt = trgt |> box :?> Tensor<double>
                        trgt.Value <- BLAS.cblas_ddot (x.Size, x.Ptr, x.Inc, y.Ptr, y.Inc)))
            else
                trgt.FillSumAxis 0 (a * b)

        member this.MatVecDot (trgt, a, b) =
            if isBlasSupported then
                use a = BLAS.GetMatrix (a, isSource=true, isTarget=false, canTranspose=true)
                use x = BLAS.GetVector (b, isSource=true, isTarget=false)
                use y = BLAS.GetVector (trgt, isSource=false, isTarget=true)
                BLAS.Invoke<'T, unit>
                    (singleFn=(fun () -> BLAS.cblas_sgemv (BLAS.CBLAS_LAYOUT.CblasColMajor,
                                                           a.CTrans, a.Rows, a.Cols, 1.0f,
                                                           a.Ptr, a.Ld, x.Ptr, x.Inc,
                                                           0.0f, y.Ptr, y.Inc)),
                     doubleFn=(fun () -> BLAS.cblas_dgemv (BLAS.CBLAS_LAYOUT.CblasColMajor,
                                                           a.CTrans, a.Rows, a.Cols, 1.0,
                                                           a.Ptr, a.Ld, x.Ptr, x.Inc,
                                                           0.0, y.Ptr, y.Inc)))  
                y.FetchResult()
            else
                trgt.FillSumAxis 1 (a * Tensor.padLeft b)

        member this.MatMatDot (trgt, a, b) =
            if isBlasSupported then
                use a = BLAS.GetMatrix (a, isSource=true, isTarget=false, canTranspose=true)
                use b = BLAS.GetMatrix (b, isSource=true, isTarget=false, canTranspose=true)
                use c = BLAS.GetMatrix (trgt, isSource=false, isTarget=true, canTranspose=false)
                BLAS.Invoke<'T, unit>
                    (singleFn=(fun () -> BLAS.cblas_sgemm (BLAS.CBLAS_LAYOUT.CblasColMajor,
                                                           a.CTrans, b.CTrans, a.OpRows, b.OpCols, a.OpCols, 
                                                           1.0f, a.Ptr, a.Ld, b.Ptr, b.Ld,
                                                           0.0f, c.Ptr, c.Ld)),
                     doubleFn=(fun () -> BLAS.cblas_dgemm (BLAS.CBLAS_LAYOUT.CblasColMajor,
                                                           a.CTrans, b.CTrans, a.OpRows, b.OpCols, a.OpCols, 
                                                           1.0, a.Ptr, a.Ld, b.Ptr, b.Ld,
                                                           0.0, c.Ptr, c.Ld)))              
                c.FetchResult()
            else
                trgt.FillSumAxis 1 (Tensor.padRight a * Tensor.padLeft b)

        member this.BatchedMatMatDot (trgt, a, b) =
            if isBlasSupported then
                use a = BLAS.GetMatrix (a, isSource=true, isTarget=false, canTranspose=true)
                use b = BLAS.GetMatrix (b, isSource=true, isTarget=false, canTranspose=true)
                use c = BLAS.GetMatrix (trgt, isSource=false, isTarget=true, canTranspose=false)
                BLAS.Invoke<'T, unit>
                    (singleFn=(fun () -> BLAS.cblas_sgemm_batch (BLAS.CBLAS_LAYOUT.CblasColMajor,
                                                                 [|a.CTrans|], [|b.CTrans|], 
                                                                 [|a.OpRows|], [|b.OpCols|], [|a.OpCols|], [|1.0f|],
                                                                 a.Ptrs, [|a.Ld|], b.Ptrs, [|b.Ld|],
                                                                 [|0.0f|], c.Ptrs, [|c.Ld|],
                                                                 1L, [|a.BatchSize|])),
                     doubleFn=(fun () -> BLAS.cblas_dgemm_batch (BLAS.CBLAS_LAYOUT.CblasColMajor,
                                                                 [|a.CTrans|], [|b.CTrans|], 
                                                                 [|a.OpRows|], [|b.OpCols|], [|a.OpCols|], [|1.0|],
                                                                 a.Ptrs, [|a.Ld|], b.Ptrs, [|b.Ld|],
                                                                 [|0.0|], c.Ptrs, [|c.Ld|],
                                                                 1L, [|a.BatchSize|])))
                c.FetchResult()
            else
                trgt.FillSumAxis 2 (a.[*, *, *, NewAxis] * b.[*, NewAxis, *, *])

        member this.BatchedInvert (trgt, src) =
            if not isBlasSupported then
                raise (NotImplementedException("this operation is only supported for floating point numbers"))

            // inversion is done in place, so we have to copy first if trgt and src are different
            if not (trgt = src) then
                (this :> ITensorBackend<_>).Copy (trgt, src)

            let size = trgt.Shape.[trgt.NDims-2]
            use a = BLAS.GetMatrix (trgt, isSource=true, isTarget=true, canTranspose=true)

            // loop over batch 
            for s in 0 .. int a.BatchSize - 1 do
                // compute LU factorization
                let ipiv : BLAS.lapack_int[] = Array.zeroCreate (int32 size)
                let info =
                    BLAS.Invoke<'T, BLAS.lapack_int> 
                        (singleFn=(fun () -> BLAS.LAPACKE_sgetrf (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Cols, a.Ptrs.[s], a.Ld, ipiv)),
                            doubleFn=(fun () -> BLAS.LAPACKE_dgetrf (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Cols, a.Ptrs.[s], a.Ld, ipiv)))
                if info < 0L then failwithf "LAPACK argument error %d" info
                if info > 0L then raise (SingularMatrixError "cannot invert singular matrix")

                // compute matrix inverse
                let info =
                    BLAS.Invoke<'T, BLAS.lapack_int> 
                        (singleFn=(fun () -> BLAS.LAPACKE_sgetri (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Ptrs.[s], a.Ld, ipiv)),
                            doubleFn=(fun () -> BLAS.LAPACKE_dgetri (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Ptrs.[s], a.Ld, ipiv)))
                if info < 0L then failwithf "LAPACK argument error %d" info
                if info > 0L then raise (SingularMatrixError "cannot invert singular matrix")
            a.FetchResult()

        member this.BatchedSVD (trgtS, trgtUV, src) =
            if not isBlasSupported then
                raise (NotImplementedException("this operation is only supported for floating point numbers"))

            let src = src.Copy(order=ColumnMajor) // LAPACK destorys src
            let batchShp, M, N, K = Tensor.SVDSizes src

            use a = BLAS.GetMatrix (src, isSource=true, isTarget=false, canTranspose=false)
            use s = BLAS.GetVector (trgtS, isSource=false, isTarget=true, reqLinear=true)
            match trgtUV with
            | Some (trgtU, trgtV) ->
                use u = BLAS.GetMatrix (trgtU, isSource=false, isTarget=true, canTranspose=false)
                use vt = BLAS.GetMatrix (trgtV.T, isSource=false, isTarget=true, canTranspose=false)
                for smpl in 0 .. int a.BatchSize - 1 do
                    let info =
                        BLAS.Invoke<'T, BLAS.lapack_int>
                            (singleFn=(fun() -> BLAS.LAPACKE_sgesdd (BLAS.LAPACK_COL_MAJOR, 'A', M, N, a.Ptrs.[smpl], a.Ld, s.Ptrs.[smpl], u.Ptrs.[smpl], u.Ld, vt.Ptrs.[smpl], vt.Ld)),
                             doubleFn=(fun() -> BLAS.LAPACKE_dgesdd (BLAS.LAPACK_COL_MAJOR, 'A', M, N, a.Ptrs.[smpl], a.Ld, s.Ptrs.[smpl], u.Ptrs.[smpl], u.Ld, vt.Ptrs.[smpl], vt.Ld)))
                    if info < 0L then failwithf "LAPACK argument error %d" info
                    if info > 0L then failwithf "SVD did not converge: %d" info
            | None -> 
                for smpl in 0 .. int a.BatchSize - 1 do
                    let info =
                        BLAS.Invoke<'T, BLAS.lapack_int>
                            (singleFn=(fun() -> BLAS.LAPACKE_sgesdd (BLAS.LAPACK_COL_MAJOR, 'N', M, N, a.Ptrs.[smpl], a.Ld, s.Ptrs.[smpl], nativeint 0, 1L, nativeint 0, 1L)),
                             doubleFn=(fun() -> BLAS.LAPACKE_dgesdd (BLAS.LAPACK_COL_MAJOR, 'N', M, N, a.Ptrs.[smpl], a.Ld, s.Ptrs.[smpl], nativeint 0, 1L, nativeint 0, 1L)))
                    if info < 0L then failwithf "LAPACK argument error %d" info
                    if info > 0L then failwithf "SVD did not converge: %d" info

        member this.SymmetricEigenDecomposition (part, eigVals, eigVecs, src) =
            if not isBlasSupported then
                raise (NotImplementedException("this operation is only supported for floating point numbers"))

            let size = src.Shape.[0]
            let part = 
                match part with
                | UpperPart -> 'U'
                | LowerPart -> 'L'
            if not (eigVecs = src) then
                (this :> ITensorBackend<_>).Copy (eigVecs, src)

            use a = BLAS.GetMatrix (eigVecs, isSource=true, isTarget=true, canTranspose=false)
            use w = BLAS.GetVector (eigVals, isSource=false, isTarget=true)
            let info = 
                BLAS.Invoke<'T, BLAS.lapack_int> 
                    (singleFn=(fun () -> BLAS.LAPACKE_ssyevd (BLAS.LAPACK_COL_MAJOR, 'V', part, a.Rows, a.Ptr, a.Ld, w.Ptr)),
                     doubleFn=(fun () -> BLAS.LAPACKE_dsyevd (BLAS.LAPACK_COL_MAJOR, 'V', part, a.Rows, a.Ptr, a.Ld, w.Ptr)))
            if info < 0L then failwithf "LAPACK argument error %d" info
            if info > 0L then raise (SingularMatrixError "cannot compute eigen decomposition of singular matrix")
            a.FetchResult()
            w.FetchResult()

        member this.GetEnumerator() : IEnumerator<'T> = 
            let s = seq {
                let mutable pos = PosIter32 this.FastLayout
                while pos.Active do
                    yield this.Data.[pos.Addr]
                    pos.MoveNext()
            }
            s.GetEnumerator()

        member this.GetEnumerator() : System.Collections.IEnumerator =
            (this :> IEnumerable<'T>).GetEnumerator() :> System.Collections.IEnumerator


/// Factory for host tensors.
and TensorHostDevice private () =
    inherit BaseTensorDevice()
    static member Instance = TensorHostDevice () 

    override this.Id = "Host"
    override this.Create nElems =
        TensorHostStorage<_> nElems :> ITensorStorage<_>
    override this.Zeroed = true


module internal HostTensorHelpers = 

    let ensureCAndOffsetFree (x: Tensor<'T>) =
        if x.Dev <> (TensorHostDevice.Instance :> ITensorDevice) then
            let msg = sprintf "require a Host tensor but got a %s tensor" x.Dev.Id
            raise (StorageMismatch msg)
        if TensorLayout.isC x.Layout && x.Layout.Offset = 0L then x
        else Tensor.copy (x, order=RowMajor)

type private HDFFuncs =

    static member Write<'T> (hdf5: HDF5, path: string, x: Tensor<'T>) =
        let x = HostTensorHelpers.ensureCAndOffsetFree x
        let storage = x.Storage :?> TensorHostStorage<'T>
        hdf5.Write (path, storage.Data, Tensor.shape x)

    static member Read<'T> (hdf5: HDF5, name: string) =
        let (data: 'T []), shape = hdf5.Read (name)
        Tensor<'T> (TensorLayout.newC shape, TensorHostStorage<'T> data)         



/// Host tensor functions.
module HostTensor =

    /// Tensor located on host using a .NET array as storage.
    let Dev = TensorHostDevice.Instance :> ITensorDevice

    let transfer x = Tensor.transfer Dev x

    let empty<'T> = Tensor.empty<'T> Dev

    let zeros<'T> = Tensor.zeros<'T> Dev 

    let ones<'T> = Tensor.ones<'T> Dev

    let falses = Tensor.falses Dev

    let trues = Tensor.trues Dev

    let scalar<'T> = Tensor.scalar<'T> Dev

    let init<'T> = Tensor.init<'T> Dev

    let filled<'T> = Tensor.filled<'T> Dev

    let identity<'T> = Tensor.identity<'T> Dev

    let counting = Tensor.counting Dev

    let inline arange start incr stop = 
        Tensor.arange Dev start incr stop

    let inline linspace start stop nElems = 
        Tensor.linspace Dev start stop nElems
  
    /// Creates a one-dimensional Tensor using the specified data.
    /// The data is referenced, not copied.
    let usingArray (data: 'T []) =
        let shp = [data.LongLength]
        let layout = TensorLayout.newC shp
        let storage = TensorHostStorage<'T> (data)
        Tensor<'T> (layout, storage) 

    /// Creates a one-dimensional Tensor using the specified data.
    /// The data is copied.
    let ofArray (data: 'T []) =
        let shp = [Array.length data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0]])

    /// Creates a two-dimensional Tensor using the specified data. 
    /// The data is copied.
    let ofArray2D (data: 'T [,]) =
        let shp = [Array2D.length1 data; Array2D.length2 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1]])

    /// Creates a three-dimensional Tensor using the specified data. 
    /// The data is copied.
    let ofArray3D (data: 'T [,,]) =
        let shp = [Array3D.length1 data; Array3D.length2 data; Array3D.length3 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2]])

    /// Creates a four-dimensional Tensor using the specified data. 
    /// The data is copied.
    let ofArray4D (data: 'T [,,,]) =
        let shp = [Array4D.length1 data; Array4D.length2 data; 
                   Array4D.length3 data; Array4D.length4 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2], int32 idx.[3]])

    /// Creates a one-dimensional Tensor using the specified sequence.       
    let ofSeq (data: 'T seq) =
        data |> Array.ofSeq |> usingArray

    /// Creates a one-dimensional Tensor using the specified sequence and shape.       
    let ofSeqWithShape shape (data: 'T seq) =
        let nElems = shape |> List.fold (*) 1L
        data |> Seq.take (int32 nElems) |> ofSeq |> Tensor.reshape shape

    /// Creates a one-dimensional Tensor using the specified list.       
    let ofList (data: 'T list) =
        data |> Array.ofList |> usingArray

    /// Creates a two-dimensional Tensor using the specified list of lists.       
    let ofList2D (data: 'T list list) =
        data |> array2D |> ofArray2D

    /// Creates an Array from the data in this Tensor. The data is copied.
    let toArray (ary: Tensor<_>) =
        if Tensor.nDims ary <> 1 then failwith "Tensor must have 1 dimension"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array.init shp.[0] (fun i0 -> ary.[[int64 i0]])

    /// Creates an Array2D from the data in this Tensor. The data is copied.
    let toArray2D (ary: Tensor<_>) =
        if Tensor.nDims ary <> 2 then failwith "Tensor must have 2 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array2D.init shp.[0] shp.[1] (fun i0 i1 -> ary.[[int64 i0; int64 i1]])

    /// Creates an Array3D from the data in this Tensor. The data is copied.
    let toArray3D (ary: Tensor<_>) =
        if Tensor.nDims ary <> 3 then failwith "Tensor must have 3 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array3D.init shp.[0] shp.[1] shp.[2] (fun i0 i1 i2 -> ary.[[int64 i0; int64 i1; int64 i2]])
       
    /// Creates an Array4D from the data in this Tensor. The data is copied.
    let toArray4D (ary: Tensor<_>) =
        if Tensor.nDims ary <> 4 then failwith "Tensor must have 4 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array4D.init shp.[0] shp.[1] shp.[2] shp.[3] (fun i0 i1 i2 i3 -> ary.[[int64 i0; int64 i1; int64 i2; int64 i3]])

    /// Creates a list from the data in this Tensor. The data is copied.
    let toList (ary: Tensor<_>) =
        ary |> toArray |> Array.toList

    /// Writes the given host tensor into the HDF5 file under the given path.
    let write (hdf5: HDF5) (path: string) (x: ITensor) =
        callGeneric<HDFFuncs, unit> "Write" [x.DataType] (hdf5, path, x)

    /// Reads the tensor of data type 'T with the given path from an HDF5 file.
    let read<'T> (hdf5: HDF5) (path: string) : Tensor<'T> =
        HDFFuncs.Read (hdf5, path)

    /// Reads the tensor with the given path from an HDF5 file and returns it
    /// as an ITensor with the data type as stored in the HDF5 file.
    let readUntyped (hdf5: HDF5) (path: string) = 
        let dataType = hdf5.GetDataType path
        callGeneric<HDFFuncs, ITensor> "Read" [dataType] (hdf5, path)

    /// Creates a tensor of given shape filled with random integer numbers between
    /// minValue and maxValue.
    let randomInt (rnd: Random) (minValue, maxValue) shp =
        rnd.Seq (minValue, maxValue) 
        |> ofSeqWithShape shp  

    /// Creates a tensor of given shape filled with random floating-point numbers 
    /// uniformly placed between minValue and maxValue.
    let randomUniform (rnd: Random) (minValue: 'T, maxValue: 'T) shp =
        rnd.SeqDouble (conv<float> minValue, conv<float> maxValue) 
        |> Seq.map conv<'T>
        |> ofSeqWithShape shp    

    /// Creates a tensor of given shape filled with random samples from a normal
    /// distribution with the specified mean and variance.
    let randomNormal (rnd: Random) (mean: 'T, variance: 'T) shp =
        rnd.SeqNormal (conv<float> mean, conv<float> variance) 
        |> Seq.map conv<'T>
        |> ofSeqWithShape shp       
    
