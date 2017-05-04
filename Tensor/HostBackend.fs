﻿namespace ArrayNDNS

open System
open System.Reflection
open System.Numerics
open System.Threading.Tasks
open System.Linq.Expressions
open System.Collections.Generic
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open Basics


/// BLAS / LAPACK library imports
module private BLAS =
    type lapack_int = int64
    let LAPACK_ROW_MAJOR = 101
    let LAPACK_COL_MAJOR = 102

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_ssyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dsyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)

type private BLAS =
    /// Call BLAS/LAPACK function depending on data type.
    static member invoke<'T, 'R> (?singleFn: unit -> 'R, 
                                  ?doubleFn: unit -> 'R,
                                  ?int32Fn: unit -> 'R,
                                  ?int64Fn: unit -> 'R) : 'R =
        match typeof<'T> with
        | t when t=typeof<single> && singleFn.IsSome -> singleFn.Value () 
        | t when t=typeof<double> && doubleFn.IsSome -> doubleFn.Value () 
        | t when t=typeof<int32> && int32Fn.IsSome -> int32Fn.Value () 
        | t when t=typeof<int64> && int64Fn.IsSome -> int64Fn.Value () 
        | t -> failwithf "unsupported data type for BLAS operation: %A" t


module private Tools =
    let inline checkedInt layout (x: int64) =
        if int64 FSharp.Core.int.MinValue <= x && x <= int64 FSharp.Core.int.MaxValue then
            int x
        else failwithf "cannot convert tensor layout %A to 32-bit integer" layout

/// Fast layout operations.
[<Struct>]
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
            assert (0 <= pos.[d] && pos.[d] < this.Shape.[d])
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
        assert (fl.IsPosValid startPos)
        assert (0 <= fromDim && fromDim < fl.NDims)
        assert (0 <= toDim && toDim < fl.NDims)
        let active = 
            fl.Shape 
            |> Array.indexed 
            |> Array.forall (fun (d, s) -> 
                if fromDim <= d && d <= toDim then fl.Shape.[d] > 0
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
        assert (this.Active)

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
    static let tso = myAsm.GetType("ArrayNDNS.Operators", true)

    static let a = Expression.Parameter(typeof<'T>, "a")
    static let b = Expression.Parameter(typeof<'T>, "b")
    static let c = Expression.Parameter(typeof<'TC>, "c")
    static let cond = Expression.Parameter(typeof<bool>, "cond")

    member val ConvertFunc = 
        Expression.Lambda<Func<'TC, 'T>>(Expression.Convert(c, typeof<'T>), c).Compile()
    member inline this.Convert cv = this.ConvertFunc.Invoke(cv)

    member val UnaryPlusFunc = 
        Expression.Lambda<Func<'T, 'T>>(Expression.UnaryPlus(a), a).Compile()
    member inline this.UnaryPlus av = this.UnaryPlusFunc.Invoke(av)

    member val UnaryMinusFunc = 
        Expression.Lambda<Func<'T, 'T>>(Expression.Negate(a), a).Compile()
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
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Add(a, b), a, b).Compile()
    member inline this.Add av bv = this.AddFunc.Invoke(av, bv)
        
    member val SubtractFunc = 
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Subtract(a, b), a, b).Compile()
    member inline this.Subtract av bv = this.SubtractFunc.Invoke(av, bv)

    member val MultiplyFunc = 
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Multiply(a, b), a, b).Compile()
    member inline this.Multiply av bv = this.MultiplyFunc.Invoke(av, bv)

    member val DivideFunc = 
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Divide(a, b), a, b).Compile()
    member inline this.Divide av bv = this.DivideFunc.Invoke(av, bv)

    member val ModuloFunc = 
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Modulo(a, b), a, b).Compile()
    member inline this.Modulo av bv = this.ModuloFunc.Invoke(av, bv)

    member val PowerFunc = 
        // note: power is currently significantly slower than other operations
        let m = fso.GetMethod("op_Exponentiation").MakeGenericMethod (typeof<'T>, typeof<'T>)        
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Call(m, a, b), a, b).Compile()
    member inline this.Power av bv = this.PowerFunc.Invoke(av, bv)

    member val MaxFunc = 
        let cond = Expression.GreaterThan(a, b)
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Condition(cond, a, b), a, b).Compile()
    member inline this.Max av bv = this.MaxFunc.Invoke(av, bv)

    member val MinFunc = 
        let cond = Expression.LessThan(a, b)
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Condition(cond, a, b), a, b).Compile()
    member inline this.Min av bv = this.MinFunc.Invoke(av, bv)

    member val IsFiniteFunc : ('T -> bool) =
        match typeof<'T> with
        | t when t=typeof<single> -> 
            unbox (fun (v: single) -> not (System.Single.IsInfinity v || System.Single.IsNaN v))
        | t when t=typeof<double> -> 
            unbox (fun (v: double) -> not (System.Double.IsInfinity v || System.Double.IsNaN v))
        | _ -> (fun _ -> true)
    member inline this.IsFinite av = this.IsFiniteFunc av

    member val EqualFunc = 
        Expression.Lambda<Func<'T, 'T, bool>>(Expression.Equal(a, b), a, b).Compile()
    member inline this.Equal av bv = this.EqualFunc.Invoke(av, bv)

    member val NotEqualFunc = 
        Expression.Lambda<Func<'T, 'T, bool>>(Expression.NotEqual(a, b), a, b).Compile()
    member inline this.NotEqual av bv = this.NotEqualFunc.Invoke(av, bv)

    member val LessFunc = 
        Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThan(a, b), a, b).Compile()
    member inline this.Less av bv = this.LessFunc.Invoke(av, bv)

    member val LessOrEqualFunc = 
        Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThanOrEqual(a, b), a, b).Compile()
    member inline this.LessOrEqual av bv = this.LessOrEqualFunc.Invoke(av, bv)

    member val GreaterFunc = 
        Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThan(a, b), a, b).Compile()
    member inline this.Greater av bv = this.GreaterFunc.Invoke(av, bv)

    member val GreaterOrEqualFunc = 
        Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThanOrEqual(a, b), a, b).Compile()
    member inline this.GreaterOrEqual av bv = this.GreaterOrEqualFunc.Invoke(av, bv)

    member val IfThenElseFunc =
        Expression.Lambda<Func<bool, 'T, 'T, 'T>>(Expression.Condition(cond, a, b), cond, a, b).Compile()
    member inline this.IfThenElse condv ifTrue ifFalse = this.IfThenElseFunc.Invoke(condv, ifTrue, ifFalse)


/// Generic scalar operation primitives.
module internal ScalarPrimitives = 
    let private instances = Dictionary<Type * Type, obj>()
    let For<'T, 'TC> () =
        let types = typeof<'T>, typeof<'TC>
        match instances.TryFind types with
        | Some inst -> inst :?> ScalarPrimitives<'T, 'TC>
        | None ->
            let inst = ScalarPrimitives<'T, 'TC> ()
            instances.Add (types, inst)
            inst
        

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
                    trgt.Data.[trgtPosIter.Addr ] <- scalarOp [||]
                elif isIndexed then
                    for d in 0 .. nd - 1 do
                        pos64.[d] <- int64 trgtPosIter.Pos.[d]
                    for i in 0 .. shape.[nd-1] - 1 do
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
                        src3Addr <- src3Addr + src2.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                else
                    for i in 0 .. shape.[nd-1] - 1 do
                        trgt.Data.[trgtAddr] <- scalarOp [||] src1.Data.[src1Addr] src2.Data.[src2Addr] src3.Data.[src3Addr]
                        trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        src2Addr <- src2Addr + src2.FastLayout.Stride.[nd-1]
                        src3Addr <- src3Addr + src2.FastLayout.Stride.[nd-1]
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
        assert (srcs |> Array.forall (fun src -> 
            List.ofArray trgt.FastLayout.Shape = List.ofArray src.FastLayout.Shape))
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
                                        initial: 'TS,
                                        isIndexed: bool, useThreads: bool) =        
        let nd = src1.FastLayout.NDims
        let shape = src1.FastLayout.Shape
        assert (trgt.FastLayout.NDims = nd-1)
        assert (List.ofArray trgt.FastLayout.Shape = List.ofArray src1.FastLayout.Shape.[0 .. nd-2])
                              
        let inline loops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let pos64 = Array.zeroCreate nd
            while trgtPosIter.Active do
                let mutable src1Addr = src1PosIter.Addr
                let mutable state = initial
                if nd = 0 then
                    trgt.Data.[trgtPosIter.Addr] <- foldOp [||] state src1.Data.[src1Addr] |> extractOp
                elif isIndexed then
                    for d in 0 .. nd - 1 do
                        pos64.[d] <- int64 trgtPosIter.Pos.[d]
                    pos64.[nd-1] <- 0L
                    for i in 0 .. shape.[nd-1] - 1 do
                        state <- foldOp pos64 state src1.Data.[src1Addr]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                        pos64.[nd-1] <- pos64.[nd-1] + 1L
                    trgt.Data.[trgtPosIter.Addr] <- extractOp state
                else
                    let mutable value = initial
                    for i in 0 .. shape.[nd-1] - 1 do
                        value <- foldOp [||] value src1.Data.[src1Addr]
                        src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                    trgt.Data.[trgtPosIter.Addr] <- extractOp value
                trgtPosIter.MoveNext()
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
        let inline op pos a b = p.Max a b
        ScalarOps.ApplyBinaryOp (op, trgt, src1, src2, isIndexed=false, useThreads=true)

    static member MinElemwise (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op pos a b = p.Min a b
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
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=conv<'T> 0, isIndexed=false, useThreads=true)     

    static member ProductLastAxis (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (res: 'T) (v: 'T) = p.Multiply res v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=conv<'T> 1, isIndexed=false, useThreads=true)  

    static member MaxLastAxis (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (res: 'T) (v: 'T) = p.Max res v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=minValue<'T>, isIndexed=false, useThreads=true)  

    static member MinLastAxis (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (res: 'T) (v: 'T) = p.Min res v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=maxValue<'T>, isIndexed=false, useThreads=true)  

    static member AllLastAxis (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>) =
        let inline op (srcIdx: int64[]) (res: bool) (v: bool) = res && v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=true, isIndexed=false, useThreads=true)  

    static member AnyLastAxis (trgt: DataAndLayout<bool>, src1: DataAndLayout<bool>) =
        let inline op (srcIdx: int64[]) (res: bool) (v: bool) = res || v
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=false, isIndexed=false, useThreads=true)  

    static member ArgMaxLastAxis (trgt: DataAndLayout<int64>, src1: DataAndLayout<'T>) =
        let nd = src1.FastLayout.NDims
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (maxPos, maxVal) (v: 'T) = 
            if p.Greater v maxVal then srcIdx.[nd-1], v
            else maxPos, maxVal
        ScalarOps.ApplyAxisFold (op, fst, trgt, src1, initial=(-1L, minValue<'T>), 
                                 isIndexed=true, useThreads=true)     

    static member ArgMinLastAxis (trgt: DataAndLayout<int64>, src1: DataAndLayout<'T>) =
        let nd = src1.FastLayout.NDims
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (minPos, minVal) (v: 'T) = 
            if p.Less v minVal then srcIdx.[nd-1], v
            else minPos, minVal
        ScalarOps.ApplyAxisFold (op, fst, trgt, src1, initial=(-1L, maxValue<'T>), 
                                 isIndexed=true, useThreads=true)     


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
        VectorOps.Method<BinaryDelegate<'T>>("MaxElemwise").Invoke (trgt, src1, src2) 

    static member MinElemwise (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("MinElemwise").Invoke (trgt, src1, src2) 

    static member CanUse (trgt: DataAndLayout<'T>, ?src1: DataAndLayout<'T1>, ?src2: DataAndLayout<'T2>) =
        let nd = trgt.FastLayout.NDims
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


/// pinned .NET managed memory (wraps a GCHandle)
type PinnedMemory (gcHnd: GCHandle, size: int64) =       

    /// pointer to storage array 
    member this.Ptr = gcHnd.AddrOfPinnedObject()

    /// size of storage array in bytes
    member this.Size = size

    interface IDisposable with
        member this.Dispose() = gcHnd.Free()


/// Information for calling BLAS/LAPACK routines.
type private BlasInfo (memory: PinnedMemory,
                       offset: nativeint,
                       rows:   int64,
                       cols:   int64,
                       ld:     int64) =

    member this.Ptr  : nativeint       = memory.Ptr + offset
    member this.Rows : BLAS.lapack_int = rows
    member this.Cols : BLAS.lapack_int = cols
    member this.Ld   : BLAS.lapack_int = ld

    interface IDisposable with
        member this.Dispose() = (memory :> IDisposable).Dispose()


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
        member this.Id = "Host"
        member this.Backend layout =
            TensorHostBackend<'T> (layout, this) :> ITensorBackend<_>
        member this.Factory = 
            TensorHostStorageFactory.Instance :> ITensorStorageFactory

    override this.Equals other =
        match other with
        | :? TensorHostStorage<'T> as os ->
            LanguagePrimitives.PhysicalEquality this.Data os.Data
        | _ -> false            

    override this.GetHashCode () =
        RuntimeHelpers.GetHashCode data

/// Backend for host tensors.
and TensorHostBackend<'T> (layout: TensorLayout, storage: TensorHostStorage<'T>) =

    /// fast layout
    member val internal FastLayout = FastLayout32 layout

    /// underlying TensorHostStorate<'T>
    member this.Storage = storage

    /// underlying data array
    member val Data = storage.Data

    /// data array and fast layout
    member inline internal this.DataAndLayout = 
        {Data=this.Data; FastLayout=this.FastLayout}
            
    ///// C++ type name
    //member this.CPPType = 
    //    let dims = TensorLayout.nDims layout
    //    let shp = TensorLayout.shape layout
    //    let str = TensorLayout.stride layout
    //    let ofst = TensorLayout.offset layout
    //    let cppDataType = Util.cppType this.DataType
    //    let shapeStr = 
    //        if dims = 0 then "" 
    //        else "<" + (shp |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
    //    let strideStr = 
    //        "<" + ((ofst :: str) |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
    //    sprintf "ArrayND%dD<%s, ShapeStatic%dD%s, StrideStatic%dD%s>" 
    //        dims cppDataType dims shapeStr dims strideStr            

    static member internal GetDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>) =
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Backend :?> TensorHostBackend<'TA>).DataAndLayout 

    static member internal ElemwiseDataAndLayout (t: Tensor<'T>) =
        // try to find stride 1 dimension and move it to the back
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout        

    static member internal ElemwiseDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>) =
        // try to find stride 1 dimension and move it to the back
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Backend :?> TensorHostBackend<'TA>).DataAndLayout 

    static member internal ElemwiseDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>, b: Tensor<'TB>) =
        // try to find stride 1 dimension and move it to the back
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Backend :?> TensorHostBackend<'TA>).DataAndLayout,
        (b.Backend :?> TensorHostBackend<'TB>).DataAndLayout 

    static member internal ElemwiseDataAndLayout (t: Tensor<'T>, a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) =
        // try to find stride 1 dimension and move it to the back
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Backend :?> TensorHostBackend<'TA>).DataAndLayout,
        (b.Backend :?> TensorHostBackend<'TB>).DataAndLayout,
        (c.Backend :?> TensorHostBackend<'TC>).DataAndLayout 

    /// Returns a BlasInfo that exposes the transpose of the specfied matrix to BLAS
    /// (in column-major order).
    static member private GetTransposedBlas (mat: Tensor<'T>, allowCopy: bool) =
        if mat.NDims <> 2 then failwithf "BLAS call requires a matrix but got shape %A" mat.Shape
        if not (mat.Shape.[0] > 0L && mat.Shape.[1] > 0L) then 
            failwithf "BLAS call requires a non-empty matrix but got shape %A" mat.Shape
        let str = mat.Layout.Stride
        if str.[0] >= 1L && str.[0] >= mat.Shape.[1] && str.[1] = 1L then
            let storage = mat.Storage :?> TensorHostStorage<'T>
            new BlasInfo (storage.Pin(), nativeint (mat.Layout.Offset * sizeof64<'T>),
                          mat.Shape.[1], mat.Shape.[0], str.[0])
        elif allowCopy then TensorHostBackend<_>.GetTransposedBlas (Tensor.copy mat, false)
        else 
            let msg =
                sprintf "matrix with shape %A and strides %A is incompatible with BLAS/LAPACK"
                        mat.Shape mat.Layout.Stride
            raise (StrideMismatch msg)

    interface ITensorBackend<'T> with
        member this.Item 
            with get idx = this.Data.[this.FastLayout.Addr idx]
            and set idx value = this.Data.[this.FastLayout.Addr idx] <- value

        member this.FillConst (value, trgt) =
            let trgt= TensorHostBackend<_>.ElemwiseDataAndLayout (trgt)
            if VectorOps.CanUse (trgt) then VectorOps.Fill (value, trgt)
            else ScalarOps.Fill (value, trgt)

        member this.Copy (trgt, src) =
            if TensorLayout.hasContiguousMemory trgt.Layout &&
               TensorLayout.hasContiguousMemory src.Layout &&
               trgt.Layout.Stride = src.Layout.Stride then
                // use array block copy for contiguous memory block
                let trgt, src = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, src)
                Array.Copy (src.Data, src.FastLayout.Offset, 
                            trgt.Data, trgt.FastLayout.Offset, trgt.FastLayout.NElems)
            else 
                let trgt, src = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, src)
                if VectorOps.CanUse (trgt, src) then VectorOps.Copy (trgt, src)
                else ScalarOps.Copy (trgt, src)

        member this.Convert (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.Convert (trgt, a)

        member this.Fill (fn, trgt, useThreads) = 
            let trgt = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt)
            let inline scalarOp idx = fn ()
            ScalarOps.ApplyNoaryOp (scalarOp, trgt, isIndexed=false, useThreads=useThreads)

        member this.FillIndexed (fn, trgt, useThreads) = 
            let trgt = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt)
            ScalarOps.ApplyNoaryOp (fn, trgt, isIndexed=true, useThreads=useThreads)

        member this.Map (fn, trgt, a, useThreads) = 
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            let inline scalarOp idx av = fn av
            ScalarOps.ApplyUnaryOp (scalarOp, trgt, a, isIndexed=false, useThreads=useThreads)

        member this.MapIndexed (fn, trgt, a, useThreads) = 
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.ApplyUnaryOp (fn, trgt, a, isIndexed=true, useThreads=useThreads)

        member this.Map2 (fn, trgt, a, b, useThreads) = 
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
            let inline scalarOp idx av bv = fn av bv
            ScalarOps.ApplyBinaryOp (scalarOp, trgt, a, b, isIndexed=false, useThreads=useThreads)

        member this.MapIndexed2 (fn, trgt, a, b, useThreads) =
            let trgt, a, b = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a, b)
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
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            let inline foldOp idx state xv = fn state xv
            ScalarOps.ApplyAxisFold (foldOp, id, trgt, a, initial, isIndexed=false, useThreads=useThreads)

        member this.FoldLastAxisIndexed (fn, initial, trgt, a, useThreads) = 
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            ScalarOps.ApplyAxisFold (fn, id, trgt, a, initial, isIndexed=true, useThreads=useThreads)

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
            ()

        member this.MatVecDot (trgt, a, b) =
            ()

        member this.MatMatDot (trgt, a, b) =
            ()

        member this.BatchedMatMatDot (trgt, a, b) =
            ()

        member this.Invert (trgt, src) =
            // inversion is done in place, so we have to copy first if trgt and src are different
            if not (trgt = src) then
                (this :> ITensorBackend<_>).Copy (trgt, src)

            // iterate over all batch dimensions
            let batchShp = trgt.Shape.[0 .. trgt.NDims-3]
            for batchIdx in TensorLayout.allIdxOfShape batchShp do
                let batchRng = batchIdx |> List.map RngElem
                let rng = batchRng @ [RngAll; RngAll]                  
                let aAry = trgt.[rng]

                // compute LU factorization
                use a = TensorHostBackend<_>.GetTransposedBlas (aAry, allowCopy=false)
                let ipiv : BLAS.lapack_int[] = Array.zeroCreate (int32 aAry.Shape.[0])
                let info =
                    BLAS.invoke<'T, BLAS.lapack_int> 
                        (singleFn=(fun () -> BLAS.LAPACKE_sgetrf (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Cols, a.Ptr, a.Ld, ipiv)),
                         doubleFn=(fun () -> BLAS.LAPACKE_dgetrf (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Cols, a.Ptr, a.Ld, ipiv)))
                if info < 0L then failwithf "LAPACK argument error %d" info
                if info > 0L then raise (SingularMatrixError "cannot invert singular matrix")

                // compute matrix inverse
                let info =
                    BLAS.invoke<'T, BLAS.lapack_int> 
                        (singleFn=(fun () -> BLAS.LAPACKE_sgetri (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Ptr, a.Ld, ipiv)),
                         doubleFn=(fun () -> BLAS.LAPACKE_dgetri (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Ptr, a.Ld, ipiv)))
                if info < 0L then failwithf "LAPACK argument error %d" info
                if info > 0L then raise (SingularMatrixError "cannot invert singular matrix")

        member this.SymmetricEigenDecomposition (eigVals, eigVecs, src) =
            let size = src.Shape.[0]
            (this :> ITensorBackend<_>).Copy (eigVecs, src)
            let eigVals = eigVals |> Tensor.reshape [1L; size]

            use a = TensorHostBackend.GetTransposedBlas (eigVecs.T, allowCopy=false)
            use w = TensorHostBackend.GetTransposedBlas (eigVals, allowCopy=false)
            let info = 
                BLAS.invoke<'T, BLAS.lapack_int> 
                    (singleFn=(fun () -> BLAS.LAPACKE_ssyevd (BLAS.LAPACK_COL_MAJOR, 'V', 'L', a.Rows, a.Ptr, a.Ld, w.Ptr)),
                     doubleFn=(fun () -> BLAS.LAPACKE_dsyevd (BLAS.LAPACK_COL_MAJOR, 'V', 'L', a.Rows, a.Ptr, a.Ld, w.Ptr)))
            if info < 0L then failwithf "LAPACK argument error %d" info
            if info > 0L then raise (SingularMatrixError "cannot compute eigen decomposition of singular matrix")

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
and TensorHostStorageFactory private () =
    static member Instance = TensorHostStorageFactory () 

    interface ITensorStorageFactory with
        member this.Create nElems = 
            TensorHostStorage<_> nElems :> ITensorStorage<_>
        member this.Zeroed = true
            

[<AutoOpen>]            
module HostTensorTypes =
    /// Tensor located on host using a .NET array as storage.
    let DevHost = TensorHostStorageFactory.Instance


/// Host tensor functions.
module HostTensor =

    let zeros<'T> = Tensor.zeros<'T> DevHost 

    let ones<'T> = Tensor.ones<'T> DevHost

    let falses = Tensor.falses DevHost

    let trues = Tensor.trues DevHost

    let scalar<'T> = Tensor.scalar<'T> DevHost

    let init<'T> = Tensor.init<'T> DevHost





