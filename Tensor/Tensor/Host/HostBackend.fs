namespace Tensor.Host

open System
open System.Reflection
open System.Numerics
open System.Threading.Tasks
open System.Linq.Expressions
open System.Collections.Generic
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

open Tensor
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
    static member internal GetDataAndLayout (t: ITensorFrontend<'T>) =
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout

    /// gets DataAndLayout for specified tensors
    static member internal GetDataAndLayout (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>) =
        (t.Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Backend :?> TensorHostBackend<'TA>).DataAndLayout 

    /// gets DataAndLayout for specified tensors
    static member internal GetDataAndLayout (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>, b: ITensorFrontend<'TB>) =
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
    static member internal ElemwiseDataAndLayout (t: ITensorFrontend<'T>) =        
        let tl, ls = TensorHostBackend<_>.ElemwiseLayouts (t.Layout, [])
        (t.Relayout(tl).Backend :?> TensorHostBackend<'T>).DataAndLayout        

    /// gets DataAndLayout for specified tensors, optimized for an element-wise operation
    static member internal ElemwiseDataAndLayout (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>) =
        let tl, ls = TensorHostBackend<_>.ElemwiseLayouts (t.Layout, [a.Layout])
        (t.Relayout(tl).Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Relayout(ls.[0]).Backend :?> TensorHostBackend<'TA>).DataAndLayout 

    /// gets DataAndLayout for specified tensors, optimized for an element-wise operation
    static member internal ElemwiseDataAndLayout (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>, b: ITensorFrontend<'TB>) =
        let tl, ls = TensorHostBackend<_>.ElemwiseLayouts (t.Layout, [a.Layout; b.Layout])
        (t.Relayout(tl).Backend :?> TensorHostBackend<'T>).DataAndLayout, 
        (a.Relayout(ls.[0]).Backend :?> TensorHostBackend<'TA>).DataAndLayout,
        (b.Relayout(ls.[1]).Backend :?> TensorHostBackend<'TB>).DataAndLayout 

    /// gets DataAndLayout for specified tensors, optimized for an element-wise operation
    static member internal ElemwiseDataAndLayout (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>, b: ITensorFrontend<'TB>, c: ITensorFrontend<'TC>) =
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

        member this.FillIncrementing (start, incr, trgt) =
            let trgt = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt)
            ScalarOps.FillIncrementing (start, incr, trgt)        

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
            
        member this.MaskedGet (trgt, src, masks) =
            let trgt = TensorHostBackend<_>.GetDataAndLayout trgt
            let masks = masks |> Array.map (Option.map TensorHostBackend<_>.GetDataAndLayout)
            let src = TensorHostBackend<_>.GetDataAndLayout src
            ScalarOps.MaskedGet (trgt, src, masks)

        member this.MaskedSet (trgt, masks, src) =
            let trgt = TensorHostBackend<_>.GetDataAndLayout trgt
            let masks = masks |> Array.map (Option.map TensorHostBackend<_>.GetDataAndLayout)
            let src = TensorHostBackend<_>.GetDataAndLayout src
            ScalarOps.MaskedSet (trgt, masks, src)        
            
        member this.TrueIndices (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.TrueIndices (trgt, src)    

        member this.CountTrueLastAxis (trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.CountTrueLastAxis (trgt, src)

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
            
        member this.FindLastAxis (value, trgt, src) =
            let trgt, src = TensorHostBackend<_>.GetDataAndLayout (trgt, src)
            ScalarOps.FindLastAxis (value, trgt, src)            

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
                (trgt :?> Tensor<'T>).FillSumAxis 0 ((a :?> Tensor<'T>) * (b :?> Tensor<'T>))

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
                (trgt :?> Tensor<'T>).FillSumAxis 1 ((a :?> Tensor<'T>) * Tensor.padLeft (b :?> Tensor<'T>))

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
                (trgt :?> Tensor<'T>).FillSumAxis 1 (Tensor.padRight (a :?> Tensor<'T>) * Tensor.padLeft (b :?> Tensor<'T>))

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
                (trgt :?> Tensor<'T>).FillSumAxis 2 ((a :?> Tensor<'T>).[*, *, *, NewAxis] * (b :?> Tensor<'T>).[*, NewAxis, *, *])

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
                if info > 0L then raise (SingularMatrixException "cannot invert singular matrix")

                // compute matrix inverse
                let info =
                    BLAS.Invoke<'T, BLAS.lapack_int> 
                        (singleFn=(fun () -> BLAS.LAPACKE_sgetri (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Ptrs.[s], a.Ld, ipiv)),
                            doubleFn=(fun () -> BLAS.LAPACKE_dgetri (BLAS.LAPACK_COL_MAJOR, a.Rows, a.Ptrs.[s], a.Ld, ipiv)))
                if info < 0L then failwithf "LAPACK argument error %d" info
                if info > 0L then raise (SingularMatrixException "cannot invert singular matrix")
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
                | MatrixPart.Upper -> 'U'
                | MatrixPart.Lower -> 'L'
            if not (eigVecs = src) then
                (this :> ITensorBackend<_>).Copy (eigVecs, src)

            use a = BLAS.GetMatrix (eigVecs, isSource=true, isTarget=true, canTranspose=false)
            use w = BLAS.GetVector (eigVals, isSource=false, isTarget=true)
            let info = 
                BLAS.Invoke<'T, BLAS.lapack_int> 
                    (singleFn=(fun () -> BLAS.LAPACKE_ssyevd (BLAS.LAPACK_COL_MAJOR, 'V', part, a.Rows, a.Ptr, a.Ld, w.Ptr)),
                     doubleFn=(fun () -> BLAS.LAPACKE_dsyevd (BLAS.LAPACK_COL_MAJOR, 'V', part, a.Rows, a.Ptr, a.Ld, w.Ptr)))
            if info < 0L then failwithf "LAPACK argument error %d" info
            if info > 0L then raise (SingularMatrixException "cannot compute eigen decomposition of singular matrix")
            a.FetchResult()
            w.FetchResult()

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

    member this.FoldLastAxis (fn, initial, trgt, a, useThreads) = 
        let initial, trgt, a = TensorHostBackend<_>.GetDataAndLayout (initial, trgt, a)
        let inline foldOp idx state xv = fn state xv
        ScalarOps.ApplyAxisFold (foldOp, id, trgt, a, Choice2Of2 initial, isIndexed=false, useThreads=useThreads)

    member this.FoldLastAxisIndexed (fn, initial, trgt, a, useThreads) = 
        let initial, trgt, a = TensorHostBackend<_>.GetDataAndLayout (initial, trgt, a)
        ScalarOps.ApplyAxisFold (fn, id, trgt, a, Choice2Of2 initial, isIndexed=true, useThreads=useThreads)

    interface System.Collections.Generic.IEnumerable<'T> with
        member this.GetEnumerator() : IEnumerator<'T> = 
            let s = seq {
                let mutable pos = PosIter32 this.FastLayout
                while pos.Active do
                    yield this.Data.[pos.Addr]
                    pos.MoveNext()
            }
            s.GetEnumerator()

    interface System.Collections.IEnumerable with    
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

