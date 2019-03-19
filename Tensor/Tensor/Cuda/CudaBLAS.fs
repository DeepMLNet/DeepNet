namespace Tensor.Cuda

open System
open System.Threading
open System.Runtime.InteropServices

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor
open Tensor.Backend
open Tensor.Host
open Tensor.Utils
open DeepNet.Utils



module internal CudaBLASExtensions = 

    type Tensor.Backend.BLAS.MatrixInfo with

        member this.CTrans = 
            match this.Trans with
            | BLAS.NoTrans   -> CudaBlas.Operation.NonTranspose
            | BLAS.Trans     -> CudaBlas.Operation.Transpose
            | BLAS.ConjTrans -> CudaBlas.Operation.ConjugateTranspose

        member this.CPtr<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> (ctx: CudaContext) =
            use _ctx = Cuda.activate ctx
            let ptr = this.Ptr |> SizeT |> CUdeviceptr
            new CudaDeviceVariable<'T> (ptr, false, SizeT (sizeof<'T>))

        member this.CPtrs (ctx: CudaContext, stream: CUstream) =
            use _ctx = Cuda.activate ctx
            let ptrs = this.Ptrs
            let hostPtrs = new CudaPageLockedHostMemory<nativeint> (SizeT ptrs.Length)
            Marshal.Copy (ptrs, 0, hostPtrs.PinnedHostPointer, ptrs.Length)
            let devPtrs = new CudaDeviceVariable<CUdeviceptr> (SizeT ptrs.Length)
            hostPtrs.AsyncCopyToDevice (devPtrs.DevicePointer, stream)           
            let disposeFn() =
                Cuda.callback ctx stream (fun () ->
                    devPtrs.Dispose()
                    hostPtrs.Dispose())
            devPtrs, disposeFn           

        member this.CRows = int this.Rows
        member this.COpRows = int this.OpRows
        member this.CCols = int this.Cols
        member this.COpCols = int this.OpCols
        member this.CLd = int this.Ld
        member this.CBatchSize = int this.BatchSize


    type Tensor.Backend.BLAS.VectorInfo with

        member this.CPtr<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> (ctx: CudaContext, ?elems: int64) =
            use _ctx = Cuda.activate ctx
            let elems = defaultArg elems 1L
            let ptr = this.Ptr |> SizeT |> CUdeviceptr
            new CudaDeviceVariable<'T> (ptr, false, SizeT (sizeof64<'T> * elems))

        member this.CInc = int this.Inc


    type Tensor.Backend.BLAS.ScalarInfo with

        member this.CPtr<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> (ctx: CudaContext) =
            use _ctx = Cuda.activate ctx
            let ptr = this.Ptr |> SizeT |> CUdeviceptr
            new CudaDeviceVariable<'T> (ptr, false, SizeT (sizeof<'T>))    


    type CUBLAS =
        static member Invoke<'T, 'R> (?singleFn, ?doubleFn, ?int32Fn, ?int64Fn) =
            BLAS.Invoke<'T,'R> (?singleFn=singleFn, ?doubleFn=doubleFn, ?int32Fn=int32Fn, ?int64Fn=int64Fn)
