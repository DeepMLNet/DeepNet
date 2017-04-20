namespace ArrayNDNS

open ManagedCuda
open ManagedCuda.BasicTypes

open Basics
open Basics.Cuda
open System


module private CudaHelpers =

    let mutable context = None
    let initContext () =
        match context with
        | Some _ -> ()
        | None ->
            try context <- Some (new CudaContext(createNew=false))
            with e ->
                printfn "Cannot create CUDA context: %s" e.Message
                failwithf "Cannot create CUDA context: %s" e.Message

    /// create a new CUDA device variable
    let newDevVar<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (elems: int64) = 
        try new CudaDeviceVariable<'T> (SizeT elems)
        with :? CudaException as e when e.CudaError = CUResult.ErrorOutOfMemory 
                                     || e.CudaError = CUResult.ErrorUnknown ->
            let sizeInBytes = elems * sizeof64<'T>
            failwithf "CUDA memory allocation of %d MB failed (%A)" 
                      (sizeInBytes / pown 2L 20) e.CudaError


type TensorCudaStorage<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (data: CudaDeviceVariable<'T>) =

    new (nElems: int64) =
        CudaHelpers.initContext ()
        // CUDA cannot allocate memory of size zero
        let nElems = if nElems > 0L then nElems else 1L
        TensorCudaStorage<'T> (CudaHelpers.newDevVar nElems)
     
    member this.Data = data

    member this.ByteData =
        new CudaDeviceVariable<byte> (data.DevicePointer, data.SizeInBytes)        

    override this.Finalize() = 
        if data <> null then data.Dispose()

    member this.Item 
        with get (addr: int64) =
            if typeof<'T> = typeof<bool> then
                let hostBuf : byte ref = ref 0uy
                this.ByteData.CopyToHost(hostBuf, SizeT (addr * sizeof64<byte>))
                !hostBuf <> 0uy |> box |> unbox
            else
                let hostBuf = ref (new 'T())
                this.Data.CopyToHost(hostBuf, SizeT (addr * sizeof64<'T>))
                !hostBuf
                
        and set (addr: int64) (value: 'T) = 
            if typeof<'T> = typeof<bool> then
                let byteVal = if (box value :?> bool) then 1uy else 0uy
                this.ByteData.CopyToDevice(byteVal, SizeT (addr * sizeof64<byte>))
            else
                this.Data.CopyToDevice(value, SizeT (addr * sizeof64<'T>))

    interface ITensorStorage<'T> with
        member this.Backend layout = 
            TensorCudaBackend<'T> (layout, this) :> ITensorBackend<_>




and TensorCudaBackend<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (layout: TensorLayout, storage: TensorCudaStorage<'T>) =

    let toMe (x: obj) = x :?> TensorCudaBackend<'T>

    interface ITensorBackend<'T> with
        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr idx]
            and set idx value = storage.[layout |> TensorLayout.addr idx] <- value

        member this.Plus src1 src2 =
            let src1, src2 = src1 |> toMe, src2 |> toMe
            // call CUDA plus kernel
            ()
            

type TensorCudaStorageFactory () =
    interface ITensorStorageFactory with
        member this.Create nElems : ITensorStorage<'T> = 
            // we use reflection to drop the constraints on 'T 
            let tcs = typedefof<TensorCudaStorage<_>>.MakeGenericType (typeof<'T>)
            Activator.CreateInstance(tcs, nElems) :?> ITensorStorage<'T>
            
