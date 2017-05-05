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
        member this.Id = "Cuda"
        member this.Backend layout = 
            TensorCudaBackend<'T> (layout, this) :> ITensorBackend<_>
        member this.Factory =
            TensorCudaStorageFactory.Instance :> ITensorStorageFactory



and TensorCudaBackend<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (layout: TensorLayout, storage: TensorCudaStorage<'T>) =

    let toMe (x: obj) = x :?> TensorCudaBackend<'T>

    interface ITensorBackend<'T> with
        member this.ArgMaxLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.ArgMinLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.FoldLastAxis(fn, initial, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.FoldLastAxisIndexed(fn, initial, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.AllLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.AnyLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.GetEnumerator() : System.Collections.IEnumerator = raise (System.NotImplementedException())
        member this.GetEnumerator() : System.Collections.Generic.IEnumerator<'T> = raise (System.NotImplementedException())
        member this.MaxLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.MinLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.ProductLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.SumLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.Gather(trgt, srcIdxs, src) = raise (System.NotImplementedException())
        member this.IfThenElse(trgt, cond, ifTrue, ifFalse) = raise (System.NotImplementedException())
        member this.Scatter(trgt, trgtIdxs, src) = raise (System.NotImplementedException())
        member this.Abs(trgt, src1) = raise (System.NotImplementedException())
        member this.Acos(trgt, src1) = raise (System.NotImplementedException())
        member this.Asin(trgt, src1) = raise (System.NotImplementedException())
        member this.Atan(trgt, src1) = raise (System.NotImplementedException())
        member this.Ceiling(trgt, src1) = raise (System.NotImplementedException())
        member this.Cos(trgt, src1) = raise (System.NotImplementedException())
        member this.Cosh(trgt, src1) = raise (System.NotImplementedException())
        member this.Exp(trgt, src1) = raise (System.NotImplementedException())
        member this.Floor(trgt, src1) = raise (System.NotImplementedException())
        member this.Log(trgt, src1) = raise (System.NotImplementedException())
        member this.Log10(trgt, src1) = raise (System.NotImplementedException())
        member this.Round(trgt, src1) = raise (System.NotImplementedException())
        member this.Sgn(trgt, src1) = raise (System.NotImplementedException())
        member this.Sin(trgt, src1) = raise (System.NotImplementedException())
        member this.Sinh(trgt, src1) = raise (System.NotImplementedException())
        member this.Sqrt(trgt, src1) = raise (System.NotImplementedException())
        member this.Tan(trgt, src1) = raise (System.NotImplementedException())
        member this.Tanh(trgt, src1) = raise (System.NotImplementedException())
        member this.Truncate(trgt, src1) = raise (System.NotImplementedException())
        member this.IsFinite(trgt, src1) = raise (System.NotImplementedException())
        member this.UnaryMinus(trgt, src1) = raise (System.NotImplementedException())
        member this.UnaryPlus(trgt, src1) = raise (System.NotImplementedException())
        member this.Negate(trgt, src1) = raise (System.NotImplementedException())
        member this.Fill(fn, trgt, useThreads) = raise (System.NotImplementedException())
        member this.FillIndexed(fn, trgt, useThreads) = raise (System.NotImplementedException())
        member this.Convert(trgt, src) = raise (System.NotImplementedException())
        member this.FillConst (value, trgt) = failwith "notimpl"
        member this.Copy(trgt, src) = raise (System.NotImplementedException())
        member this.Map(fn, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.Map2(fn, trgt, src1, src2, useThreads) = raise (System.NotImplementedException())
        member this.MapIndexed(fn, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.MapIndexed2(fn, trgt, src1, src2, useThreads) = raise (System.NotImplementedException())
        member this.Add(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Subtract(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Multiply(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Divide(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Modulo(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Power(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Equal(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.NotEqual(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Less(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.LessOrEqual(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Greater(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.GreaterOrEqual(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.And(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Or(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Xor(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.MaxElemwise(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.MinElemwise(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)]
            and set idx value = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)] <- value
        member this.VecVecDot (trgt, a, b) =
            ()

        member this.MatVecDot (trgt, a, b) =
            ()

        member this.MatMatDot (trgt, a, b) =
            ()

        member this.BatchedMatMatDot (trgt, a, b) =
            ()

        member this.BatchedInvert (trgt, src) =
            ()

        member this.SymmetricEigenDecomposition (part, trgtEigVals, trgtEigVec, src) =
            ()

            

and TensorCudaStorageFactory () =
    static member Instance = TensorCudaStorageFactory ()

    interface ITensorStorageFactory with
        member this.Create nElems : ITensorStorage<'T> = 
            // we use reflection to drop the constraints on 'T 
            let ts = typedefof<TensorCudaStorage<_>>.MakeGenericType (typeof<'T>)
            Activator.CreateInstance(ts, nElems) :?> ITensorStorage<'T>
        member this.Zeroed = false


[<AutoOpen>]            
module CudaTensorTypes =
    let DevCuda = TensorCudaStorageFactory.Instance


module CudaTensor =

    let zeros<'T> = Tensor.zeros<'T> DevCuda





