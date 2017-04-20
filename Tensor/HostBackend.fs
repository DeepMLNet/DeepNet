namespace ArrayNDNS

open System
open System.Numerics

open Basics


type TensorHostStorage<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (data: 'T []) =

    new (nElems: int64) =
        if nElems > int64 FSharp.Core.int32.MaxValue then
            failwithf "Cannot create host tensor storage for %d elements, the current
                       limit is %d elements." nElems FSharp.Core.int32.MaxValue
        TensorHostStorage<'T>(Array.zeroCreate (int32 nElems))        

    member this.Data = data

    member this.Item 
        with get (addr: int64) = data.GetValue(addr) :?> 'T
        and set (addr: int64) (value: 'T) = data.SetValue(value, addr)

    interface ITensorStorage<'T> with
        member this.Backend layout =
            TensorHostBackend<'T> (layout, this) :> ITensorBackend<_>


and TensorHostBackend<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (layout: TensorLayout, storage: TensorHostStorage<'T>) =

    let toMe (x: obj) = x :?> TensorHostBackend<'T>

    let fastLayout = FastLayout32 layout

    interface ITensorBackend<'T> with
        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr idx]
            and set idx value = storage.[layout |> TensorLayout.addr idx] <- value

        member this.Plus src1 src2 =
            let src1, src2 = src1 |> toMe, src2 |> toMe

            // so algorithm is:
            // parallel for over dimension 0
            // normal for loop over dimensions 1 ... D-2
            // loop with SIMD increment over dimension D-1

            // how to handle binary case?
            // actually, more complex
            // the write increment must also be 1

            // there is no guarantee that strides are equal
            // actually it does not matter, we just need stride=1 for last dim in all arrays

            let fl = fastLayout
            let pos : int[] = Array.zeroCreate fl.NDims


            let simdLoop () = 
                let d = fl.NDims - 1
                let rest = fl.Shape.[d] % Vector<'T>.Count
                ()



            // perform SIMD plus
            ()
            


type TensorHostStorageFactory () =
    interface ITensorStorageFactory with
        member this.Create nElems = 
            // we use reflection to drop the constraints on 'T 
            let ts = typedefof<TensorHostStorage<_>>.MakeGenericType (typeof<'T>)
            Activator.CreateInstance(ts, nElems) :?> ITensorStorage<'T>
            



