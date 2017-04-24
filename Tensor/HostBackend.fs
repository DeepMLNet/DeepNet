namespace ArrayNDNS

open System
open System.Numerics
open System.Threading.Tasks

open Basics

module TensorHostSettings =
    let mutable UseThreads = true


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
        member this.Factory = 
            TensorHostStorageFactory.Instance :> ITensorStorageFactory
            


and TensorHostBackend<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (layout: TensorLayout, storage: TensorHostStorage<'T>) =

    let toMe (x: obj) = x :?> TensorHostBackend<'T>

    member val FastLayout = FastLayout32 layout
    member this.Storage = storage
    member val Data = storage.Data

    member inline internal trgt.ApplyNoaryOp 
            (scalarOp: unit -> 'T) (vectorOp: unit -> Vector<'T>) (hasVectorOp: bool) =        

        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
        let hasStride1InLastDim = trgt.FastLayout.Stride.[nd-1] = 1 

        let inline vectorInnerLoop (trgtAddr: int) =                   
            let mutable trgtAddr = trgtAddr
                
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            for vecIter in 0 .. vecIters-1 do
                let trgtVec = vectorOp ()
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
 
            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- scalarOp ()
                trgtAddr <- trgtAddr + 1
                       
        let inline genericInnerLoop (trgtAddr: int) =
            let mutable trgtAddr = trgtAddr
            for pos in 0 .. shape.[nd-1] - 1 do
                trgt.Data.[trgtAddr] <- scalarOp ()
                trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
         
        let inline scalarInnerLoop (trgtAddr: int) =
            trgt.Data.[trgtAddr] <- scalarOp ()

        let inline outerLoops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)

            while trgtPosIter.Active do
                if nd = 0 then
                    scalarInnerLoop trgtPosIter.Addr 
                elif hasStride1InLastDim && hasVectorOp then
                    vectorInnerLoop trgtPosIter.Addr 
                else
                    genericInnerLoop trgtPosIter.Addr 
                trgtPosIter.MoveNext()
                    
        if nd > 1 && TensorHostSettings.UseThreads && not hasStride1InLastDim then
            Parallel.For (0, shape.[0], fun dim0Pos -> outerLoops true dim0Pos) |> ignore
        else
            outerLoops false 0


    member inline internal trgt.ApplyUnaryOp 
            (scalarOp: 'T -> 'T) (vectorOp: Vector<'T> -> Vector<'T>) (hasVectorOp: bool)
            (src1: TensorHostBackend<'T>) =        

        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
        let hasStride1InLastDim = 
            trgt.FastLayout.Stride.[nd-1] = 1 && src1.FastLayout.Stride.[nd-1] = 1 

        let inline vectorInnerLoop (trgtAddr: int) (src1Addr: int) =                   
            let mutable trgtAddr, src1Addr = trgtAddr, src1Addr
                
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            for vecIter in 0 .. vecIters-1 do
                let src1Vec = Vector (src1.Data, src1Addr)
                let trgtVec = vectorOp src1Vec 
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
                src1Addr <- src1Addr + Vector<'T>.Count
 
            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- scalarOp src1.Data.[src1Addr] 
                trgtAddr <- trgtAddr + 1
                src1Addr <- src1Addr + 1
                       
        let inline genericInnerLoop (trgtAddr: int) (src1Addr: int) =
            let mutable trgtAddr, src1Addr = trgtAddr, src1Addr
            for pos in 0 .. shape.[nd-1] - 1 do
                trgt.Data.[trgtAddr] <- scalarOp src1.Data.[src1Addr] 
                trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
         
        let inline scalarInnerLoop (trgtAddr: int) (src1Addr: int) =
            trgt.Data.[trgtAddr] <- scalarOp src1.Data.[src1Addr] 

        let inline outerLoops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)

            while trgtPosIter.Active do
                if nd = 0 then
                    scalarInnerLoop trgtPosIter.Addr src1PosIter.Addr 
                elif hasStride1InLastDim && hasVectorOp then
                    vectorInnerLoop trgtPosIter.Addr src1PosIter.Addr 
                else
                    genericInnerLoop trgtPosIter.Addr src1PosIter.Addr 
                trgtPosIter.MoveNext()
                src1PosIter.MoveNext()
                    
        if nd > 1 && TensorHostSettings.UseThreads && not hasStride1InLastDim then
            Parallel.For (0, shape.[0], fun dim0Pos -> outerLoops true dim0Pos) |> ignore
        else
            outerLoops false 0


    member inline internal trgt.ApplyBinaryOp 
            (scalarOp: 'T -> 'T -> 'T)
            (vectorOp: Vector<'T> -> Vector<'T> -> Vector<'T>) 
            (hasVectorOp: bool)
            (src1: TensorHostBackend<'T>) (src2: TensorHostBackend<'T>) =        

        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
        let hasStride1InLastDim = 
            trgt.FastLayout.Stride.[nd-1] = 1 && 
            src1.FastLayout.Stride.[nd-1] = 1 && src2.FastLayout.Stride.[nd-1] = 1

        let inline vectorInnerLoop (trgtAddr: int) (src1Addr: int) (src2Addr: int) =                   
            let mutable trgtAddr, src1Addr, src2Addr = 
                trgtAddr, src1Addr, src2Addr
                
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            for vecIter in 0 .. vecIters-1 do
                let src1Vec = Vector (src1.Data, src1Addr)
                let src2Vec = Vector (src2.Data, src2Addr)
                let trgtVec = vectorOp src1Vec src2Vec
                trgtVec.CopyTo (trgt.Data, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
                src1Addr <- src1Addr + Vector<'T>.Count
                src2Addr <- src2Addr + Vector<'T>.Count
 
            let restElems = shape.[nd-1] % Vector<'T>.Count
            for restPos in 0 .. restElems - 1 do
                trgt.Data.[trgtAddr] <- scalarOp src1.Data.[src1Addr] src2.Data.[src2Addr]
                trgtAddr <- trgtAddr + 1
                src1Addr <- src1Addr + 1
                src2Addr <- src2Addr + 1
                       
        let inline genericInnerLoop (trgtAddr: int) (src1Addr: int) (src2Addr: int) =
            let mutable trgtAddr, src1Addr, src2Addr = 
                trgtAddr, src1Addr, src2Addr
            for pos in 0 .. shape.[nd-1] - 1 do
                trgt.Data.[trgtAddr] <- scalarOp src1.Data.[src1Addr] src2.Data.[src2Addr]
                trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                src1Addr <- src1Addr + src1.FastLayout.Stride.[nd-1]
                src2Addr <- src2Addr + src2.FastLayout.Stride.[nd-1]
         
        let inline scalarInnerLoop (trgtAddr: int) (src1Addr: int) (src2Addr: int) =
            trgt.Data.[trgtAddr] <- scalarOp src1.Data.[src1Addr] src2.Data.[src2Addr]

        let inline outerLoops (dim0Fixed: bool) (dim0Pos: int) =
            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate nd
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgt.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src1PosIter = 
                PosIter32 (src1.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src2PosIter = 
                PosIter32 (src2.FastLayout, startPos, fromDim=fromDim, toDim=nd-2)

            while trgtPosIter.Active do
                if nd = 0 then
                    scalarInnerLoop trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
                elif hasStride1InLastDim && hasVectorOp then
                    vectorInnerLoop trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
                else
                    genericInnerLoop trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
                trgtPosIter.MoveNext()
                src1PosIter.MoveNext()
                src2PosIter.MoveNext()
                    
        if nd > 1 && TensorHostSettings.UseThreads && not hasStride1InLastDim then
            Parallel.For (0, shape.[0], fun dim0Pos -> outerLoops true dim0Pos) |> ignore
        else
            outerLoops false 0


    member inline internal trgt.ApplyNAryOp (scalarOp: 'T[] -> 'T) (srcs: TensorHostBackend<'T>[]) =        
        assert (srcs |> Array.forall (fun src -> 
            List.ofArray trgt.FastLayout.Shape = List.ofArray src.FastLayout.Shape))
        let nSrcs = srcs.Length
        let nd = trgt.FastLayout.NDims
        let shape = trgt.FastLayout.Shape
                        
        let inline genericInnerLoop (trgtAddr: int) (srcsAddr: int[]) =
            let mutable trgtAddr = trgtAddr
            let srcVals : 'T[] = Array.zeroCreate nSrcs
            for pos in 0 .. shape.[nd-1] - 1 do
                for s in 0 .. nSrcs-1 do
                    srcVals.[s] <- srcs.[s].Data.[srcsAddr.[s]]
                trgt.Data.[trgtAddr] <- scalarOp srcVals
                trgtAddr <- trgtAddr + trgt.FastLayout.Stride.[nd-1]
                for s in 0 .. nSrcs-1 do
                    srcsAddr.[s] <- srcsAddr.[s] + srcs.[s].FastLayout.Stride.[nd-1]
         
        let inline scalarInnerLoop (trgtAddr: int) (srcsAddr: int[]) =    
            let srcsValue = 
                (srcs, srcsAddr) 
                ||> Array.map2 (fun (src: TensorHostBackend<'T>) addr -> src.Data.[addr])
            trgt.Data.[trgtAddr] <- scalarOp srcsValue

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
                    genericInnerLoop trgtPosIter.Addr srcsAddr
                trgtPosIter.MoveNext()
                for s in 0 .. nSrcs-1 do
                    srcsPosIter.[s].MoveNext()
                    
        if nd > 1 && TensorHostSettings.UseThreads then
            Parallel.For (0, shape.[0], fun dim0Pos -> outerLoops true dim0Pos) |> ignore
        else
            outerLoops false 0
            

    interface ITensorBackend<'T> with
        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr idx]
            and set idx value = storage.[layout |> TensorLayout.addr idx] <- value

        member this.Plus src1 src2 =
            let s = ScalarOps.ForType<'T> ()
            let inline scalarOp a b = s.Plus a b
            let inline vectorOp a b = Vector.Add(a, b)
            this.ApplyBinaryOp scalarOp vectorOp true (toMe src1) (toMe src2)




and TensorHostStorageFactory () =
    static member Instance = TensorHostStorageFactory () 

    interface ITensorStorageFactory with
        member this.Create nElems = 
            // we use reflection to drop the constraints on 'T 
            let ts = typedefof<TensorHostStorage<_>>.MakeGenericType (typeof<'T>)
            Activator.CreateInstance(ts, nElems) :?> ITensorStorage<'T>
            

[<AutoOpen>]            
module HostTensorTypes =
    let DevHost = TensorHostStorageFactory.Instance


type HostTensor () =

    static member zeros<'T> (shape: int64 list) : Tensor<'T> =
        Tensor<'T>.zeros (shape, DevHost)



