namespace ArrayNDNS

open System
open System.Numerics
open System.Threading.Tasks

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

    member private this.FastLayout = fastLayout
    member this.Storage = storage
    member this.Data = this.Storage.Data

    member this.BinaryOp 
            (vecOp: Vector<'T> -> Vector<'T> -> Vector<'T>) 
            (src1: TensorHostBackend<'T>) (src2: TensorHostBackend<'T>) =        

        let trgtFl, src1Fl, src2Fl = this.FastLayout, src1.FastLayout, src2.FastLayout
        let trgtData, src1Data, src2Data = this.Data, src1.Data, src2.Data
        let nd = trgtFl.NDims
        let shape = trgtFl.Shape

        let inline stride1InnerLoop 
                (trgtAddr: int) (src1Addr: int) (src2Addr: int)
                (trgtRest: 'T[]) (src1Rest: 'T[]) (src2Rest: 'T[]) =          
                    
            let mutable trgtAddr, src1Addr, src2Addr = 
                trgtAddr, src1Addr, src2Addr
                
            let vecIters = shape.[nd-1] / Vector<'T>.Count
            for vecIter in 0 .. vecIters - 1 do
                let src1Vec = Vector (src1Data, src1Addr)
                let src2Vec = Vector (src2Data, src2Addr)
                let trgtVec = vecOp src1Vec src2Vec
                trgtVec.CopyTo (trgtData, trgtAddr)
                trgtAddr <- trgtAddr + Vector<'T>.Count
                src1Addr <- src1Addr + Vector<'T>.Count
                src2Addr <- src2Addr + Vector<'T>.Count
 
            let restElems = shape.[nd-1] % Vector<'T>.Count
            if restElems > 0 then
                for restPos in 0 .. restElems - 1 do
                    src1Rest.[restPos] <- src1Data.[src1Addr]
                    src2Rest.[restPos] <- src2Data.[src2Addr]
                    src1Addr <- src1Addr + 1
                    src2Addr <- src2Addr + 1
                let src1Vec = Vector src1Rest
                let src2Vec = Vector src2Rest
                let trgtVec = vecOp src1Vec src2Vec
                for restPos in 0 .. restElems - 1 do
                    trgtData.[trgtAddr] <- trgtVec.[restPos]
                    trgtAddr <- trgtAddr + 1     
                        
        let inline genericInnerLoop
                (trgtAddr: int) (src1Addr: int) (src2Addr: int)
                (trgtBuf: 'T[]) (src1Buf: 'T[]) (src2Buf: 'T[]) =          

            let mutable trgtAddr, src1Addr, src2Addr = 
                trgtAddr, src1Addr, src2Addr
            let mutable bufWritePos = 0
            let mutable srcPos = 0
            for pos in 0 .. shape.[nd-1] - 1 do
                src1Buf.[bufWritePos] <- src1Data.[src1Addr]
                src2Buf.[bufWritePos] <- src2Data.[src2Addr]
                src1Addr <- src1Addr + src1Fl.Stride.[nd-1]
                src2Addr <- src2Addr + src2Fl.Stride.[nd-1]
                bufWritePos <- bufWritePos + 1

                if bufWritePos = Vector<'T>.Count || pos = shape.[nd-1] - 1 then
                    let src1Vec = Vector src1Buf
                    let src2Vec = Vector src2Buf
                    let trgtVec = vecOp src1Vec src2Vec
                    for bufReadPos in 0 .. bufWritePos - 1 do
                        trgtData.[trgtAddr] <- trgtVec.[bufReadPos]
                        trgtAddr <- trgtAddr + trgtFl.Stride.[nd-1]
                    bufWritePos <- 0
          
        let inline scalarInnerLoop
                (trgtAddr: int) (src1Addr: int) (src2Addr: int)
                (trgtBuf: 'T[]) (src1Buf: 'T[]) (src2Buf: 'T[]) =    

            src1Buf.[0] <- src1Data.[src1Addr]
            src2Buf.[0] <- src2Data.[src2Addr]
            let src1Vec = Vector src1Buf
            let src2Vec = Vector src2Buf
            let trgtVec = vecOp src1Vec src2Vec
            trgtBuf.[trgtAddr] <- trgtVec.[0]

        let inline outerLoops (dim0Fixed: bool) (dim0Pos: int) =
            let trgtBuf : 'T[] = Array.zeroCreate Vector<'T>.Count
            let src1Buf : 'T[] = Array.zeroCreate Vector<'T>.Count
            let src2Buf : 'T[] = Array.zeroCreate Vector<'T>.Count

            let fromDim = if dim0Fixed then 1 else 0
            let startPos = Array.zeroCreate trgtFl.NDims
            if dim0Fixed then startPos.[0] <- dim0Pos

            let mutable trgtPosIter = 
                PosIter32 (trgtFl, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src1PosIter = 
                PosIter32 (src1Fl, startPos, fromDim=fromDim, toDim=nd-2)
            let mutable src2PosIter = 
                PosIter32 (src2Fl, startPos, fromDim=fromDim, toDim=nd-2)

            while trgtPosIter.Active do
                if nd = 0 then
                    scalarInnerLoop
                        trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
                        trgtBuf src1Buf src2Buf
                elif trgtFl.Stride.[nd-1] = 1 && 
                     src1Fl.Stride.[nd-1] = 1 && src2Fl.Stride.[nd-1] = 1 then
                    stride1InnerLoop 
                        trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
                        trgtBuf src1Buf src2Buf
                else
                    genericInnerLoop
                        trgtPosIter.Addr src1PosIter.Addr src2PosIter.Addr
                        trgtBuf src1Buf src2Buf
                trgtPosIter.MoveNext()
                src1PosIter.MoveNext()
                src2PosIter.MoveNext()
                    
        let useThreads = true
        if nd > 1 && useThreads then
            Parallel.For (0, shape.[0], fun dim0Pos -> outerLoops true dim0Pos) |> ignore
        else
            outerLoops false 0


    interface ITensorBackend<'T> with
        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr idx]
            and set idx value = storage.[layout |> TensorLayout.addr idx] <- value

        member this.Plus src1 src2 =
            let inline op a b = Vector.Add<'T> (a, b)
            this.BinaryOp op (toMe src1) (toMe src2)



type TensorHostStorageFactory () =
    interface ITensorStorageFactory with
        member this.Create nElems = 
            // we use reflection to drop the constraints on 'T 
            let ts = typedefof<TensorHostStorage<_>>.MakeGenericType (typeof<'T>)
            Activator.CreateInstance(ts, nElems) :?> ITensorStorage<'T>
            



