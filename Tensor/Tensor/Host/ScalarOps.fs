namespace Tensor.Host

open System
open System.Threading.Tasks
open System.Collections.Generic

open Tensor
open Tensor.Utils
open Tensor.Backend
     

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

    static member FillIncrementing (start: 'T, incr: 'T, trgt: DataAndLayout<'T>) =
        let p = ScalarPrimitives.For<'T, int64>()
        let inline op (pos: int64[]) = p.Add start (p.Multiply incr (p.Convert pos.[0]))
        ScalarOps.ApplyNoaryOp (op, trgt, isIndexed=true, useThreads=true)        

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

    static member CountTrueLastAxis (trgt: DataAndLayout<int64>, src1: DataAndLayout<bool>) =
        let inline op (srcIdx: int64[]) (res: int64) (v: bool) = if v then res+1L else res
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 0L, isIndexed=false, useThreads=true)     

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
        ScalarOps.ApplyAxisFold (op, fst, trgt, src1, initial=Choice1Of2 (NotFound, minValue<'T>), 
                                 isIndexed=true, useThreads=true)     

    static member ArgMinLastAxis (trgt: DataAndLayout<int64>, src1: DataAndLayout<'T>) =
        let nd = src1.FastLayout.NDims
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) (minPos, minVal) (v: 'T) = 
            if p.Less v minVal then srcIdx.[nd-1], v
            else minPos, minVal
        ScalarOps.ApplyAxisFold (op, fst, trgt, src1, initial=Choice1Of2 (NotFound, maxValue<'T>), 
                                 isIndexed=true, useThreads=true)     
                                 
    static member FindLastAxis (value: 'T, trgt: DataAndLayout<int64>, src1: DataAndLayout<'T>) =
        let nd = src1.FastLayout.NDims
        let p = ScalarPrimitives.For<'T, 'T>()
        let inline op (srcIdx: int64[]) pos (v: 'T) =
            if pos <> NotFound then pos
            else 
                if p.Equal v value then srcIdx.[nd-1]
                else NotFound
        ScalarOps.ApplyAxisFold (op, id, trgt, src1, initial=Choice1Of2 NotFound, 
                                 isIndexed=true, useThreads=true)             
                                          
    static member inline MaskedGet (trgt: DataAndLayout<'T>, 
                                    src: DataAndLayout<'T>, 
                                    masks: DataAndLayout<bool> option []) =
        let mutable trgtPosIter = PosIter32 trgt.FastLayout
        let mutable srcPosIter = PosIter32 src.FastLayout
        while trgtPosIter.Active do                      
            let maskVal =
                Array.zip masks srcPosIter.Pos 
                |> Array.fold (fun s (m, p) -> match m with 
                                               | Some m -> s && m.Data.[m.FastLayout.UncheckedAddr [|p|]]
                                               | _ -> s) true 
            if maskVal then
                trgt.Data.[trgtPosIter.Addr] <- src.Data.[srcPosIter.Addr]
                trgtPosIter.MoveNext()
            srcPosIter.MoveNext()
                                 
    static member inline MaskedSet (trgt: DataAndLayout<'T>, 
                                    masks: DataAndLayout<bool> option [],
                                    src: DataAndLayout<'T>) =
        let mutable trgtPosIter = PosIter32 trgt.FastLayout
        let mutable srcPosIter = PosIter32 src.FastLayout
        while trgtPosIter.Active do                      
            let maskVal =
                Array.zip masks trgtPosIter.Pos 
                |> Array.fold (fun s (m, p) -> match m with 
                                               | Some m -> s && m.Data.[m.FastLayout.UncheckedAddr [|p|]]
                                               | _ -> s) true 
            if maskVal then
                trgt.Data.[trgtPosIter.Addr] <- src.Data.[srcPosIter.Addr]
                srcPosIter.MoveNext()
            trgtPosIter.MoveNext()              
            
    static member inline TrueIndices (trgt: DataAndLayout<int64>, src: DataAndLayout<bool>) =
        let mutable trgtPosIter = PosIter32 trgt.FastLayout
        let mutable srcPosIter = PosIter32 src.FastLayout
        while trgtPosIter.Active do                      
            if src.Data.[srcPosIter.Addr] then
                for d in 0 .. src.FastLayout.NDims-1 do
                    trgt.Data.[trgtPosIter.Addr] <- int64 srcPosIter.Pos.[d]
                    trgtPosIter.MoveNext()
            srcPosIter.MoveNext()
