namespace ArrayNDNS

open System
open System.Numerics
open System.Threading.Tasks
open System.Linq.Expressions
open System.Collections.Generic


open Basics
open System.Reflection

module TensorHostSettings =
    let mutable UseThreads = true

open TensorHostSettings


type internal DataAndLayout<'T> = {
    Data:       'T[]
    FastLayout: FastLayout32
}


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


type internal FillDelegate<'T>   = delegate of 'T * DataAndLayout<'T> -> unit
type internal UnaryDelegate<'T>  = delegate of DataAndLayout<'T> * DataAndLayout<'T> -> unit
type internal BinaryDelegate<'T> = delegate of DataAndLayout<'T> * DataAndLayout<'T> * DataAndLayout<'T> -> unit
type internal CopyDelegate<'T>   = delegate of DataAndLayout<'T> * DataAndLayout<'T> -> unit

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


type TensorHostStorage<'T> (data: 'T []) =

    new (nElems: int64) =
        if nElems > int64 FSharp.Core.int32.MaxValue then
            failwithf "Cannot create host tensor storage for %d elements, the current
                       limit is %d elements." nElems FSharp.Core.int32.MaxValue
        TensorHostStorage<'T> (Array.zeroCreate (int32 nElems))        

    member this.Data = data

    interface ITensorStorage<'T> with
        member this.Id = "Host"
        member this.Backend layout =
            TensorHostBackend<'T> (layout, this) :> ITensorBackend<_>
        member this.Factory = 
            TensorHostStorageFactory.Instance :> ITensorStorageFactory
            


and TensorHostBackend<'T> (layout: TensorLayout, storage: TensorHostStorage<'T>) =

    let toMe (x: obj) = x :?> TensorHostBackend<'T>

    member val internal FastLayout = FastLayout32 layout
    member this.Storage = storage
    member val Data = storage.Data
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

    interface ITensorBackend<'T> with
        member this.Item 
            with get idx = this.Data.[this.FastLayout.Addr idx]
            and set idx value = this.Data.[this.FastLayout.Addr idx] <- value

        member this.FillConst (value, trgt) =
            let trgt= TensorHostBackend<_>.ElemwiseDataAndLayout (trgt)
            if VectorOps.CanUse (trgt) then VectorOps.Fill (value, trgt)
            else ScalarOps.Fill (value, trgt)

        member this.Copy (trgt, a) =
            let trgt, a = TensorHostBackend<_>.ElemwiseDataAndLayout (trgt, a)
            if VectorOps.CanUse (trgt, a) then VectorOps.Copy (trgt, a)
            else ScalarOps.Copy (trgt, a)

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
            let nd = src.NDims
            let trgt = (trgt.Backend :?> TensorHostBackend<'T>).DataAndLayout
            let src = (src.Backend :?> TensorHostBackend<'T>).DataAndLayout
            let srcIndices = 
                srcIndices 
                |> List.map (Option.map (fun i -> (i.Backend :?> TensorHostBackend<int64>).DataAndLayout))
                |> Array.ofList
            ScalarOps.Gather (trgt, srcIndices, src)

        member this.Scatter (trgt: Tensor<'T>, trgtIndices, src) =
            let trgt = (trgt.Backend :?> TensorHostBackend<'T>).DataAndLayout
            let src = (src.Backend :?> TensorHostBackend<'T>).DataAndLayout
            let trgtIndices = 
                trgtIndices 
                |> List.map (Option.map (fun i -> (i.Backend :?> TensorHostBackend<int64>).DataAndLayout))
                |> Array.ofList
            ScalarOps.Scatter (trgt, trgtIndices, src)

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



and TensorHostStorageFactory () =
    static member Instance = TensorHostStorageFactory () 

    interface ITensorStorageFactory with
        member this.Create nElems = 
            TensorHostStorage<_> nElems :> ITensorStorage<_>
        member this.Zeroed = true
            

[<AutoOpen>]            
module HostTensorTypes =
    let DevHost = TensorHostStorageFactory.Instance


module HostTensor =

    let zeros<'T> = Tensor.zeros<'T> DevHost 

    let ones<'T> = Tensor.ones<'T> DevHost

    let falses = Tensor.falses DevHost

    let trues = Tensor.trues DevHost

    let scalar<'T> = Tensor.scalar<'T> DevHost

    let init<'T> = Tensor.init<'T> DevHost





