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



// delegates for VectorOps
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
            | _ -> failwith "vector operation not applicable to the given tensor"
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
            | _ -> failwith "vector operation not applicable to the given tensor"
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
            | _ -> failwith "vector operation not applicable to the given tensor"
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
        VectorOps.Method<BinaryDelegate<'T>>("MaxElemwiseImpl").Invoke (trgt, src1, src2) 

    static member MinElemwise (trgt: DataAndLayout<'T>, src1: DataAndLayout<'T>, src2: DataAndLayout<'T>) =
        VectorOps.Method<BinaryDelegate<'T>>("MinElemwiseImpl").Invoke (trgt, src1, src2) 

    static member CanUse (trgt: DataAndLayout<'T>, ?src1: DataAndLayout<'T1>, ?src2: DataAndLayout<'T2>) =
        match trgt.FastLayout.NDims with
        | 0 -> false
        | nd ->
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
