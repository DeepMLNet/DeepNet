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
open DeepNet.Utils



/// Fast layout operations.
[<Struct>]
[<StructuredFormatDisplay("FastLayout32 (Shape={Shape} Offset={Offset} Stride={Stride})")>]
type internal FastLayout32 = 
    val NDims   : int
    val NElems  : int
    val Offset  : int
    val Shape   : int []
    val Stride  : int []

    static member inline private checkedInt layout (x: int64) =
        if int64 FSharp.Core.int.MinValue <= x && x <= int64 FSharp.Core.int.MaxValue then
            int x
        else failwithf "cannot convert tensor layout %A to 32-bit integer" layout

    new (layout: TensorLayout) = {
        NDims   = TensorLayout.nDims layout
        NElems  = TensorLayout.nElems layout |> FastLayout32.checkedInt layout
        Offset  = TensorLayout.offset layout |> FastLayout32.checkedInt layout
        Shape   = TensorLayout.shape layout |> List.toArray |> Array.map (FastLayout32.checkedInt layout)
        Stride  = TensorLayout.stride layout |> List.toArray |> Array.map (FastLayout32.checkedInt layout)
    }

    member inline this.IsPosValid (pos: int[]) =
        if pos.Length = this.NDims then
            Array.forall2 (fun i size -> 0 <= i && i < size) pos this.Shape
        else false

    member inline this.UncheckedAddr (pos: int[]) =
        let mutable addr = this.Offset
        for d=0 to this.NDims-1 do
            addr <- addr + pos.[d] * this.Stride.[d]
        addr

    member inline this.Addr (pos: int64[]) =
        if pos.Length <> this.NDims then
            indexOutOfRange "Position %A has wrong dimensionality for tensor of shape %A." pos this.Shape
        let mutable addr = this.Offset           
        for d=0 to this.NDims-1 do
            let p = int pos.[d]
            if (0 <= p && p < this.Shape.[d]) then
                addr <- addr + p * this.Stride.[d]
            else
                indexOutOfRange "Position %A is out of range for tensor of shape %A." pos this.Shape
        addr



/// Fast index operations.
[<Struct>]
type internal PosIter32 = 
    val Pos             : int []
    val mutable Addr    : int
    val mutable Active  : bool
    val Shape           : int []
    val Stride          : int []               
    val FromDim         : int
    val ToDim           : int

    new (fl: FastLayout32, ?startPos, ?fromDim, ?toDim) = 
        let startPos = defaultArg startPos (Array.zeroCreate fl.NDims)
        let fromDim = defaultArg fromDim 0
        let toDim = defaultArg toDim (fl.NDims - 1)
        #if DEBUG
        if not (0 <= fromDim) then
            failwithf "fromDim=%d out of range for shape %A" fromDim fl.Shape
        if not (toDim < fl.NDims) then
            failwithf "toDim=%d out of range for shape %A" toDim fl.Shape
        #endif
        let active = 
            [0 .. fl.NDims]
            |> List.forall (fun d -> 
                if fromDim <= d && d <= toDim then 
                    0 <= startPos.[d] && startPos.[d] < fl.Shape.[d] 
                else true)
        {
            Pos     = Array.copy startPos
            Addr    = fl.UncheckedAddr startPos
            Active  = active
            Shape   = fl.Shape
            Stride  = fl.Stride
            FromDim = fromDim
            ToDim   = toDim
        }

    member inline this.MoveNext () =
        #if DEBUG
        if not this.Active then
            failwith "iteration past end attempted"
        #endif      

        // try incrementing starting from last axis
        let mutable increment = true
        let mutable d = this.ToDim
        while increment && d >= this.FromDim do
            if this.Pos.[d] = this.Shape.[d] - 1 then
                // was last element of that axis
                this.Addr <- this.Addr - this.Pos.[d] * this.Stride.[d]
                this.Pos.[d] <- 0
                d <- d - 1
            else
                // can increment this axis
                this.Addr <- this.Addr + this.Stride.[d]
                this.Pos.[d] <- this.Pos.[d] + 1
                increment <- false  
        // if we tried to increment past first axis, then iteration finished                            
        if d < this.FromDim then this.Active <- false                  



/// Data and fast layout of a host tensor.
type internal DataAndLayout<'T> = {
    Data:       'T[]
    FastLayout: FastLayout32
}

