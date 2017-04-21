namespace ArrayNDNS


[<AutoOpen>]
module FastLayoutTypes =

    [<Struct>]
    type FastLayoutT = 
        val NDims   : int
        val NElems  : int64
        val Offset  : int64
        val Shape   : int64 []
        val Stride  : int64 []

        new (layout: TensorLayout) = {
            NDims   = TensorLayout.nDims layout
            NElems  = TensorLayout.nElems layout
            Offset  = TensorLayout.offset layout
            Shape   = TensorLayout.shape layout |> List.toArray
            Stride  = TensorLayout.stride layout |> List.toArray
        }

        member inline this.Addr (idx: int64[]) =
            let mutable addr = this.Offset
            for d=0 to this.NDims-1 do
                addr <- addr + idx.[d] * this.Stride.[d]
            addr

    let inline private checkedInt layout (x: int64) =
        if int64 FSharp.Core.int.MinValue <= x && x <= int64 FSharp.Core.int.MaxValue then
            int x
        else failwithf "cannot convert tensor layout %A to 32-bit integer" layout

    [<Struct>]
    type FastLayout32 = 
        val NDims   : int
        val NElems  : int
        val Offset  : int
        val Shape   : int []
        val Stride  : int []

        new (layout: TensorLayout) = {
            NDims   = TensorLayout.nDims layout
            NElems  = TensorLayout.nElems layout |> checkedInt layout
            Offset  = TensorLayout.offset layout |> checkedInt layout
            Shape   = TensorLayout.shape layout |> List.toArray |> Array.map (checkedInt layout)
            Stride  = TensorLayout.stride layout |> List.toArray |> Array.map (checkedInt layout)
        }

        member inline this.IsPosValid (pos: int[]) =
            if pos.Length = this.NDims then
                Array.forall2 (fun i size -> 0 <= i && i < size) pos this.Shape
            else false

        member inline this.Addr (pos: int[]) =
            assert (this.IsPosValid pos)
            let mutable addr = this.Offset
            for d=0 to this.NDims-1 do
                addr <- addr + pos.[d] * this.Stride.[d]
            addr

    [<Struct>]
    type PosIter32 = 
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
            assert (fl.IsPosValid startPos)
            assert (0 <= fromDim && fromDim < fl.NDims)
            assert (0 <= toDim && toDim < fl.NDims)
            let active = 
                fl.Shape 
                |> Array.indexed 
                |> Array.forall (fun (d, s) -> 
                    if fromDim <= d && d <= toDim then fl.Shape.[d] > 0
                    else true)
            {
                Pos     = Array.copy startPos
                Addr    = fl.Addr startPos
                Active  = active
                Shape   = fl.Shape
                Stride  = fl.Stride
                FromDim = fromDim
                ToDim   = toDim
            }

        member inline this.MoveNext () =
            assert (this.Active)

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



module FastLayout =

    let ofLayout layout =
        FastLayoutT layout

    /// sequential enumeration of all addresses
    let inline allAddr (fl: FastLayoutT) = seq {
        if fl.NDims = 0 then
            yield fl.Offset
        else
            let pos = Array.zeroCreate fl.NDims
            let mutable addr = fl.Offset
            let mutable moreElements = fl.NElems > 0L
                
            while moreElements do
                yield addr

                let mutable increment = true
                let mutable dim = fl.NDims - 1
                while increment && dim >= 0 do
                    if pos.[dim] = fl.Shape.[dim] - 1L then
                        // was last element of that axis
                        addr <- addr - pos.[dim] * fl.Stride.[dim]
                        pos.[dim] <- 0L
                        dim <- dim - 1
                    else
                        // can increment this axis
                        addr <- addr + fl.Stride.[dim]
                        pos.[dim] <- pos.[dim] + 1L
                        increment <- false  
                            
                if dim < 0 then 
                    // tried to increment past zero axis
                    // iteration finished
                    moreElements <- false                  
    }
            

