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
            

