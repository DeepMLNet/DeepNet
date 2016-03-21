namespace ArrayNDNS


[<AutoOpen>]
module FastLayoutTypes =

    [<Struct>]
    type FastLayoutT = 
        val NDims   : int
        val NElems  : int
        val Offset  : int
        val Shape   : int []
        val Stride  : int []

        new (layout: ArrayNDLayoutT) = {
            NDims   = ArrayNDLayout.nDims layout
            NElems  = ArrayNDLayout.nElems layout
            Offset  = ArrayNDLayout.offset layout
            Shape   = ArrayNDLayout.shape layout |> List.toArray
            Stride  = ArrayNDLayout.stride layout |> List.toArray
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
            let mutable moreElements = fl.NElems > 0
                
            while moreElements do
                yield addr

                let mutable increment = true
                let mutable dim = fl.NDims - 1
                while increment && dim >= 0 do
                    if pos.[dim] = fl.Shape.[dim] - 1 then
                        // was last element of that axis
                        addr <- addr - pos.[dim] * fl.Stride.[dim]
                        pos.[dim] <- 0
                        dim <- dim - 1
                    else
                        // can increment this axis
                        addr <- addr + fl.Stride.[dim]
                        pos.[dim] <- pos.[dim] + 1
                        increment <- false  
                            
                if dim < 0 then 
                    // tried to increment past zero axis
                    // iteration finished
                    moreElements <- false                  
    }
            

