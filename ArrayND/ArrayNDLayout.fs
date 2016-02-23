namespace ArrayNDNS

open Basics


[<AutoOpen>]
module ArrayNDLayoutTypes =
    // layout (shape, offset, stride) of an ArrayND
    type ArrayNDLayoutT = {
        /// shape
        Shape: int list;
        /// offset in elements
        Offset: int;
        /// stride in elements
        Stride: int list;
    }

    /// range specification
    type RangeT = 
        | Elem of int
        | Rng of int * int
        | NewAxis
        | All
        | AllFill


module ArrayNDLayout =
    /// checks that the given index is valid for the given shape
    let inline checkIndex shp idx =
        if List.length shp <> List.length idx then
            failwithf "index %A has other dimensionality than shape %A" idx shp
        if not (List.forall2 (fun s i -> 0 <= i && i < s) shp idx) then 
            failwithf "index %A out of range for shape %A" idx shp

    /// address of element
    let inline addr idx a =
        checkIndex a.Shape idx
        Seq.map2 (*) idx a.Stride |> Seq.fold (+) a.Offset
  
    /// shape 
    let inline shape a = a.Shape

    /// stride
    let inline stride a = a.Stride

    /// offset 
    let inline offset a = a.Offset

    /// number of dimensions
    let inline nDims a = List.length (shape a)

    /// number of elements 
    let inline nElems a = List.fold (*) 1 (shape a)

    /// sequence of all indices 
    let inline allIdx a =
        let rec generate shp = seq {
            match shp with
            | l::ls ->
                for i=0 to l - 1 do
                    for is in generate ls do
                        yield i::is
            | [] -> yield []
        } 
        generate (shape a)

    /// all indices of the given dimension
    let inline allIdxOfDim dim a =
        { 0 .. a.Shape.[dim] - 1}

    /// computes the stride given the shape for the ArrayND to be continguous (row-major)
    let rec contiguousStride (shape: int list) =
        match shape with
        | [] -> []
        | [l] -> [1]
        | l::(lp::lrest) ->
            match contiguousStride (lp::lrest) with 
            | sp::srest -> (lp*sp)::sp::srest
            | [] -> failwith "unexpected"    

    /// computes the stride given the shape for the ArrayND to be in Fortran order (column-major)
    let inline columnMajorStride (shape: int list) =
        let rec buildStride elemsLeft shape =
            match shape with
            | [] -> []
            | l :: ls ->
                elemsLeft :: buildStride (l * elemsLeft) ls
        buildStride 1 shape

    /// a contiguous (row-major) ArrayND layout of the given shape 
    let inline newContiguous shp =
        {Shape=shp; Stride=contiguousStride shp; Offset=0;}

    /// a Fortran (column-major) ArrayND layout of the given shape 
    let inline newColumnMajor shp =
        {Shape=shp; Stride=columnMajorStride shp; Offset=0;}

    /// true if the ArrayND is contiguous
    let inline isContiguous a = (stride a = contiguousStride (shape a))

    /// true if the ArrayND is in Fortran order
    let inline isColumnMajor a = (stride a = columnMajorStride (shape a))

    /// adds a new dimension of size one to the left
    let inline padLeft a =
        {a with Shape=1::a.Shape; Stride=0::a.Stride}

    /// adds a new dimension of size one to the right
    let inline padRight a =
        {a with Shape=a.Shape @ [1]; Stride=a.Stride @ [0]}

    /// broadcast the given dimensionto the given size
    let inline broadcastDim dim size a =
        if size < 0 then invalidArg "size" "size must be positive"
        match (shape a).[dim] with
        | 1 -> {a with Shape=List.set dim size a.Shape; Stride=List.set dim 0 a.Stride}
        | _ -> failwithf "dimension %d of shape %A must be of size 1 to broadcast" dim (shape a)

    /// pads shapes from the left until they have same rank
    let rec padToSame a b =
        if nDims a < nDims b then padToSame (padLeft a) b
        elif nDims b < nDims a then padToSame a (padLeft b)
        else a, b

    /// broadcasts to have the same size
    let inline broadcastToSame ain bin =
        let mutable a, b = padToSame ain bin
        for d = 0 to (nDims a) - 1 do
            match (shape a).[d], (shape b).[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim d bl a
            | al, bl when bl = 1 -> b <- broadcastDim d al b
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size" (shape ain) (shape bin)
        a, b

    /// broadcasts a ArrayND to the given shape
    let inline broadcastToShape bs ain =
        let bsDim = List.length bs
        if bsDim <> nDims ain then
            failwithf "shape %A has different rank than shape %A" bs (shape ain)

        let mutable a = ain
        for d = 0 to bsDim - 1 do
            match (shape a).[d], bs.[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim d bl a
            | _ -> failwithf "cannot broadcast shape %A to shape %A" (shape ain) bs
        a

    /// Reshape layout under the assumption that it is contiguous.
    /// The number of elements must not change.
    let inline reshape shp a =
        if not (isContiguous a) then
            invalidArg "a" "layout must be contiguous for reshape"
        let shpElems = List.fold (*) 1 shp
        if shpElems <> nElems a then
            failwithf "cannot reshape from shape %A (with %d elements) to shape %A (with %d elements)" 
                (shape a) (nElems a) shp shpElems
        {a with Shape=shp; Stride=contiguousStride shp;}

    /// swaps the given dimensions
    let inline swapDim ax1 ax2 a =
        let nElems = nElems a
        if not (0 <= ax1 && ax1 < nElems && 0 <= ax2 && ax2 < nElems) then
            failwithf "cannot swap dimension %d with %d of for shape %A" ax1 ax2 (shape a)
        let shp, str = shape a, stride a
        {a with Shape=shp |> List.set ax1 shp.[ax2] |> List.set ax2 shp.[ax1]; 
                Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1];}

    /// transposes the given layout of a matrix
    let inline transpose a =
        if nDims a <> 2 then failwithf "cannot transpose non-matrix of shape %A" (shape a)
        swapDim 0 1 a

    /// reorders the axes as specified
    let inline reorderAxes (newOrder: int list) a =
        {a with Shape = List.permute (fun i -> newOrder.[i]) a.Shape;
                Stride = List.permute (fun i -> newOrder.[i]) a.Stride;}

    /// creates a subview layout
    let rec view ranges a =
        let checkElementRange isEnd nElems i =
            let nElems = if isEnd then nElems + 1 else nElems
            if not (0 <= i && i < nElems) then
                failwithf "index %d out of range in slice %A for shape %A" i ranges (shape a)
        let failIncompatible () =
            failwithf "slice %A is incompatible with shape %A" ranges (shape a)

        let rec recView ranges a =
            match ranges, a.Shape, a.Stride with
            | AllFill::rRanges, _::rShps, _ when List.length rShps > List.length rRanges ->
                recView (All :: AllFill :: rRanges) a
            | AllFill::rRanges, _::rShps, _ when List.length rShps = List.length rRanges ->
                recView (All :: rRanges) a
            | AllFill::rRanges, _, _ ->
                recView rRanges a
            | (All | Elem _ | Rng _ as idx)::rRanges, shp::rShps, str::rStrs ->
                let ra = recView rRanges {a with Shape=rShps; Stride=rStrs} 
                match idx with 
                | All ->
                    {ra with Shape = shp::ra.Shape;
                             Stride = str::ra.Stride}
                | Elem i -> 
                    checkElementRange false shp i
                    {ra with Offset = ra.Offset + i*str;
                             Stride = ra.Stride;
                             Shape = ra.Shape} 
                | Rng(start, stop) ->
                    checkElementRange false shp start
                    checkElementRange true shp stop
                    {ra with Offset = ra.Offset + start*str;
                             Shape = (stop - start)::ra.Shape;
                             Stride = str::ra.Stride} 
                | AllFill | NewAxis -> failwith "impossible"
            | NewAxis::rRanges, _, _ ->
                let ra = recView rRanges a
                {ra with Shape = 1::ra.Shape; 
                         Stride = 0::ra.Stride}
            | [], [], _ -> a 
            | _ -> failIncompatible ()         

        recView ranges a

    let allSourceRangesAndTargetIdxsForAxisReduction dim a =
        if not (0 <= dim && dim < nDims a) then
            failwithf "reduction dimension %d out of range for shape %A" dim (shape a)

        let rec generate shape dim = seq {
            match shape with
            | l::ls ->
                let rest = generate ls (dim-1)
                if dim = 0 then
                    for is, ws in rest do
                        yield All::is, ws
                else
                    for i=0 to l - 1 do
                        for is, ws in rest do
                            yield Elem i::is, i::ws
            | [] -> yield [], []
        } 
        generate (shape a) dim  
