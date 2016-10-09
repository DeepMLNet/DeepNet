namespace ArrayNDNS

open Basics


[<AutoOpen>]
module ArrayNDLayoutTypes =
    // layout (shape, offset, stride) of an ArrayND
    type ArrayNDLayoutT = {
        /// shape
        Shape:  int list
        /// offset in elements
        Offset: int
        /// stride in elements
        Stride: int list
    }

    /// range specification
    type RangeT = 
        /// single element
        | RngElem of int
        /// range from / to (including)
        | Rng of (int option) * (int option)
        /// insert broadcastable axis of size 1
        | RngNewAxis
        /// fill (...)
        | RngAllFill

    /// all elements
    let RngAll = Rng (None, None)


module ArrayNDLayout =

    /// checks that the layout is valid
    let inline check a =
        if a.Shape.Length <> a.Stride.Length then
            failwithf "shape and stride must have same number of entries: %A" a
        for s in a.Shape do
            if s < 0 then failwithf "shape cannot have negative entries: %A" a

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

    /// checks that the given axis is valid 
    let inline checkAxis ax a =
        if not (0 <= ax && ax < nDims a) then
            failwithf "axis %d out of range for array with shape %A" ax a.Shape

    /// a sequence of indicies enumerating all elements of the array with the given shape
    let rec allIdxOfShape shp = seq {
        match shp with
        | l::ls ->
            for i=0 to l - 1 do
                for is in allIdxOfShape ls do
                    yield i::is
        | [] -> yield []
    } 

    /// sequence of all indices 
    let inline allIdx a =
        allIdxOfShape (shape a)

    /// all indices of the given dimension
    let inline allIdxOfDim dim a =
        { 0 .. a.Shape.[dim] - 1}

    /// computes the stride given the shape for the ArrayND to be continguous (row-major)
    let rec cStride (shape: int list) =
        match shape with
        | [] -> []
        | [l] -> [1]
        | l::(lp::lrest) ->
            match cStride (lp::lrest) with 
            | sp::srest -> (lp*sp)::sp::srest
            | [] -> failwith "unexpected"    

    /// computes the stride given the shape for the ArrayND to be in Fortran order (column-major)
    let fStride (shape: int list) =
        let rec buildStride elemsLeft shape =
            match shape with
            | [] -> []
            | l :: ls ->
                elemsLeft :: buildStride (l * elemsLeft) ls
        buildStride 1 shape

    /// a contiguous (row-major) ArrayND layout of the given shape 
    let newC shp =
        {Shape=shp; Stride=cStride shp; Offset=0;}

    /// a Fortran (column-major) ArrayND layout of the given shape 
    let newF shp =
        {Shape=shp; Stride=fStride shp; Offset=0;}

    /// an ArrayND layout for an empty (zero elements) vector (1D)
    let emptyVector =
        {Shape=[0]; Stride=[1]; Offset=0;}

    /// True if strides are equal at all dimensions with size > 1.
    let stridesEqual (shp: int list) (aStr: int list) (bStr: int list) =
        List.zip3 shp aStr bStr
        |> List.forall (fun (s, a, b) -> if s > 1 then a = b else true)

    /// true if the ArrayND is contiguous
    let isC a = 
        stridesEqual a.Shape (stride a) (cStride a.Shape)

    /// true if the ArrayND is in Fortran order
    let isF a = 
        stridesEqual a.Shape (stride a) (fStride a.Shape)

    /// true if the memory of the ArrayND is a contiguous block
    let hasContiguousMemory a =
        isC a || isF a
        // TODO: extend to any memory ordering

    /// adds a new dimension of size one to the left
    let padLeft a =
        {a with Shape=1::a.Shape; Stride=0::a.Stride}

    /// adds a new dimension of size one to the right
    let padRight a =
        {a with Shape=a.Shape @ [1]; Stride=a.Stride @ [0]}

    /// cuts one dimension from the left
    let cutLeft a =
        if nDims a = 0 then failwith "cannot remove dimensions from scalar"
        {a with Shape=a.Shape.[1..]; Stride=a.Stride.[1..]}

    /// cuts one dimension from the right
    let cutRight a =
        if nDims a = 0 then failwith "cannot remove dimensions from scalar"
        let nd = nDims a
        {a with Shape=a.Shape.[.. nd-2]; Stride=a.Stride.[.. nd-2]}       

    /// broadcast the given dimension to the given size
    let broadcastDim dim size a =
        if size < 0 then invalidArg "size" "size must be positive"
        match (shape a).[dim] with
        | 1 -> {a with Shape=List.set dim size a.Shape; Stride=List.set dim 0 a.Stride}
        | _ -> failwithf "dimension %d of shape %A must be of size 1 to broadcast" dim (shape a)

    /// pads shapes from the left until they have same rank
    let rec padToSame a b =
        if nDims a < nDims b then padToSame (padLeft a) b
        elif nDims b < nDims a then padToSame a (padLeft b)
        else a, b

    /// pads shapes from the left until they have same rank
    let rec padToSameMany sas =
        let nDimsNeeded = sas |> List.map nDims |> List.max
        sas 
        |> List.map (fun sa ->
            let mutable sa = sa
            while nDims sa < nDimsNeeded do
                sa <- padLeft sa
            sa)

    /// cannot broadcast to same shape
    exception CannotBroadcast of string

    /// broadcasts to have the same size in the given dimensions
    let broadcastToSameInDims dims ain bin =
        let mutable a, b = ain, bin
        for d in dims do
            if not (d < nDims a && d < nDims b) then
                sprintf "cannot broadcast shapes %A and %A in non-existant dimension %d" 
                    (shape ain) (shape bin) d |> CannotBroadcast |> raise                    
            match (shape a).[d], (shape b).[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim d bl a
            | al, bl when bl = 1 -> b <- broadcastDim d al b
            | _ -> 
                sprintf "cannot broadcast shapes %A and %A to same size in dimensions %A" 
                    (shape ain) (shape bin) dims |> CannotBroadcast |> raise
        a, b       

    /// broadcasts to have the same size in the given dimensions    
    let broadcastToSameInDimsMany dims sas =
        let mutable sas = sas
        for d in dims do
            if not (sas |> List.forall (fun sa -> d < nDims sa)) then
                sprintf "cannot broadcast shapes %A to same size in non-existant dimension %d" sas d
                |> CannotBroadcast |> raise 
            let ls = sas |> List.map (fun sa -> sa.Shape.[d])
            if ls |> List.exists ((=) 1) then
                let nonBc = ls |> List.filter (fun l -> l <> 1)
                match Set nonBc |> Set.count with
                | 0 -> ()
                | 1 ->
                    let target = List.head nonBc
                    sas <- sas |> List.map (fun sa -> 
                        if sa.Shape.[d] <> target then sa |> broadcastDim d target
                        else sa)
                | _ ->
                    sprintf "cannot broadcast shapes %A to same size in dimension %d because \
                             they don't agree in the target size" sas d  
                             |> CannotBroadcast |> raise              
            elif Set ls |> Set.count > 1 then
                failwithf "non-broadcast dimension %d of shapes %A does not agree" d sas
        sas

    /// broadcasts to have the same size
    let broadcastToSame ain bin =
        let a, b = padToSame ain bin
        try
            broadcastToSameInDims [0..nDims a - 1] a b
        with CannotBroadcast _ ->
            sprintf "cannot broadcast shapes %A and %A to same size" (shape ain) (shape bin)
            |> CannotBroadcast |> raise

    /// broadcasts to have the same size
    let broadcastToSameMany sas =
        match sas with
        | [] -> []
        | _ ->
            let sas = padToSameMany sas
            try
                broadcastToSameInDimsMany [0 .. (nDims sas.Head - 1)] sas
            with CannotBroadcast _ ->
                sprintf "cannot broadcast shapes %A to same size" (sas |> List.map shape)
                |> CannotBroadcast |> raise

    /// broadcasts a ArrayND to the given shape
    let broadcastToShape bs ain =
        let bsDim = List.length bs
        if bsDim < nDims ain then
            failwithf "cannot broadcast to shape %A from shape %A of higher rank" bs (shape ain)        

        let mutable a = ain
        while nDims a < bsDim do
            a <- padLeft a
        for d = 0 to bsDim - 1 do
            match (shape a).[d], bs.[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim d bl a
            | _ -> failwithf "cannot broadcast shape %A to shape %A" (shape ain) bs
        a

    /// returns true if at least one dimension is broadcasted
    let isBroadcasted a =
        (shape a, stride a)
        ||> List.exists2 (fun shp str -> str = 0 && shp > 1)

    /// Reshape layout under the assumption that it is contiguous.
    /// The number of elements must not change.
    let reshape shp a =
        if not (isC a) then
            invalidArg "a" "layout must be contiguous for reshape"

        let shp =
            match List.filter ((=) -1) shp |> List.length with
            | 0 -> shp
            | 1 ->
                let elemsSoFar = List.fold (*) -1 shp
                let elemsNeeded = nElems a
                if elemsNeeded % elemsSoFar = 0 then
                    List.map (fun s -> if s = -1 then elemsNeeded / elemsSoFar else s) shp
                else
                    failwithf "cannot reshape from %A to %A because %d / %d is not an integer" 
                        (shape a) shp elemsNeeded elemsSoFar
            | _ -> failwithf "only the size of one dimension can be determined automatically, but shape was %A" shp
          
        let shpElems = List.fold (*) 1 shp
        if shpElems <> nElems a then
            failwithf "cannot reshape from shape %A (with %d elements) to shape %A (with %d elements)" 
                (shape a) (nElems a) shp shpElems
        {a with Shape=shp; Stride=cStride shp}

    /// swaps the given dimensions
    let swapDim ax1 ax2 a =
        if not (0 <= ax1 && ax1 < nDims a && 0 <= ax2 && ax2 < nDims a) then
            failwithf "cannot swap dimension %d with %d of for shape %A" ax1 ax2 (shape a)
        let shp, str = shape a, stride a
        {a with Shape=shp |> List.set ax1 shp.[ax2] |> List.set ax2 shp.[ax1]; 
                Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1];}

    /// Transposes the given layout of a matrix.
    /// If the array has more then two dimensions, the last two axes are swapped.
    let transpose a =
        let nd = nDims a
        if nd < 2 then failwithf "cannot transpose non-matrix of shape %A" (shape a)
        swapDim (nd-2) (nd-1) a

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    let permuteAxes (permut: int list) a =
        if nDims a <> List.length permut then
            failwithf "permutation %A must have same rank as shape %A" permut (shape a)
        {a with Shape = List.permute (fun i -> permut.[i]) a.Shape
                Stride = List.permute (fun i -> permut.[i]) a.Stride}

    /// Reverses the elements in the specified dimension.
    let reverseAxis ax a =
        checkAxis ax a
        {a with Offset = a.Offset + (a.Shape.[ax] - 1) * a.Stride.[ax]
                Stride = a.Stride |> List.set ax (-a.Stride.[ax])}

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
            | RngAllFill::rRanges, _::rShps, _ when List.length rShps > List.length rRanges ->
                recView (RngAll :: RngAllFill :: rRanges) a
            | RngAllFill::rRanges, _::rShps, _ when List.length rShps = List.length rRanges ->
                recView (RngAll :: rRanges) a
            | RngAllFill::rRanges, _, _ ->
                recView rRanges a
            | (RngElem _ | Rng _ as idx)::rRanges, shp::rShps, str::rStrs ->
                let ra = recView rRanges {a with Shape=rShps; Stride=rStrs} 
                match idx with 
                | RngElem i -> 
                    checkElementRange false shp i
                    {ra with Offset = ra.Offset + i*str;
                             Stride = ra.Stride;
                             Shape = ra.Shape} 
                | Rng(start, stop) ->
                    let start = defaultArg start 0
                    let stop = defaultArg stop (shp - 1)
                    if start = stop + 1 then
                        // allow slices starting at the element past the last 
                        // element and are empty
                        checkElementRange true shp start
                    else
                        checkElementRange false shp start
                    checkElementRange true shp stop
                    {ra with Offset = ra.Offset + start*str;
                                Shape = (stop + 1 - start)::ra.Shape;
                                Stride = str::ra.Stride} 
                | RngAllFill | RngNewAxis -> failwith "impossible"
            | RngNewAxis::rRanges, _, _ ->
                let ra = recView rRanges a
                {ra with Shape = 1::ra.Shape; 
                         Stride = 0::ra.Stride}
            | [], [], _ -> a 
            | _ -> failIncompatible ()         

        recView ranges a

    let allSrcRngsAndTrgtIdxsForAxisReduce dim a =
        if not (0 <= dim && dim < nDims a) then
            failwithf "reduction dimension %d out of range for shape %A" dim (shape a)

        let rec generate shape dim = seq {
            match shape with
            | l::ls ->
                let rest = generate ls (dim-1)
                if dim = 0 then
                    for is, ws in rest do
                        yield RngAll::is, ws
                else
                    for i=0 to l - 1 do
                        for is, ws in rest do
                            yield RngElem i::is, i::ws
            | [] -> yield [], []
        } 
        generate (shape a) dim  


    /// Creates a layout that extracts the diagonal along the given axes.
    /// The first axis is replaced with the diagonal and the second axis is removed.
    let diagAxis ax1 ax2 a =
        checkAxis ax1 a
        checkAxis ax2 a
        if ax1 = ax2 then failwithf "axes to use for diagonal must be different"
        if a.Shape.[ax1] <> a.Shape.[ax2] then
            failwithf "array must have same dimensions along axis %d and %d to extract diagonal \
                       but it has shape %A" ax1 ax2 a.Shape
              
        let newShape, newStride = 
            [for ax, (sh, st) in List.indexed (List.zip a.Shape a.Stride) do
                match ax with
                | _ when ax=ax1 -> yield sh, a.Stride.[ax1] + a.Stride.[ax2]
                | _ when ax=ax2 -> ()
                | _ -> yield sh, st
            ] |> List.unzip                
        {a with Shape=newShape; Stride=newStride}

