namespace Tensor

open System

open Tensor.Utils



/// cannot broadcast to same shape
exception CannotBroadcast of msg:string with override __.Message = __.msg

/// invalid tensor layout specification
exception InvalidTensorLayout of msg:string with override __.Message = __.msg

/// the layout of this tensor makes this operation impossible without copying it
exception ImpossibleWithoutCopy of msg:string with override __.Message = __.msg



/// Layout (shape, offset, stride) of a Tensor.
type TensorLayout = {
    /// Shape.
    Shape:  int64 list
    /// Offset (to first element) in elements.
    Offset: int64
    /// Stride in elements.
    Stride: int64 list
} with
    /// Number of dimensions.
    member this.NDims = List.length this.Shape
    /// Number of elements.
    member this.NElems = List.fold (*) 1L this.Shape



/// Functions for working with TensorLayout.
module TensorLayout =

    /// checks that the layout is valid
    let inline check a =
        if a.Shape.Length <> a.Stride.Length then
            let msg = 
                sprintf "shape %A and stride %A must have same number of entries"
                        a.Shape a.Stride
            raise (InvalidTensorLayout msg)
        for s in a.Shape do
            if s < 0L then 
                let msg = sprintf "shape %A cannot have negative entires" a.Shape
                raise (InvalidTensorLayout msg)

    /// checks that the given index is valid for the given shape
    let inline checkIndex shp idx =
        if List.length shp <> List.length idx then
            let msg =
                sprintf "index %A has other dimensionality than tensor of shape %A" idx shp
            raise (IndexOutOfRangeException msg)
        if not (List.forall2 (fun s i -> 0L <= i && i < s) shp idx) then 
            let msg =
                sprintf "index %A out of range for tensor of shape %A" idx shp
            raise (IndexOutOfRangeException msg)

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
    let inline nElems a = List.fold (*) 1L (shape a)

    /// checks that the given axis is valid 
    let inline checkAxis ax a =
        if not (0 <= ax && ax < nDims a) then
            let msg =
                sprintf "axis %d out of range for tensor with shape %A" ax a.Shape
            raise (IndexOutOfRangeException msg)
   
    /// a sequence of indicies enumerating all elements of the array with the given shape
    let rec allIdxOfShape shp = seq {
        match shp with
        | l::ls ->
            for i=0L to l - 1L do
                for is in allIdxOfShape ls do
                    yield i::is
        | [] -> yield []
    } 

    /// sequence of all indices 
    let inline allIdx a =
        allIdxOfShape (shape a)

    /// all indices of the given dimension
    let inline allIdxOfDim dim a =
        { 0L .. a.Shape.[dim] - 1L}

    /// Computes the strides for the given shape using the specified ordering.
    /// The axis that is first in the ordering gets stride 1.
    /// The resulting strides will be independent of the shape of the axis 
    /// that appears last in the ordering.
    /// A C-order stride corresponds to the ordering: [n; n-1; ...; 2; 1; 0].
    /// A Fortran-order stride corresponds to the ordering: [0; 1; 2; ...; n-1; n].
    let orderedStride (shape: int64 list) (order: int list) =
        if not (Permutation.is order) then
            invalidArg "order" (sprintf "the stride order %A is not a permutation" order)
        if order.Length <> shape.Length then
            invalidArg "order" (sprintf "the stride order %A is incompatible with the shape %A" order shape)
        let rec build cumElems order =
            match order with
            | o :: os -> cumElems :: build (cumElems * shape.[o]) os
            | [] -> []
        build 1L order |> List.permute (fun i -> order.[i])

    /// computes the stride given the shape for the ArrayND to be in C-order (row-major)
    let cStride (shape: int64 list) =
        orderedStride shape (List.rev [0 .. shape.Length-1])

    /// computes the stride given the shape for the ArrayND to be in Fortran-order (column-major)
    let fStride (shape: int64 list) =
        orderedStride shape [0 .. shape.Length-1]

    /// a ArrayND layout of the given shape and stride order
    let newOrdered shp strideOrder =
        {Shape=shp; Stride=orderedStride shp strideOrder; Offset=0L}

    /// a C-order (row-major) ArrayND layout of the given shape 
    let newC shp =
        {Shape=shp; Stride=cStride shp; Offset=0L}

    /// a Fortran-order (column-major) ArrayND layout of the given shape 
    let newF shp =
        {Shape=shp; Stride=fStride shp; Offset=0L}

    /// an ArrayND layout for an empty (zero elements) vector (1D)
    let emptyVector =
        {Shape=[0L]; Stride=[1L]; Offset=0L}

    /// True if strides are equal at all dimensions with size > 1.
    let stridesEqual (shp: int64 list) (aStr: int64 list) (bStr: int64 list) =
        List.zip3 shp aStr bStr
        |> List.forall (fun (s, a, b) -> if s > 1L then a = b else true)

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
        {a with Shape=1L::a.Shape; Stride=0L::a.Stride}

    /// adds a new dimension of size one to the right
    let padRight a =
        {a with Shape=a.Shape @ [1L]; Stride=a.Stride @ [0L]}

    /// Inserts an axis of size 1 before the specified position.
    let insertAxis ax a =
        if not (0 <= ax && ax <= nDims a) then
            let msg =
                sprintf "axis %d out of range for tensor with shape %A" ax a.Shape
            raise (IndexOutOfRangeException msg)
        {a with Shape = a.Shape |> List.insert ax 1L
                Stride = a.Stride |> List.insert ax 0L}        

    /// cuts one dimension from the left
    let cutLeft a =
        if nDims a = 0 then invalidArg "a" "cannot remove dimensions from scalar"
        {a with Shape=a.Shape.[1..]; Stride=a.Stride.[1..]}

    /// cuts one dimension from the right
    let cutRight a =
        if nDims a = 0 then invalidArg "a" "cannot remove dimensions from scalar"
        let nd = nDims a
        {a with Shape=a.Shape.[.. nd-2]; Stride=a.Stride.[.. nd-2]}       

    /// broadcast the given dimension to the given size
    let broadcastDim dim size a =
        if size < 0L then invalidArg "size" "size must be positive"
        match (shape a).[dim] with
        | 1L -> {a with Shape=List.set dim size a.Shape; Stride=List.set dim 0L a.Stride}
        | _ -> 
            let msg = 
                sprintf "dimension %d of shape %A must be of size 1 to broadcast" dim (shape a)
            raise (CannotBroadcast msg)

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

    /// broadcasts to have the same size in the given dimensions
    let broadcastToSameInDims dims ain bin =
        let mutable a, b = ain, bin
        for d in dims do
            if not (d < nDims a && d < nDims b) then
                let msg = 
                    sprintf "cannot broadcast shapes %A and %A in non-existant dimension %d" 
                        (shape ain) (shape bin) d 
                raise (CannotBroadcast msg)
            match (shape a).[d], (shape b).[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1L -> a <- broadcastDim d bl a
            | al, bl when bl = 1L -> b <- broadcastDim d al b
            | _ -> 
                let msg = 
                    sprintf "cannot broadcast shapes %A and %A to same size in dimensions %A" 
                        (shape ain) (shape bin) dims
                raise (CannotBroadcast msg)
        a, b       

    /// broadcasts to have the same size in the given dimensions    
    let broadcastToSameInDimsMany dims sas =
        let mutable sas = sas
        for d in dims do
            if not (sas |> List.forall (fun sa -> d < nDims sa)) then
                let msg =
                    sprintf "cannot broadcast shapes %A to same size in non-existant dimension %d" sas d
                raise (CannotBroadcast msg)
            let ls = sas |> List.map (fun sa -> sa.Shape.[d])
            if ls |> List.exists ((=) 1L) then
                let nonBc = ls |> List.filter (fun l -> l <> 1L)
                match Set nonBc |> Set.count with
                | 0 -> ()
                | 1 ->
                    let target = List.head nonBc
                    sas <- sas |> List.map (fun sa -> 
                        if sa.Shape.[d] <> target then sa |> broadcastDim d target
                        else sa)
                | _ ->
                    let msg =
                        sprintf "cannot broadcast shapes %A to same size in dimension %d because \
                                 they don't agree in the target size" sas d  
                    raise (CannotBroadcast msg)
            elif Set ls |> Set.count > 1 then
                let msg =
                    sprintf "non-broadcast dimension %d of shapes %A does not agree" d sas
                raise (CannotBroadcast msg)
        sas

    /// broadcasts to have the same size
    let broadcastToSame ain bin =
        let a, b = padToSame ain bin
        try
            broadcastToSameInDims [0..nDims a - 1] a b
        with CannotBroadcast _ ->
            let msg =
                sprintf "cannot broadcast shapes %A and %A to same size" (shape ain) (shape bin)
            raise (CannotBroadcast msg)

    /// broadcasts to have the same size
    let broadcastToSameMany sas =
        match sas with
        | [] -> []
        | _ ->
            let sas = padToSameMany sas
            try
                broadcastToSameInDimsMany [0 .. (nDims sas.Head - 1)] sas
            with CannotBroadcast _ ->
                let msg =
                    sprintf "cannot broadcast shapes %A to same size" (sas |> List.map shape)
                raise (CannotBroadcast msg)

    /// broadcasts a tensor to the given shape
    let broadcastToShape bs ain =
        let bsDim = List.length bs
        if bsDim < nDims ain then
            let msg = 
                sprintf "cannot broadcast to shape %A from shape %A of higher rank" bs (shape ain)        
            raise (CannotBroadcast msg)

        let mutable a = ain
        while nDims a < bsDim do
            a <- padLeft a
        for d = 0 to bsDim - 1 do
            match (shape a).[d], bs.[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1L -> a <- broadcastDim d bl a
            | _ -> 
                let msg =
                    sprintf "cannot broadcast shape %A to shape %A" (shape ain) bs
                raise (CannotBroadcast msg)
        a

    /// returns true if at least one dimension is broadcasted
    let isBroadcasted a =
        (shape a, stride a)
        ||> List.exists2 (fun shp str -> str = 0L && shp > 1L)

    /// Reshape layout under the assumption that it is contiguous.
    /// The number of elements must not change.
    /// Returns Some newLayout when reshape is possible without copy
    /// Returns None when a copy is required.
    let tryReshape shp a =
        // replace on occurence of "Remainder" in new shape with required size to keep number of
        // elements constant
        let shp =
            match shp |> List.filter ((=) Remainder) |> List.length with
            | 0 -> shp
            | 1 ->
                let elemsSoFar = 
                    shp 
                    |> List.filter ((<>) Remainder)
                    |> List.fold (*) 1L 
                let elemsNeeded = nElems a
                if elemsNeeded % elemsSoFar = 0L then
                    shp |> List.map (fun s -> if s = Remainder then elemsNeeded / elemsSoFar else s) 
                else
                    invalidArg "shp"
                        (sprintf "cannot reshape from %A to %A because %d / %d is not an integer" 
                                 (shape a) shp elemsNeeded elemsSoFar)
            | _ -> 
                invalidArg "shp"
                    (sprintf "only the size of one dimension can be determined automatically, but shape was %A" shp)
          
        // check that number of elements does not change
        let shpElems = List.fold (*) 1L shp
        if shpElems <> nElems a then
            invalidArg "shp"
                (sprintf "cannot reshape from shape %A (with %d elements) to shape %A (with %d elements)" 
                         (shape a) (nElems a) shp shpElems)

        // try to transform stride using singleton insertions and removals
        let rec tfStride newStr newShp aStr aShp =
            match newShp, aStr, aShp with
            | nSize::newShps, aStr::aStrs, aSize::aShps when nSize=aSize ->
                tfStride (newStr @ [aStr]) newShps aStrs aShps
            | 1L::newShps, _, _ ->
                tfStride (newStr @ [0L]) newShps aStr aShp
            | _, _::aStrs, 1L::aShps ->
                tfStride newStr newShp aStrs aShps
            | [], [], [] -> Some newStr
            | _ -> None

        match tfStride [] shp a.Stride a.Shape with
        | _ when isC a -> Some {a with Shape=shp; Stride=cStride shp}
        | Some newStr -> 
            //printfn "Using stride transform to reshape from\n%A\nto\n%A\n" a {a with Shape=shp; Stride=newStr}
            Some {a with Shape=shp; Stride=newStr}
        | None -> None

    /// Returns true if a can be reshaped into shp without copying.
    /// The number of elements must not change.
    let canReshape shp a =
        match tryReshape shp a with
        | Some _ -> true
        | None -> false

    /// Reshape layout under the assumption that it is contiguous.
    /// The number of elements must not change.
    /// An error is raised, if reshape is impossible without copying.
    let reshape shp a =
        match tryReshape shp a with
        | Some layout -> layout
        | None -> 
            let msg =
                sprintf "cannot reshape layout %A into shape %A without copying" a shp
            raise (ImpossibleWithoutCopy msg)

    /// swaps the given dimensions
    let swapDim ax1 ax2 a =
        if not (0 <= ax1 && ax1 < nDims a && 0 <= ax2 && ax2 < nDims a) then
            invalidArg "ax1" 
                (sprintf "cannot swap dimension %d with %d of for shape %A" ax1 ax2 (shape a))
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
            invalidArg "permut"
                (sprintf "permutation %A must have same rank as shape %A" permut (shape a))
        {a with Shape = List.permute (fun i -> permut.[i]) a.Shape
                Stride = List.permute (fun i -> permut.[i]) a.Stride}

    /// Reverses the elements in the specified dimension.
    let reverseAxis ax a =
        checkAxis ax a
        {a with Offset = a.Offset + (a.Shape.[ax] - 1L) * a.Stride.[ax]
                Stride = a.Stride |> List.set ax (-a.Stride.[ax])}

    /// creates a subview layout
    let rec view ranges a =
        let checkElementRange isEnd nElems i =
            let nElems = if isEnd then nElems + 1L else nElems
            if not (0L <= i && i < nElems) then
                let msg =
                    sprintf "index %d out of range in slice %A for shape %A" i ranges (shape a)
                raise (IndexOutOfRange msg)
        let failIncompatible () =
            let msg =
                sprintf "slice %A is incompatible with shape %A" ranges (shape a)
            raise (IndexOutOfRange msg)

        let rec recView ranges a =
            match ranges, a.Shape, a.Stride with
            | RngAllFill::rRanges, _::rShps, _ when List.length rShps > List.length rRanges ->
                recView (RngAll::RngAllFill::rRanges) a
            | RngAllFill::rRanges, _::rShps, _ when List.length rShps = List.length rRanges ->
                recView (RngAll::rRanges) a
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
                    let start = defaultArg start 0L
                    let stop = defaultArg stop (shp - 1L)
                    if stop >= start then
                        // non-empty slice
                        checkElementRange false shp start
                        checkElementRange true shp stop
                        {ra with Offset = ra.Offset + start*str;
                                  Shape = (stop + 1L - start)::ra.Shape;
                                 Stride = str::ra.Stride} 
                    else
                        // empty slice
                        // We allow start and stop to be out of range in this case.
                        {ra with Offset = ra.Offset;
                                  Shape = 0L::ra.Shape;
                                 Stride = str::ra.Stride} 
                | RngAllFill | RngNewAxis -> failwith "impossible"
            | RngNewAxis::rRanges, _, _ ->
                let ra = recView rRanges a
                {ra with Shape = 1L::ra.Shape; 
                         Stride = 0L::ra.Stride}
            | [], [], _ -> a 
            | _ -> failIncompatible ()         

        recView ranges a

    let allSrcRngsAndTrgtIdxsForAxisReduce dim a =
        if not (0 <= dim && dim < nDims a) then
            invalidArg "dim" 
                (sprintf "reduction dimension %d out of range for shape %A" dim (shape a))

        let rec generate shape dim = seq {
            match shape with
            | l::ls ->
                let rest = generate ls (dim-1)
                if dim = 0 then
                    for is, ws in rest do
                        yield RngAll::is, ws
                else
                    for i=0L to l - 1L do
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
        if ax1 = ax2 then 
            invalidArg "ax1" "axes to use for diagonal must be different"
        if a.Shape.[ax1] <> a.Shape.[ax2] then
            invalidArg "a" 
                (sprintf "array must have same dimensions along axis %d and %d to extract diagonal 
                          but it has shape %A" ax1 ax2 a.Shape)
              
        let newShape, newStride = 
            [for ax, (sh, st) in List.indexed (List.zip a.Shape a.Stride) do
                match ax with
                | _ when ax=ax1 -> yield sh, a.Stride.[ax1] + a.Stride.[ax2]
                | _ when ax=ax2 -> ()
                | _ -> yield sh, st
            ] |> List.unzip                
        {a with Shape=newShape; Stride=newStride}

    /// Computes the linear index of a given index.
    /// Linear indexing is performed in row-major order.
    let idxToLinear a idx =
        checkIndex a.Shape idx
        List.map2 (*) idx (cStride a.Shape) |> List.sum

    /// Computes the index of a given linear index.
    let linearToIdx a linear = 
        let idx =
            (linear, cStride a.Shape) 
            |> List.unfold (fun (l, str) ->
                match str with
                | s::rs -> Some (l / s, (l % s, rs))
                | _ -> None)
        checkIndex a.Shape idx
        idx

            