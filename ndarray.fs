module NDArray
open Util


/// an N-dimensional array with reshape and subview abilities
type NDArray = 
    {Shape: int list;
     Stride: int list; 
     Offset: int;
     Data: float[]}


////////////////////////////////////////////////////////////////////////////////////////////////
// element access
////////////////////////////////////////////////////////////////////////////////////////////////

/// checks that the given index is valid for the given shape
let checkIndex shp idx =
    if List.length shp <> List.length idx then
        failwithf "index %A has other dimensionality than shape %A" idx shp
    if not (List.forall2 (fun s i -> 0 <= i && i < s) shp idx) then 
        failwithf "index %A out of range for shape %A" idx shp
    
/// address of element
let addr idx a =
    checkIndex a.Shape idx
    Seq.map2 (*) idx a.Stride |> Seq.fold (+) a.Offset
    
/// get element value
let get idx a =
    a.Data.[addr idx a]
    
/// set element value
let set idx value a =
    a.Data.[addr idx a] <- value


////////////////////////////////////////////////////////////////////////////////////////////////
// shape functions
////////////////////////////////////////////////////////////////////////////////////////////////
            
/// number of dimensions of a NDArray
let nDim a = List.length a.Shape
    
/// shape of NDArray
let shape a = a.Shape

/// stride of NDArray
let stride a = a.Stride

/// offset of NDArray
let offset a = a.Offset

/// number of elements of shape
let nElems shp =
    List.fold (*) 1 shp

/// sequence of all indices of a NDArray of shape shp
let rec allIdx shp =
    seq {
        match shp with
            | l::ls ->
                for i=0 to l - 1 do
                    for is in allIdx ls do
                        yield i::is
            | [] -> yield []
    } 

/// all indices of the given dimension
let allIdxOfDim dim (shp: int list) =
    { 0 .. shp.[dim] - 1}

/// sequence of all elements of a NDArray
let allElems a =
    allIdx a.Shape |> Seq.map (fun i -> get i a)

/// computes the stride given the shape for the NDArray to be continguous (row-major)
let rec contiguousStride (shape: int list) =
    match shape with
        | [] -> []
        | [l] -> [1]
        | l::(lp::lrest) ->
            match contiguousStride (lp::lrest) with 
                | sp::srest -> (lp*sp)::sp::srest
                | [] -> failwith "unexpected"    

/// true if the NDArray is continguous
let isContiguous (a: NDArray) =
    a.Stride = contiguousStride a.Shape    

/// creates a new continguous NDArray of the given shape and fills it with zeros
let newContinguous shp =
    {Shape=shp; Stride=contiguousStride shp; Offset=0; Data=Array.zeroCreate (nElems shp)}

/// checks that two NDArrays have the same shape
let checkSameShape a b =
    if a.Shape <> b.Shape then
        failwithf "cannot apply operation to NDArrays of different shapes %A and %A" a.Shape b.Shape

/// Copies all elements from source to destination.
/// Both NDArrays must have the same shape.
let copyTo source dest =
    checkSameShape source dest
    for idx in allIdx source.Shape do
        set idx (get idx source) dest

/// Returns a continguous copy of the given NDArray.
let copy source =
    let dest = newContinguous source.Shape
    copyTo source dest
    dest

/// If the NDArray is not continguous, returns a continguous copy; otherwise
/// the given NDArray is returned unchanged.
let makeContinguous a =
    if isContiguous a then a else copy a

/// adds a new dimension of size one to the left
let padLeft a =
    {a with Shape=1::a.Shape; Stride=0::a.Stride}

/// adds a new dimension of size one to the right
let padRight a =
    {a with Shape=a.Shape @ [1]; Stride=a.Stride @ [0]}

/// broadcast the given dimension of an NDArray to the given size
let broadcastDim a dim size =
    if size < 0 then failwith "size must be positive"
    match (shape a).[dim] with
        | 1 -> {a with Shape=List.set dim size a.Shape; Stride=List.set dim 0 a.Stride}
        | _ -> failwithf "dimension %d of shape %A must be of size 1 to broadcast" dim (shape a)

/// pads shapes from the left until they have same rank
let rec padToSame a b =
    if nDim a < nDim b then
        padToSame (padLeft a) b
    elif nDim b < nDim a then
        padToSame a (padLeft b)
    else
        a, b

/// broadcasts two NDArrays to have the same size
let broadcastToSame ain bin =
    let mutable a, b = padToSame ain bin
    for d = 0 to (nDim a) - 1 do
        match (shape a).[d], (shape b).[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim a d bl
            | al, bl when bl = 1 -> b <- broadcastDim b d al
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size" (shape ain) (shape bin)
    a, b

/// broadcasts a NDArray to the given shape
let broadcastToShape bs ain =
    let bsDim = List.length bs
    if bsDim <> nDim ain then
        failwithf "shape %A has different rank than NDArray with shape %A" bs (shape ain)

    let mutable a = ain
    for d = 0 to bsDim - 1 do
        match (shape a).[d], bs.[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim a d bl
            | _ -> failwithf "cannot broadcast NDArray with shape %A to shape %A" (shape ain) bs
    a

/// Reshape array under the assumption that it is continguous.
/// The number of elements must not change.
let reshape shp a =
    if nElems shp <> nElems (shape a) then
        failwithf "cannot reshape from %A (with %d elements) to %A (with %d elements)" 
            (shape a) (nElems (shape a)) shp (nElems shp)
    {(makeContinguous a) with Shape=shp; Stride=contiguousStride shp}

/// swaps the given dimensions
let swapDim ax1 ax2 a =
    let nElems = nElems (shape a)
    if not (0 <= ax1 && ax1 < nElems && 0 <= ax2 && ax2 < nElems) then
        failwithf "cannot swap dimension %d with %d of array with shape %A" ax1 ax2 (shape a)
    let shp = shape a
    let nShp = shp |> List.set ax1 shp.[ax2] |> List.set ax2 shp.[ax1]
    let str = stride a
    let nStr = str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1]
    {a with Shape=nShp; Stride=nStr}

/// transposes the given matrix
let transpose a =
    if nDim a <> 2 then
        failwithf "cannot transpose array of shape %A" a
    swapDim 0 1 a

////////////////////////////////////////////////////////////////////////////////////////////////
// subtensor
////////////////////////////////////////////////////////////////////////////////////////////////

/// slice specification
type Slice = 
    | Elem of int
    | Rng of int * int
    | NewAxis
    | All

/// creates a subview of an NDArray
let rec view slices a =
    let checkElementRange nElems i =
        if not (0 <= i && i < nElems) then
            failwithf "index %d out of range in slice %A for array of shape %A" i slices (shape a)
    let failIncompatible () =
        failwithf "slice %A is incompatible with array of shape %A" slices (shape a)

    let rec recView slices a =
        match slices, a.Shape, a.Stride with
        | (All | Elem _ | Rng _ as idx)::rSlices, shp::rShps, str::rStrs ->
            let ra = recView rSlices {a with Shape=rShps; Stride=rStrs} 
            match idx with 
            | All ->
                {ra with Shape = shp::ra.Shape;
                         Stride = str::ra.Stride}
            | Elem i -> 
                checkElementRange shp i
                {ra with Offset = ra.Offset + i*str;
                         Stride = ra.Stride;
                         Shape = ra.Shape} 
            | Rng(start, stop) ->
                checkElementRange shp start
                checkElementRange shp stop
                {ra with Offset = ra.Offset + start*str;
                         Shape = (stop - start)::ra.Shape;
                         Stride = str::ra.Stride} 
            | NewAxis -> failwith "impossible"
        | NewAxis::rSlices, _, _ ->
            let ra = recView rSlices a
            {ra with Shape = 1::ra.Shape; 
                     Stride = 0::ra.Stride}
        | [], [], _ -> a 
        | _ -> failIncompatible ()         

    recView slices a


////////////////////////////////////////////////////////////////////////////////////////////////
// array creation functions
////////////////////////////////////////////////////////////////////////////////////////////////

/// NDArray with zero dimensions (scalar) and given value
let scalar f =
    let a = newContinguous [] 
    set [] f a
    a

/// NDArray of given shape filled with zeros.
let zeros shape =
    newContinguous shape
    
/// NDArray of same shape as a filled with zeros.
let zerosLike a =
    newContinguous (shape a)

/// NDArray of given shape filled with ones.
let ones shape =
    let a = newContinguous shape
    for idx in allIdx shape do
        set idx 1. a
    a

/// NDArray of same shape as a filled with ones.
let onesLike a =
    ones (shape a)

/// NDArray with ones on the diagonal of given shape.
let identity shape =
    let a = zeros shape
    let ndim = List.length shape
    for i = 0 to (List.min shape) - 1 do
        set (List.replicate ndim i) 1. a
    a

////////////////////////////////////////////////////////////////////////////////////////////////
// elementwise operations
////////////////////////////////////////////////////////////////////////////////////////////////   
      
/// Applies the given function elementwise to the given NDArray.
let elemwise (f: float -> float) a =
    let c = zerosLike a
    for idx in allIdx (shape a) do
        set idx (f (get idx a)) c
    c
            
/// Applies the given binary function elementwise to the two given NDArrays.
let elemwise2 (f: float -> float -> float) a b =
    let a, b = broadcastToSame a b
    let c = zerosLike a
    for idx in allIdx (shape a) do
        set idx (f (get idx a) (get idx b)) c
    c        
        
let add a b =
    elemwise2 (+) a b        
        
let substract a b =
    elemwise2 (-) a b        

let multiply a b =
    elemwise2 (*) a b        

let divide a b =
    elemwise2 (/) a b        

let negate a =
    elemwise (~-) a

let power a b =
    elemwise2 ( ** ) a b

let exp a =
    elemwise exp a

let log a =
    elemwise log a 


////////////////////////////////////////////////////////////////////////////////////////////////
// reduction operations
////////////////////////////////////////////////////////////////////////////////////////////////         

let allSourceSlicesAndTargetIdxsForAxisReduction dim a =
    if not (0 <= dim && dim < nDim a) then
        failwithf "reduction dimension %d out of range for array of shape %A" dim (shape a)

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
    generate a.Shape dim  
       
/// applies the given reduction function over the given dimension
let axisReduce (f: NDArray -> NDArray) dim a =
    let c = zeros (List.without dim a.Shape)
    for srcSlice, dstIdx in allSourceSlicesAndTargetIdxsForAxisReduction dim a do
        set dstIdx (f (view srcSlice a) |> get []) c
    c

/// elementwise sum
let sum a =
    allElems a |> Seq.sum |> scalar

/// elementwise sum over given axis
let sumAxis = axisReduce sum 
    
/// elementwise product
let product a =
    allElems a |> Seq.fold (*) 1. |> scalar

/// elementwise product over given axis
let productAxis = axisReduce product 

////////////////////////////////////////////////////////////////////////////////////////////////
// tensor operations
////////////////////////////////////////////////////////////////////////////////////////////////         

/// dot product between vec*vec, mat*vec, mat*mat
let rec dot a b =
    match nDim a, nDim b with
        | 1, 1 when shape a = shape b -> 
            multiply a b |> sum
        | 2, 1 when (shape a).[1] = (shape b).[0] -> 
            dot a (padRight b) |> view [All; Elem 0] 
        | 2, 2 when (shape a).[1] = (shape b).[0] ->
            let nI = a.Shape.[0]
            let nJ = a.Shape.[1]
            let nK = b.Shape.[1]
            let c = zeros [nI; nK]
            for k=0 to nK - 1 do
                for i=0 to nI - 1 do
                    let v = {0 .. nJ - 1}
                            |> Seq.map (fun j -> (get [i; j] a) * (get [j; k] b))
                            |> Seq.sum
                    set [i; k] v c
            c
        | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" 
                    (shape a) (shape b)

type BlockSpec =
    | Blocks of BlockSpec list
    | Arrays of NDArray list

/// array constructed of other arrays
let blockArray bs =

    let rec commonShape joinDim shps =               
        match shps with
        | [shp] ->
            List.set joinDim -1 shp
        | shp::rShps ->
            let commonShp = commonShape joinDim [shp]
            if commonShp <> commonShape joinDim rShps then
                failwithf "block array blocks must have same rank and be identical in all but the join dimension"
            commonShp
        | [] -> []

    let joinSize joinDim (shps: int list list) =
        shps |> List.map (fun shp -> shp.[joinDim]) |> List.sum

    let joinShape joinDim shps =
        commonShape joinDim shps 
            |> List.set joinDim (joinSize joinDim shps)

    let rec joinedBlocksShape joinDim bs =
        match bs with
        | Blocks blcks ->
            blcks |> List.map (joinedBlocksShape (joinDim + 1)) |> joinShape joinDim
        | Arrays arys ->
            arys |> List.map shape |> joinShape joinDim

    let rec blockPosAndContents (joinDim: int) startPos bs = seq {
        match bs with
        | Blocks blcks ->
            let mutable pos = startPos
            for blck in blcks do
                yield! blockPosAndContents (joinDim + 1) pos blck 
                let blckShape = joinedBlocksShape joinDim blck
                pos <- List.set joinDim (pos.[joinDim] + blckShape.[joinDim]) pos
        | Arrays arys ->
            let mutable pos = startPos
            for ary in arys do
                yield pos, ary
                pos <- List.set joinDim (pos.[joinDim] + (shape ary).[joinDim]) pos
    }
            
    let joinedShape = joinedBlocksShape 0 bs
    let joined = zeros joinedShape
    let startPos = List.replicate (List.length joinedShape) 0

    for pos, ary in blockPosAndContents 0 startPos bs do
        let slice = List.map2 (fun p s -> Rng(p, p + s)) pos (shape ary)
        let joinedSlice = joined |> view slice 
        copyTo ary joinedSlice

    joined

    
/// tensor product
let tensorProduct a b =
    let a, b = padToSame a b
    let aShp = shape a

    let rec generate pos = 
        match List.length pos with
        | dim when dim = nDim a - 1 ->
            let arys = seq {
                for p = 0 to aShp.[dim] - 1 do
                    let aElem = get (pos @ [p]) a
                    yield multiply (scalar aElem) b
            }                     
            Arrays (Seq.toList arys)
        | dim ->
            let blcks = 
                seq {for p in 0 .. aShp.[dim] - 1 -> generate (pos @ [p])}
            Blocks (Seq.toList blcks)   

    generate [] |> blockArray

////////////////////////////////////////////////////////////////////////////////////////////////
// operators
////////////////////////////////////////////////////////////////////////////////////////////////  

/// transpose
type T = T

type NDArray with

    // elementwise binary
    static member (+) (a: NDArray, b: NDArray) = add a b
    static member (-) (a: NDArray, b: NDArray) = substract a b    
    static member (*) (a: NDArray, b: NDArray) = multiply a b
    static member (/) (a: NDArray, b: NDArray) = divide a b
    static member Pow (a: NDArray, b: NDArray) = power a b

    // elementwise binary with float
    static member (+) (a: NDArray, b: float) = a + (scalar b)
    static member (-) (a: NDArray, b: float) = a - (scalar b)
    static member (*) (a: NDArray, b: float) = a * (scalar b)
    static member (/) (a: NDArray, b: float) = a / (scalar b)
    static member Pow (a: NDArray, b: float) = a ** (scalar b)

    static member (+) (a: float, b: NDArray) = (scalar a) + b
    static member (-) (a: float, b: NDArray) = (scalar a) - b
    static member (*) (a: float, b: NDArray) = (scalar a) * b
    static member (/) (a: float, b: NDArray) = (scalar a) / b
    static member Pow (a: float, b: NDArray) = (scalar a) ** b

    // transposition
    static member Pow (a: NDArray, b: T) = transpose a 

    // tensor binary
    static member (.*) (a: NDArray, b: NDArray) = dot a b
    static member (%*) (a: NDArray, b: NDArray) = tensorProduct a b

    // unary
    static member (~-) (a: NDArray) = negate a
