module NDArray
open Util


type NDArray = 
    {Shape: int list;
     Stride: int list; 
     Offset: int;
     Data: float[]}


////////////////////////////////////////////////////////////////////////////////////////////////
// shape functions
////////////////////////////////////////////////////////////////////////////////////////////////
            
let nDim a = List.length a.Shape
    
let shape a = a.Shape

let padLeft a =
    {a with Shape=1::a.Shape; Stride=0::a.Stride}

let padRight a =
    {a with Shape=a.Shape @ [1]; Stride=a.Stride @ [0]}

let broadcastDim a dim size =
    match (shape a).[dim] with
        | 1 -> {a with Shape=List.set dim size a.Shape; Stride=List.set dim 0 a.Stride}
        | _ -> failwithf "dimension %d must be of size 1 to broadcast (is %A)" dim (shape a)

let broadcastToSame ain bin =
    let mutable a = ain
    let mutable b = bin 
    while nDim a < nDim b do
        a <- padLeft a
    while nDim b < nDim a do
        b <- padLeft b
    for d = 0 to (nDim a) - 1 do
        match (shape a).[d], (shape b).[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim a d bl
            | al, bl when bl = 1 -> b <- broadcastDim b d al
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size" (shape ain) (shape bin)
    a, b

let broadcastToShape bs ain =
    let bsDim = List.length bs
    let mutable a = ain
    while nDim a < bsDim do
        a <- padLeft a
    if bsDim < nDim a then
        failwithf "shape %A has less dimensions than NDArray with shape %A" bs (shape ain)
    for d = 0 to bsDim - 1 do
        match (shape a).[d], bs.[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim a d bl
            | _ -> failwithf "cannot broadcast NDArray with shape %A to shape %A" (shape ain) bs
    a

let checkSameShape a b =
    if a.Shape <> b.Shape then
        failwithf "inequal shapes:  %A <> %A" a.Shape b.Shape
         
let rec compactStride (shape: int list) =
    match shape with
        | [] -> []
        | [l] -> [1]
        | l::(lp::lrest) ->
            match compactStride (lp::lrest) with 
                | sp::srest -> (lp*sp)::sp::srest
                | [] -> failwith "unexpected"    
     
let rec lengthOfShape shape =
    Seq.fold (*) 1 shape

let allIdx a =
    let rec allIdxRec shape = seq {
        match shape with
            | l::ls ->
                for i=0 to l - 1 do
                    for is in allIdxRec ls do
                        yield i::is
            | [] -> yield []
    } 
    allIdxRec a.Shape   

let idxOfDim a dim =
    { 0 .. (shape a).[dim] - 1}

////////////////////////////////////////////////////////////////////////////////////////////////
// subtensor
////////////////////////////////////////////////////////////////////////////////////////////////

let checkElementRange nElems i =
    if not (0 <= i && i < nElems) then
        failwithf "element index %d out of range (%d elements)" i nElems

//let checkDimAvail shp =
//    if List.isEmpty shp then
//        failwith "incompatible number of dimensions"

type IdxSpec = 
    | Elem of int
    | Rng of int * int
    | NewAxis
    | All

let rec view a idxs =
    match idxs with
        | (All | Elem _ | Rng(_, _) as idx)::ridxs ->
            match a.Shape, a.Stride with
                | shp::rShps, str::rStrs ->
                    let ra = view {a with Shape=rShps; Stride=rStrs} ridxs
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
                | _ -> failwith "incompatible shapes"
        | NewAxis::ridxs ->
                let ra = view a ridxs
                {ra with Shape = 1::ra.Shape; Stride = 0::ra.Stride}
        | [] -> 
            match a.Shape with
                | shp::rShps -> failwith "incompatible shapes"
                | [] -> a


////////////////////////////////////////////////////////////////////////////////////////////////
// element access
////////////////////////////////////////////////////////////////////////////////////////////////

let addr a idx =
    Seq.map2 (*) idx a.Stride |> Seq.sum
    
let get a idx =
    a.Data.[addr a idx]
    
let set a idx value =
    a.Data.[addr a idx] <- value

let allElems a =
    allIdx a |> Seq.map (get a)

////////////////////////////////////////////////////////////////////////////////////////////////
// array creation functions
////////////////////////////////////////////////////////////////////////////////////////////////

let scalar s =
    {Shape=[]; Stride=[]; Offset=0; Data=Array.create 1 s}

let zeros shape =
    {Shape=shape; Stride=compactStride shape; Offset=0; Data=Array.zeroCreate (lengthOfShape shape)}
    
let zerosLike a =
    zeros (shape a)

let ones shape =
    {Shape=shape; Stride=compactStride shape; Offset=0; Data=Array.create (lengthOfShape shape) 1.0}

let onesLike a =
    ones (shape a)

let identity shape =
    let m = zeros shape
    let md = List.length shape
    let dl = List.min shape
    for i = 0 to dl - 1 do
        set m (List.replicate md i) 1.
    m

////////////////////////////////////////////////////////////////////////////////////////////////
// elementwise operations
////////////////////////////////////////////////////////////////////////////////////////////////   
      
let elemwise (f: float -> float) a =
    let c = zerosLike a
    for idx in allIdx a do
        set c idx (f (get a idx))
    c
            
let elemwise2 (f: float -> float -> float) a b =
    let a, b = broadcastToSame a b
    let c = zerosLike a
    for idx in allIdx a do
        set c idx (f (get a idx) (get b idx))
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

let allIdxForReduction dim a =
    let rec allIdxRec shape dim = seq {
        match shape with
            | l::ls ->
                let rest = allIdxRec ls (dim-1)
                if dim = 0 then
                    for is, ws in rest do
                        yield All::is, ws
                else
                    for i=0 to l - 1 do
                        for is, ws in rest do
                            yield Elem i::is, i::ws
            | [] -> yield [], []
    } 
    match a.Shape with
        | [] -> Seq.empty
        | _ -> allIdxRec a.Shape dim  
       
let axisReduce (f: NDArray -> NDArray) dim a =
    let c = zeros (List.without dim a.Shape)
    for vidx, widx in allIdxForReduction dim a do
        get (f (view a vidx)) [] |> set c widx 
    c

let sum a =
    allElems a |> Seq.sum |> scalar
let sumAxis = axisReduce sum 
    
let product a =
    allElems a |> Seq.reduce (*) |> scalar
let productAxis = axisReduce product 


////////////////////////////////////////////////////////////////////////////////////////////////
// matrix operations
////////////////////////////////////////////////////////////////////////////////////////////////         

let rec dot a b =
    match nDim a, nDim b with
        | 0, _ | _, 0  -> multiply a b
        | 1, 1 -> multiply a b |> sum
        | 2, 1 -> view (dot a (padRight b)) [All; Elem 0]
        //| 1, 2 -> dot (padRight a) b
        | 2, 2 when a.Shape.[1] = b.Shape.[0] ->
            let I = a.Shape.[0]
            let J = a.Shape.[1]
            let K = b.Shape.[1]
            let c = zeros [I; K]
            for k=0 to K - 1 do
                for i=0 to I - 1 do
                    {0 .. J - 1}
                        |> Seq.map (fun j -> (get a [i; j]) * (get b [j; k]))
                        |> Seq.sum
                        |> set c [i; k]
            c
        | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" 
                    a.Shape b.Shape

    
type NDArray with

    // elementwise binary
    static member (+) (a: NDArray, b: NDArray) = add a b
    static member (-) (a: NDArray, b: NDArray) = substract a b    
    static member (*) (a: NDArray, b: NDArray) = multiply a b
    static member (/) (a: NDArray, b: NDArray) = divide a b
    static member Pow (a: NDArray, b: NDArray) = power a b


    // unary
    static member (~-) (a: NDArray) = negate a
