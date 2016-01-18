module NDArray

module List =
    let rec set lst elem value =
        match lst, elem with
            | l::ls, 0 -> value::ls
            | l::ls, _ -> l::(set ls (elem-1) value)
            | [], _ -> invalidArg "elem" "element index out of bounds"

    let without elem lst =
        List.concat [List.take elem lst; List.skip (elem+1) lst] 


type ndarray = {shape: int list;
                stride: int list; 
                offset: int;
                data: float[]}

////////////////////////////////////////////////////////////////////////////////////////////////
// shape functions
////////////////////////////////////////////////////////////////////////////////////////////////
            
let ndim a = List.length a.shape
    
let shape a = a.shape

let padLeft a =
    {a with shape=1::a.shape; stride=0::a.stride}

let padRight a =
    {a with shape=List.append a.shape [1]; stride=List.append a.stride [0]}

let broadcast a dim size =
    match (shape a).[dim] with
        | 1 -> {a with shape=List.set a.shape dim size; stride=List.set a.stride dim 0}
        | _ -> failwithf "dimension %d must be of size 1 to broadcast (is %A)" dim (shape a)

let broadcastToSame ain bin =
    let mutable a = ain
    let mutable b = bin 
    while ndim a < ndim b do
        a <- padLeft a
    while ndim b < ndim a do
        b <- padLeft b
    for d = 0 to (ndim a) - 1 do
        match (shape a).[d], (shape b).[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcast a d bl
            | al, bl when bl = 1 -> b <- broadcast b d al
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size" (shape ain) (shape bin)
    a, b

let checkSameShape a b =
    if a.shape <> b.shape then
        failwithf "inequal shapes:  %A <> %A" a.shape b.shape
         
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
    allIdxRec a.shape   

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
            match a.shape, a.stride with
                | shp::rShps, str::rStrs ->
                    let ra = view {a with shape=rShps; stride=rStrs} ridxs
                    match idx with 
                        | All ->
                            {ra with shape = shp::ra.shape;
                                     stride = str::ra.stride}
                        | Elem i -> 
                            checkElementRange shp i
                            {ra with offset = ra.offset + i*str;
                                     stride = ra.stride;
                                     shape = ra.shape} 
                        | Rng(start, stop) ->
                            checkElementRange shp start
                            checkElementRange shp stop
                            {ra with offset = ra.offset + start*str;
                                     shape = (stop - start)::ra.shape;
                                     stride = str::ra.stride} 
                        | NewAxis -> failwith "impossible"
                | _ -> failwith "incompatible shapes"
        | NewAxis::ridxs ->
                let ra = view a ridxs
                {ra with shape = 1::ra.shape; stride = 0::ra.stride}
        | [] -> 
            match a.shape with
                | shp::rShps -> failwith "incompatible shapes"
                | [] -> a


////////////////////////////////////////////////////////////////////////////////////////////////
// element access
////////////////////////////////////////////////////////////////////////////////////////////////

let addr a idx =
    Seq.map2 (*) idx a.stride |> Seq.sum
    
let get a idx =
    a.data.[addr a idx]
    
let set a idx value =
    a.data.[addr a idx] <- value

let allElems a =
    allIdx a |> Seq.map (get a)

////////////////////////////////////////////////////////////////////////////////////////////////
// array creation functions
////////////////////////////////////////////////////////////////////////////////////////////////

let scalar s =
    {shape=[]; stride=[]; offset=0; data=Array.create 1 s}

let scalarBroadcastedTo a s =
    let _, ss = scalar s |> broadcastToSame a 
    ss

let zeros shape =
    {shape=shape; stride=compactStride shape; offset=0; data=Array.zeroCreate (lengthOfShape shape)}
    
let zerosLike a =
    zeros (shape a)

let ones shape =
    {shape=shape; stride=compactStride shape; offset=0; data=Array.create (lengthOfShape shape) 1.0}

let onesLike a =
    ones (shape a)
    
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
    match a.shape with
        | [] -> Seq.empty
        | _ -> allIdxRec a.shape dim  
       
let axisReduce (f: ndarray -> ndarray) dim a =
    let c = zeros (List.without dim a.shape)
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
    match ndim a, ndim b with
        | 0, _ | _, 0  -> multiply a b
        | 1, 1 -> multiply a b |> product
        | 2, 1 -> view (dot a (padRight b)) [All; Elem 0]
        | 1, 2 -> dot (padRight a) b
        | 2, 2 when a.shape.[1] = b.shape.[0] ->
            let I = a.shape.[0]
            let J = a.shape.[1]
            let K = b.shape.[1]
            let c = zeros [I; K]
            for k=0 to K - 1 do
                for i=0 to I - 1 do
                    {0 .. J - 1}
                        |> Seq.map (fun j -> (get a [i; j]) * (get b [j; k]))
                        |> Seq.sum
                        |> set c [i; k]
            c
        | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" 
                    a.shape b.shape

    
      

