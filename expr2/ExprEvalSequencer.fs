module ExprEvalSequencer

open Util
open Shape
open Op




type StreamT = int
type EventT = {EventObjectId: int; CorrelationId: int; SendingExecUnitId: int}

type StrideT = int list

[<Measure>]
type MemAllocId

type MemAllocT = {Id: int<MemAllocId>; Size: int}
type NDArrayViewT = {Memory: MemAllocT; Shape: int list;
                     Offset: int; Stride: int list;}

module NDArrayView =
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
    let isContiguous (a: NDArrayViewT) =
        a.Stride = contiguousStride a.Shape    

    /// number of dimensions
    let nDim a =
        List.length a.Shape


/// true if views a and b have at least one element in common
let overlapping a b = false // TODO

/// template instantiation specification
type TemplateInstantiationT = {FuncName: string; TmplArgs: string list; 
                               RetType: string; ArgTypes: string list;}

type KernelT = int
type KernelArgsT = obj list

type GridDimT = int * int * int
type BlockDimT = int * int * int
type WorkDimT = int * int * int


type CudaOpT =
    // memory operations
    | MemcpyDtoD of NDArrayViewT * NDArrayViewT
    | MemcpyHtoD of NDArrayViewT * NDArrayViewT
    | MemcpyDtoH of NDArrayViewT * NDArrayViewT
    | Memset of float * NDArrayViewT
    // kernel execution
    | LaunchKernel of TemplateInstantiationT * WorkDimT * KernelArgsT


type EventPlaceHolderT = EventT option ref

type ExeOpT =
    | CudaOp of CudaOpT
    | WaitOnEvent of EventT
    | EmitEvent of EventPlaceHolderT
    | ExecUnitStartInfo of string
    | ExecUnitEndInfo

let combineWith sep items =    
    let rec combine items = 
        match items with
        | [item] -> item
        | item::rest -> item + sep + combine rest
        | [] -> ""
    items |> Seq.toList |> combine

let toStrSeq items =
    Seq.map (sprintf "%d") items

let cudaNDArrayType view =
    let dims = NDArrayView.nDim view
    let shapeStr = if dims = 0 then "" else "<" + (view.Shape |> toStrSeq |> combineWith ",") + ">"
    let strideStr = "<" + ((view.Offset :: view.Stride) |> toStrSeq |> combineWith ",") + ">"
    sprintf "NDArray%dD<Shape%dD%s, Stride%dD%s > " dims dims shapeStr dims strideStr


/// function instantiation state
type FunctionInstantiationCacheT = {mutable Instantiations: (TemplateInstantiationT * string) list;
                                    mutable Code: string} 

/// instantiates a template function with C linkage and returns the C function name
let instantiateTemplateFunction cache (ti: TemplateInstantiationT) =  
    match cache.Instantiations |> List.tryFind (fun (cti, _) -> cti = ti) with
    | Some (_, cName) -> cName
    | None ->
        // generate C function name
        let nPrv = 
            cache.Instantiations 
            |> List.filter (fun (oti, name) -> oti.FuncName = ti.FuncName) 
            |> List.length
        let firstArgStr = 
            match ti.TmplArgs with
            | fa::_ when not (fa.Contains("<") || fa.Contains(">")) -> fa
            | _ -> ""
        let cName = sprintf "%s_%s_%d" ti.FuncName firstArgStr nPrv
        cache.Instantiations <- (ti, cName) :: cache.Instantiations

        // generate template instantiation with C linkage
        let instStr =
            if List.isEmpty ti.TmplArgs then ti.FuncName
            else sprintf "%s<%s>" ti.FuncName (ti.TmplArgs |> combineWith ", ")
        let argDeclStr = ti.ArgTypes |> List.mapi (fun i t -> sprintf "%s p%d" t i)  |> combineWith ", "
        let argCallStr = ti.ArgTypes |> List.mapi (fun i t -> sprintf "p%d" i) |> combineWith ", "
        let retCmd = if ti.RetType.Trim() = "void" then "" else "return"
        let declStr =
            sprintf "extern \"C\" %s %s (%s);" ti.RetType cName argDeclStr
                + sprintf "%s %s (%s) {" ti.RetType cName argDeclStr
                + sprintf "  %s %s (%s);" retCmd instStr argCallStr
                + sprintf "}"
                + sprintf ""
        cache.Code <- cache.Code + declStr

        cName



let newContinguousView memAllocator shape = 
    {Memory=memAllocator (List.fold (*) 1 shape); 
     Shape=shape; Offset=0; 
     Stride=NDArray.contiguousStride shape}


let trgtViewGivenSrc memAllocator trgtShape reqView op srcViews srcShared  =
    // target that shares no elements with any srcView
    let outplaceTrgt =
        match reqView with
        | Some rv when not (List.exists (overlapping rv) srcViews) -> rv, false
        | _ -> newContinguousView memAllocator trgtShape, false        

    // target that reuses a srcView, if it may be overwritten
    let inplaceOverwriteTrgt =
        match List.tryFindIndex not srcShared with
        | Some i -> srcViews.[i], false
        | None -> outplaceTrgt    

    match op with
    // variable access
    | LeafOp (Var vs) ->
        // TODO: use variable memory
        newContinguousView memAllocator trgtShape, false        
    // tensor creation
    | LeafOp _ -> outplaceTrgt        

    // unary elementwise
    | UnaryOp Negate -> inplaceOverwriteTrgt
    | UnaryOp Log -> inplaceOverwriteTrgt
    | UnaryOp Exp -> inplaceOverwriteTrgt
    // reductions
    | UnaryOp Sum -> outplaceTrgt
    | UnaryOp (SumAxis _) -> outplaceTrgt
    // shape operations
    | UnaryOp (Reshape _) ->        
        // TODO: optimize: check if copy is really necessary
        if NDArrayView.isContiguous srcViews.[0] then
            {srcViews.[0] with Shape=trgtShape; Stride=NDArrayView.contiguousStride trgtShape}, srcShared.[0]
        else outplaceTrgt  // will copy
    | UnaryOp (Broadcast _) ->
        let aView, aShared = srcViews.[0], srcShared.[0]
        {aView with Shape=trgtShape; 
                    Stride=List.map3 
                        (fun aStr aShp tShp -> if aShp = tShp then aStr else 0) 
                        aView.Stride aView.Shape trgtShape}, aShared
    | UnaryOp (SwapDim (ax1, ax2)) ->
        let aView, aShared = srcViews.[0], srcShared.[0]
        let str = aView.Stride
        {aView with Shape=trgtShape; 
                    Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1]}, aShared
    // misc
    | UnaryOp (Annotated _) -> srcViews.[0], srcShared.[0]

    // binary elementwise
    | BinaryOp Add -> inplaceOverwriteTrgt
    | BinaryOp Substract -> inplaceOverwriteTrgt
    | BinaryOp Multiply -> inplaceOverwriteTrgt
    | BinaryOp Divide -> inplaceOverwriteTrgt
    | BinaryOp Power -> inplaceOverwriteTrgt
    // matrix/tensor operations
    | BinaryOp Dot -> outplaceTrgt
    | BinaryOp TensorProduct -> outplaceTrgt
      

let execItemsForElemwise trgtView cOp cOpIndexed srcViews =
    let nSrc = List.length srcViews
    let argTypes = cudaNDArrayType trgtView :: (List.map cudaNDArrayType srcViews)
    let argTypesPointers = argTypes |> List.map (fun at -> at + " *")
    let indexedStr = if cOpIndexed then "Indexed" else ""
    let kernel = 
        {FuncName=sprintf "elementwise%dAry%dD%s" nSrc (NDArrayView.nDim trgtView) indexedStr;
         TmplArgs=cOp :: argTypes;
         RetType="void";
         ArgTypes=argTypesPointers}

    let workDim = 
        match NDArrayView.nDim trgtView with
        | 0 -> (1, 1, 1)
        | 1 -> (trgtView.Shape.[0], 1, 1)
        | 2 -> (trgtView.Shape.[0], trgtView.Shape.[1], 1)
        | 3 -> (trgtView.Shape.[0], trgtView.Shape.[1], trgtView.Shape.[2])
        | d ->
            let rest = {2 .. d-1} |> Seq.map (fun i -> trgtView.Shape.[i]) |> Seq.fold (*) 1 
            (trgtView.Shape.[0], trgtView.Shape.[1], rest)

    [LaunchKernel(kernel, workDim, (trgtView.Memory :> obj) :: (List.map (fun v -> v.Memory :> obj) srcViews))]

let execItemsForOp trgtView op srcViews =
    match op with 
    // tensor creation
    | LeafOp (DiagonalOne _) -> execItemsForElemwise trgtView "DiagonalOneIEOp_t" true []
    | LeafOp (Zeros _) -> execItemsForElemwise trgtView "ZerosEOp_t" true []
    | LeafOp (ScalarConst f) -> execItemsForElemwise trgtView (sprintf "ConstEOp_t<%f>" f) true []
    | LeafOp (TensorConst(f, _)) -> execItemsForElemwise trgtView (sprintf "ConstEOp_t<%f>" f) true []
    // variable access
    | LeafOp (Var vs) ->
        // TODO: use variable memory
        
    // unary elementwise
    | UnaryOp Negate -> execItemsForElemwise trgtView "NegateEOp_t" false srcViews
    | UnaryOp Log -> execItemsForElemwise trgtView "LogEOp_t" false srcViews
    | UnaryOp Exp -> execItemsForElemwise trgtView "ExpEOp_t" false srcViews
    // reductions
    | UnaryOp Sum -> failwith "not implemented"
    | UnaryOp (SumAxis _) -> failwith "not implemented"
    // shape operations
    | UnaryOp (Reshape _) ->
        if trgtView <> srcViews.[0] then execItemsForElemwise trgtView "IdEOp_t" false srcViews
        else []
    | UnaryOp (Broadcast _) -> []
    | UnaryOp (SwapDim _) -> []
    // misc
    | UnaryOp (Annotated _) -> []
    
let canWorkInFirstPlace binaryOp = 
    match binaryOp with
    // binary elementwise
    | Add -> true
    | Substract -> true
    | Multiply -> true
    | Divide -> true
    | Power -> true
    // matrix/tensor operations
    | Dot -> false
    | TensorProduct -> false

let canWorkInSecondPlace binaryOp = canWorkInFirstPlace binaryOp



type ExecUnitIdT = int
type ExecUnitT = {Id: ExecUnitIdT; DependsOn: ExecUnitIdT list; Items: ExeOpT list; }

type EvalResultT = {ExecUnitId: ExecUnitIdT; View: NDArrayViewT; Shared: bool}
type EvalReqT = {Id: int; Expr: ExprT; Multiplicity: int; View: NDArrayViewT option; OnCompletion: EvalResultT -> unit}

let exprToExecUnits (sizeSymbolEnv: SymbolEnvT) (expr: ExprT) =
    // number of occurrences of subexpressions
    let exprOccurrences = subExprOccurrences expr

    // calculates the numeric shape
    let numShapeOf expr = shapeOf expr |> ShapeSpec.eval sizeSymbolEnv

    // execution units
    let mutable execUnits = []
    let mutable execUnitIdCnt = 0
    let newExecUnit () =
        execUnitIdCnt <- execUnitIdCnt + 1
        {Id=execUnitIdCnt; DependsOn=[]; Items=[];}
    let submitExecUnit eu =
        execUnits <- eu :: execUnits

    // storage space
    let mutable memAllocIdCnt = 0<MemAllocId>
    let mutable memAllocs = []
    let newMemory size = 
        let mem = {Id = 1<MemAllocId> * (List.length memAllocs); Size=size}
        memAllocs <- mem :: memAllocs
        mem
    let newStorageView shape =       
        {Memory=newMemory (List.fold (*) 1 shape); 
         Shape=shape; Offset=0; 
         Stride=NDArray.contiguousStride shape}

    // evaluation requestion
    let mutable evalRequests : EvalReqT list = []
    let mutable evalRequestIdCnt = 0
    let submitEvalRequest expr multiplicity storage onCompletion =
        evalRequestIdCnt <- evalRequestIdCnt + 1
        evalRequests <- {Id=evalRequestIdCnt; Expr=expr; Multiplicity=multiplicity; 
                         View=storage; OnCompletion=onCompletion} :: evalRequests

    // evaluated requests
    let mutable evaluatedExprs : Map<ExprT, EvalResultT> = Map.empty

    /// takes an evaluation request from the evaluation request queue and processes it
    let processEvalRequest () =   
        // find a request to process and target storage
        let erqToProcess, erqTarget, erqResult, erqMultiplicity, erqRequestors =
            // First, look if there are any expressions which are already computed.
            match evalRequests |> List.tryFind (fun erq -> evaluatedExprs |> Map.containsKey erq.Expr) with
            | Some computedErq -> computedErq, 
                                  Some evaluatedExprs.[computedErq.Expr].View, 
                                  Some evaluatedExprs.[computedErq.Expr],
                                  0, 0
            | None ->
                // if none, look if there is a group of request for the same expression whose requestors are all known.
                let erqsByExpr = evalRequests |> List.groupBy (fun erq -> erq.Expr)                              
                let _, erqsForExpr = erqsByExpr |> List.find (fun (expr, rs) -> 
                    rs |> List.sumBy (fun r -> r.Multiplicity) = exprOccurrences expr)
                let multiplicity = erqsForExpr |> List.sumBy (fun r -> r.Multiplicity)
                let requestors = erqsForExpr |> List.length

                // If a request from the group has a specified storage target, process it first.
                match List.tryFind (fun erq -> erq.View <> None) erqsForExpr with
                | Some erqWithStorage -> 
                    erqWithStorage, erqWithStorage.View, None, multiplicity, requestors
                | None -> 
                    // Otherwise process any (the first) request from the group.
                    erqsForExpr.[0], None, None, multiplicity, requestors
        
        /// true, if result is processed by multiple requestors
        let erqResultShared = erqRequestors > 1

        /// stores the evaluation result and executes Afterwards functions of the requestor
        let completeEvalRequest result =
            evaluatedExprs <- evaluatedExprs |> Map.add erqToProcess.Expr result
            erqToProcess.OnCompletion result

        match erqResult with
        | Some result ->
            // expr is already evaluated
            completeEvalRequest result
        | None ->
            // emit exec unit to evaluate expression
            let erqExpr = erqToProcess.Expr
            match erqExpr with
            | Leaf(op) ->
                // If no desired storage has been specified, we allocate a new one for this leaf.
                let trgtStorage = 
                    match erqTarget with
                    | Some s -> s
                    | None -> newStorageView (numShapeOf erqToProcess.Expr)
                let trgtShared = erqResultShared

                // emit execution unit 
                let eu = {newExecUnit() with Items=[ExecLeaf(trgtStorage, op)]}
                submitExecUnit eu

                completeEvalRequest {ExecUnitId=eu.Id; View=trgtStorage; Shared=trgtShared}
            | Unary(op, aExpr) -> 
                // request aExpr to be evaluated directly into our storage, if we can work inplace.
                let aReqStorage = if inplaceUnary op then erqTarget else None

                submitEvalRequest aExpr erqMultiplicity aReqStorage 
                    (fun aRes ->
                        // determine our definitive storage

                        let trgtStorage, trgtShared =
                            trgtViewGivenSrc newMemory (numShapeOf erqExpr) erqTarget op aRes.View aRes.Shared

                        // emit execution unit 
                        let eu = {newExecUnit() with Items=[ExecUnary(trgtStorage, op, aRes.View)];
                                                     DependsOn=[aRes.ExecUnitId]}                                    
                        submitExecUnit eu

                        completeEvalRequest {ExecUnitId=eu.Id; View=trgtStorage; Shared=trgtShared}
                    )                                                                       
            | Binary(op, aExpr, bExpr) ->
                // request aExpr or bExpr to be evaluated directly into our storage, if we can work inplace.
                let aReqStorage, bReqStorage = 
                    match canWorkInFirstPlace op, canWorkInSecondPlace op with
                    | true, _ -> erqTarget, None
                    | _, true -> None, erqTarget
                    | false, false -> None, None

                // callback when aExpr and bExpr requests have been evaluated
                let mutable aRes = None
                let mutable bRes = None
                let onMaybeCompleted () =
                    match aRes, bRes with
                    | Some aRes, Some bRes ->
                        // determine our definitive storage
                        let trgtStorage =
                            match erqTarget with
                            | Some s -> s
                            | None when canWorkInFirstPlace op && not aRes.Shared -> aRes.View
                            | None when canWorkInSecondPlace op && not bRes.Shared -> bRes.View
                            | None -> newStorageView (numShapeOf erqToProcess.Expr)
                        let trgtShared = 
                            (if trgtStorage = aRes.View then aRes.Shared else false) ||
                            (if trgtStorage = bRes.View then bRes.Shared else false) ||
                            erqResultShared

                        // emit execution unit 
                        let eu = {newExecUnit() with Items=[ExecBinary(trgtStorage, op, aRes.View, bRes.View)];
                                                     DependsOn=[aRes.ExecUnitId; bRes.ExecUnitId]}
                        submitExecUnit eu

                        completeEvalRequest {ExecUnitId=eu.Id; View=trgtStorage; Shared=trgtShared}
                    | _ -> ()    
                    
                submitEvalRequest aExpr erqMultiplicity aReqStorage (fun res -> aRes <- Some res; onMaybeCompleted())
                submitEvalRequest bExpr erqMultiplicity bReqStorage (fun res -> bRes <- Some res; onMaybeCompleted())
        
        // remove eval request        
        evalRequests <- evalRequests |> List.filter (fun erq -> erq.Id <> erqToProcess.Id)

    // create initial evaluation request
    let mutable exprRes = None
    submitEvalRequest expr 1 None (fun res -> exprRes <- Some res)

    // processing loop
    while not (List.isEmpty evalRequests) do
        processEvalRequest ()

    execUnits, exprRes


type MutableList<'a> = System.Collections.Generic.List<'a>
type StreamSeqT = List<ExeOpT>

/// converts execution units to stream commands
let execUnitsToStreamCommands (execUnits: ExecUnitT list) =
    /// event counter
    let mutable eventObjectCnt = 0
    /// correlation counter
    let mutable correlationCnt = 0
    /// all allocated streams
    let streams = new MutableList<StreamSeqT>()
    /// stream used by an ExecUnit        
    let mutable streamOfUnit : Map<ExecUnitIdT, StreamT> = Map.empty
    /// event emitted by an ExecUnit when its execution is finished
    let mutable eventOfUnit : Map<ExecUnitIdT, EventT> = Map.empty
    let mutable eventPlaceHolders : Map<ExecUnitIdT, EventPlaceHolderT> = Map.empty
    /// ExecUnits that still need to be processed
    let mutable execUnitsToProcess = execUnits

    /// create a new event object id
    let newEventObjectId() =
        eventObjectCnt <- eventObjectCnt + 1
        eventObjectCnt

    /// creates a new correlation id
    let newCorrelationId() =
        correlationCnt <- correlationCnt + 1
        correlationCnt

    /// creates a new stream
    let newStream () =
        streams.Add([])
        //if streams.Count - 1 = 2 then failwith "why are you allocating me?"
        streams.Count - 1

    /// length of a stream
    let streamLength (s: int) =
        List.length streams.[s]

    /// emits an ExeOp to the given streams
    let emitToStream s (exeOp: ExeOpT) =
        streams.[s] <- streams.[s] @ [exeOp]

    /// get the ExecUnit with the given id
    let execUnitById (execUnitId: ExecUnitIdT) =
        execUnits |> List.find (fun eu -> eu.Id = execUnitId)

    /// gets the ExecUnits that depend on the specified unit
    let dependants (execUnitId: ExecUnitIdT) =
        execUnits |> List.filter (fun eu -> eu.DependsOn |> List.contains execUnitId)

    /// gets the ExecUnits on which the specified unit depends
    let dependsOn (execUnitId: ExecUnitIdT) =
        (execUnitById execUnitId).DependsOn

    /// true if all ExecUnits that eu depends on have already been processed
    let dependsSatisfied eu =
        eu.DependsOn |> List.forall (fun id -> not (List.exists (fun (eutp: ExecUnitT) -> eutp.Id = id) execUnitsToProcess))

    /// stream used by ExecUnit with Id euId
    let tryGetStreamOfExecUnitId euId =
        if streamOfUnit |> Map.containsKey euId then Some streamOfUnit.[euId]
        else None

    /// all streams that are used by the successors of an ExecUnit
    let rec usedStreamsOfAndBelowExecUnit (eu: ExecUnitT) = 
        eu.Id |> dependants |> List.fold (fun ustrs subEu -> 
            match tryGetStreamOfExecUnitId subEu.Id with
            | Some us -> us::ustrs
            | None -> ustrs) 
            (match tryGetStreamOfExecUnitId eu.Id with
            | Some us -> [us]
            | None -> [])

    /// all streams that can currently be reused safely below an ExecUnit
    let rec availableStreamsBelowExecUnit (eu: ExecUnitT) =
        // streams already avaiable from the ExecUnits we depend on
        let availFromAbove = 
            eu.DependsOn |> List.map (execUnitById >> availableStreamsBelowExecUnit) |> List.concat |> Set.ofList
        // streams that end with the nodes that we depend on
        let endingHere = 
            eu.DependsOn 
            |> List.filter (fun pid -> 
                pid |> dependants |> List.exists (fun dep -> 
                    tryGetStreamOfExecUnitId dep.Id = Some streamOfUnit.[pid]) |> not)
            |> List.map (fun pid -> streamOfUnit.[pid]) 
            |> Set.ofList
        // my stream
        let myStream = Set.singleton streamOfUnit.[eu.Id]
        // streams that are used by nodes that depend on us
        let usedBelow = 
            eu.Id |> dependants |> List.map usedStreamsOfAndBelowExecUnit |> List.concat |> Set.ofList
            
        (availFromAbove + endingHere + myStream) - usedBelow |> Set.toList

    /// returns true if sucId is a successor (thus dependent on) pId
    let rec isSuccessorOf pId sucId =
        if pId = sucId then true
        else pId |> dependants |> List.exists (fun d -> isSuccessorOf d.Id sucId) 

    /// find reuseable event objects for the given ExecUnit
    let tryFindAvailableEventObjectsFor (eu: ExecUnitIdT)  =
        let rec findRec peu =
            match eventOfUnit |> Map.tryFind peu with
            | Some {EventObjectId=eObjId} when 
                    peu |> dependants |> List.forall (fun d -> isSuccessorOf d.Id eu) 
                -> Some eObjId
            | _ -> 
                peu |> dependsOn |> List.tryPick findRec            
        findRec eu

    // generate streams
    while not execUnitsToProcess.IsEmpty do
        // find an execution unit that has all dependencies satisfied
        let eu = execUnitsToProcess |> List.find dependsSatisfied

        /// all streams that are reuseable below the units we depend on
        let availStreams = 
            eu.DependsOn |> List.map (execUnitById >> availableStreamsBelowExecUnit) |> List.concat |> Set.ofList
        /// all streams of the units we depend on
        let streamTakeOverCands = 
            eu.DependsOn |> List.map (fun pId -> streamOfUnit.[pId]) |> Set.ofList
        /// all streams of the units we depend on, that have not been reused by our siblings or their successors
        let streamsNotYetTakenOver = Set.intersect availStreams streamTakeOverCands

        // If we can take over a stream, then take over the one with least commands in it
        // Otherwise, use the first available reusable stream or create a new one.
        let euStream = match streamsNotYetTakenOver |> Set.toList with
                       | _::_ as cs -> cs |> List.minBy streamLength
                       | [] -> match availStreams |> Set.toList with
                               | s::_ -> s
                               | [] -> newStream ()

        // store stream
        streamOfUnit <- streamOfUnit |> Map.add eu.Id euStream
               
        // our stream needs to wait on the results of the streams we depend on
        for endingUnitId in eu.DependsOn |> List.filter (fun pId -> streamOfUnit.[pId] <> euStream) do            
            match eventOfUnit |> Map.tryFind endingUnitId with
            | Some evt ->
                // wait on already emitted event, if possible
                WaitOnEvent evt |> emitToStream euStream
            | None ->
                // if possible, reuse an existing event object
                let evtObjId =
                    match tryFindAvailableEventObjectsFor endingUnitId with
                    | Some id -> id
                    | None -> newEventObjectId()
                let evt = {EventObjectId=evtObjId; CorrelationId=newCorrelationId(); SendingExecUnitId=endingUnitId}
                eventOfUnit <- eventOfUnit |> Map.add endingUnitId evt
                
                // fill in event placeholder and wait for event
                eventPlaceHolders.[endingUnitId] := Some evt
                WaitOnEvent evt |> emitToStream euStream

        // emit our instructions
        //sprintf "%A" eu |> ExecUnitStartInfo |> emitToStream euStream
        for cmd in eu.Items do
            cmd |> emitToStream euStream

        // emit an event placeholder to allow for synchronization
        let evtPh = ref None
        eventPlaceHolders <- eventPlaceHolders |> Map.add eu.Id evtPh
        EmitEvent evtPh |> emitToStream streamOfUnit.[eu.Id]

        // remove from queue
        execUnitsToProcess <- execUnitsToProcess |> List.withoutValue eu

    // remove empty EmitEvent placeholders
    streams 
        |> Seq.map (fun stream -> 
            stream |> List.filter (fun op -> 
                match op with
                | EmitEvent re when !re = None -> false
                | _ -> true))
        |> Seq.toList
       

// TODO: variable memory allocation
// TODO: CUDA driver call sequencing



let pitchAlignment = 256 // TODO: get from device


type CudaFlagsT = int

type EventObjectT = int


type CudaCallT =
    // memory mangement
    | MemAlloc of int<bytes> * NDArrayViewT
    | MemFree of NDArrayViewT
    // memory operations
    | MemcpyAsync of NDArrayViewT * NDArrayViewT * int<bytes> * StreamT
    | MemcpyHtoDAsync of NDArrayViewT * NDArrayViewT * int<bytes> * StreamT
    | MemcpyDtoHAsync of NDArrayViewT * NDArrayViewT * int<bytes> * StreamT
    | MemsetD32Async of uint32 * int<bytes> * StreamT
    // stream management
    | StreamCreate of StreamT * CudaFlagsT
    | StreamDestory of StreamT
    | StreamWaitEvent of StreamT * EventObjectT * CudaFlagsT
    // event mangement
    | EventCreate of EventObjectT * CudaFlagsT
    | EventDestory of EventObjectT
    | EventRecord of EventObjectT * StreamT
    | EventSynchronize of EventObjectT
    // execution control
    | LaunchKernel of KernelT * GridDimT * BlockDimT * int<bytes> * StreamT * KernelArgsT
    // dummy op
    | ExeOp of StreamT * ExeOpT


//let genNDArray (storage: St =

//let instantiateKernelForElemwiseOp trgt op a =
    


/// generates a sequence of CUDA calls from streams
let generateCalls streams =
    
    /// the number of times WaitOnEvent is called for a particular correlation
    let correlationIdWaiters =
        seq {
            for strm in streams do
                for exec in strm do
                    match exec with
                    | WaitOnEvent evt -> yield evt.CorrelationId
                    | _ -> ()
        } |> Seq.countBy id |> Map.ofSeq
        
    let rec generate streamCallHistory activeEvents streams =
        if List.exists ((<>) []) streams then
            // sort streams by call history
            let streamsSorted = 
                streams
                |> List.indexed
                |> List.sortByDescending (fun (i, strm) ->                         
                    let callsBetween = 
                        match streamCallHistory |> List.tryFindIndex ((=) i) with
                        | Some ord -> ord
                        | None -> 9999
                    let syncPenalty = 
                        match strm with
                        | EmitEvent _::_ -> 1000
                        | WaitOnEvent _::_ -> -1000
                        | _ -> 0
                    callsBetween + syncPenalty) 
        
            // find stream to process
            let strmIdToProcess, strmToProcess = 
                streamsSorted 
                |> List.find (fun (strmId, strm) ->
                    match strm with
                    | WaitOnEvent evt ::_ when 
                        activeEvents |> List.exists (fun e -> e.CorrelationId = evt.CorrelationId) -> true
                        // WaitOnEvent can only be called when EmitEvent 
                        // with same CorrelationId has been called before.
                    | WaitOnEvent _ ::_ -> false
                    | EmitEvent evtp ::_ ->
                        match !evtp with
                        | Some evt when
                            activeEvents |> List.exists (fun e -> e.EventObjectId = evt.EventObjectId) -> false
                            // EmitEvent for a given event must be called
                            // after all necessary calls to WaitOnEvent for a previous correlation.
                        | _ -> true
                    | [] -> false
                    | _ -> true)

            // book keeping
            let execOp = List.head strmToProcess       
            let remainingStreams = streams |> List.map (fun strm -> 
                if strm = strmToProcess then List.tail strm
                else strm)

            match execOp with
            | WaitOnEvent evt ->
                // remove active event
                let activeEvents = activeEvents |> List.removeValueOnce evt

                let cmd = StreamWaitEvent (strmIdToProcess, evt.EventObjectId, 0)
                cmd :: generate streamCallHistory activeEvents remainingStreams
            | EmitEvent evtp ->
                // add active event as many times as it will be waited upon
                let evt = Option.get !evtp
                let activeEvents = List.replicate correlationIdWaiters.[evt.CorrelationId] evt @ activeEvents

                let cmd = EventRecord (evt.EventObjectId, strmIdToProcess)
                cmd :: generate streamCallHistory activeEvents remainingStreams
//            | ExecLeaf (trgt, op) ->
//                // TODO
//                generate streamCallHistory activeEvents remainingStreams
            | ExecUnary (trgt, op, a) ->
                match op with
                | Negate ->
                    
                generate streamCallHistory activeEvents remainingStreams
//            | ExecBinary (trgt, op, a, b) ->
//                // TODO
//                generate streamCallHistory activeEvents remainingStreams
            | _ as eop -> 
                let streamCallHistory = strmIdToProcess :: streamCallHistory
                ExeOp(strmIdToProcess, eop) :: generate streamCallHistory activeEvents remainingStreams
        else
            // streams are all empty
            []

    generate [] [] streams
