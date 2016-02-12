module ExecUnitsGen

open Util
open Shape
open Op

/// a memory allocation execlusively for this expression (used for temporary results)
type MemAllocT = {Id: int; Size: int}

/// memory can either be internal to this expression or external (passed in variable at runtime)
type MemoryT =
    | MemAlloc of MemAllocT
    | ExternalMem of VarSpecT

/// an n-dimensional array view
type NDArrayViewT = {Memory: MemoryT; Shape: int list;
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

    /// true if the NDArrayView is continguous
    let isContiguous (a: NDArrayViewT) =
        a.Stride = contiguousStride a.Shape    

    /// creates a new memory allocation and a new NDArrayView with continguous layout
    let newContinguous memAllocator shape = 
        {Memory=memAllocator (4 * List.fold (*) 1 shape); 
         Shape=shape; Offset=0; 
         Stride=NDArray.contiguousStride shape}

    /// number of dimensions
    let nDim a =
        List.length a.Shape

    /// number of elements
    let nElems a =
        List.fold (*) 1 a.Shape

    /// transposes the view of a matrix
    let transpose a =
        if nDim a <> 2 then failwith "need a matrix to transpose"
        {a with Shape=List.rev a.Shape; Stride=List.rev a.Stride}

    /// true if NDArrayView can be target of a BLAS operation
    let isBlasTargetable a =
        if nDim a = 2 then a.Stride = NDArray.columnMajorStride a.Shape
        else false

    /// creates a new memory allocation and a new NDArrayView with 
    /// memory layout suitable for being a BLAS target
    let newBlasTarget memAllocator shape =
        {newContinguous memAllocator shape with
            Stride=NDArray.columnMajorStride shape}

    /// true if views a and b have at least one element in common
    let overlapping a b = false // TODO
    

/// Id of an ExecUnitT
type ExecUnitIdT = int

/// a group of commands that must be executed sequentially
type ExecUnitT<'e> = {Id: ExecUnitIdT; 
                      DependsOn: ExecUnitIdT list; 
                      Items: 'e list; }

/// result of an evaluation request
type EvalResultT = {ExecUnitId: ExecUnitIdT; 
                    View: NDArrayViewT; 
                    Shared: bool}

/// an evaluation request
type EvalReqT = {Id: int; 
                 Expr: UExprT; 
                 Multiplicity: int; 
                 View: NDArrayViewT option; 
                 OnCompletion: EvalResultT -> unit}

/// generator function record
type ExecUnitsGeneratorT<'e> = {ExecItemsForOp : (int -> MemoryT) -> NDArrayViewT -> AnyOpT -> NDArrayViewT list -> 'e list;
                                TrgtViewGivenSrc: (int -> MemoryT) -> NShapeSpecT -> NDArrayViewT option -> AnyOpT -> NDArrayViewT list -> bool list -> NDArrayViewT * bool;
                                SrcViewReqsGivenTrgt: NShapeSpecT -> NDArrayViewT option -> AnyOpT -> NShapeSpecT list -> NDArrayViewT option list;}

/// generates execution units that will evaluate the given unified expression
let exprToExecUnits gen (sizeSymbolEnv: SymbolEnvT) (expr: UExprT) =
    // number of occurrences of subexpressions
    let exprOccurrences = subExprOccurrences expr

    // calculates the numeric shape
    let numShapeOf expr = shapeOfUExpr expr |> ShapeSpec.eval sizeSymbolEnv

    // execution units
    let mutable execUnits = []
    let mutable execUnitIdCnt = 0
    let newExecUnit () =
        execUnitIdCnt <- execUnitIdCnt + 1
        {Id=execUnitIdCnt; DependsOn=[]; Items=[];}
    let submitExecUnit eu =
        execUnits <- eu :: execUnits

    // storage space
    let mutable memAllocIdCnt = 0
    let mutable memAllocs = []
    let newMemory size = 
        let mem = {Id = (List.length memAllocs); Size=size}
        memAllocs <- mem :: memAllocs
        MemAlloc mem

    // evaluation requestion
    let mutable evalRequests : EvalReqT list = []
    let mutable evalRequestIdCnt = 0
    let submitEvalRequest expr multiplicity storage onCompletion =
        evalRequestIdCnt <- evalRequestIdCnt + 1
        evalRequests <- {Id=evalRequestIdCnt; Expr=expr; Multiplicity=multiplicity; 
                         View=storage; OnCompletion=onCompletion} :: evalRequests

    // evaluated requests
    let mutable evaluatedExprs : Map<UExprT, EvalResultT> = Map.empty

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
            | UExpr(op, srcs) ->
                let nSrc = List.length srcs
                let mutable subreqResults : Map<UExprT, EvalResultT option> = Map.empty

                let onMaybeCompleted () =
                    if List.forall (fun s -> Map.containsKey s subreqResults) srcs then                       
                        let subres = Map.map (fun k v -> Option.get v) subreqResults

                        // determine our definitive target storage
                        let srcViews, srcShared, srcExeUnitIds = 
                            srcs 
                            |> List.map (fun s -> subres.[s].View, subres.[s].Shared, subres.[s].ExecUnitId) 
                            |> List.unzip3
                        let trgtView, trgtShared =
                            gen.TrgtViewGivenSrc newMemory (numShapeOf erqExpr) erqTarget op srcViews srcShared
                       
                        // emit execution unit 
                        let eu = {newExecUnit() with Items=gen.ExecItemsForOp newMemory trgtView op srcViews;
                                                     DependsOn=srcExeUnitIds}                                    
                        submitExecUnit eu

                        completeEvalRequest {ExecUnitId=eu.Id; View=trgtView; Shared=trgtShared}

                if List.isEmpty srcs then onMaybeCompleted ()
                else
                    let srcReqStorages = 
                        gen.SrcViewReqsGivenTrgt (numShapeOf erqExpr) erqTarget op (List.map numShapeOf srcs)                    
                    for src, srcReqStorage in List.zip srcs srcReqStorages do
                        submitEvalRequest src erqMultiplicity srcReqStorage (fun res ->
                            subreqResults <- subreqResults |> Map.add src (Some res)
                            onMaybeCompleted())     

        // remove eval request        
        evalRequests <- evalRequests |> List.filter (fun erq -> erq.Id <> erqToProcess.Id)

    // create initial evaluation request
    let mutable exprRes = None
    submitEvalRequest expr 1 None (fun res -> exprRes <- Some res)

    // processing loop
    while not (List.isEmpty evalRequests) do
        processEvalRequest ()

    execUnits, exprRes, memAllocs

