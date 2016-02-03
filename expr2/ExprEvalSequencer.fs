module ExprEvalSequencer

open System.Collections.Generic
open Op
open Shape

type StrideT = int list

type StorageSlotT = {Name: string; Address: int; Size: int}

type StorageTargetT = {Slot: StorageSlotT; Offset: int; Stride: StrideT}

type StorageAllocationsT = StorageSlotT list

type StreamT = int

type EventT = int

type ExeOpT =
    | ExecLeaf of StorageTargetT * LeafOpT
    | ExecUnary of StorageTargetT * UnaryOpT * StorageTargetT
    | ExecBinary of StorageTargetT * BinaryOpT * StorageTargetT * StorageTargetT
    | WaitOnEvent of EventT
    | EmitEvent of EventT

type ExecUnitIdT = int
type ExecUnitT = {Id: ExecUnitIdT; 
                  DependsOn: ExecUnitIdT list; 
                  Items: ExeOpT list; 
                  Computes: ExprT option;
                  Target: StorageTargetT option;}

//[<StructuredFormatDisplay("{AsString}")>]
//type ExeSeqItemT = {Stream: StreamT; ExeOp: ExeOpT;}
//    with member this.AsString = sprintf "%A" this.ExeOp



module ExeOp =
    /// returns the target storage of the ExeOp
    let target eop =
        match eop with
        | ExecLeaf(t, _) | ExecUnary(t, _, _) | ExecBinary(t, _, _, _) -> t


module ExeSeq =
    let empty : ExeSeqT = []

    /// finds the ExeSequenceItem that computes expr
    let tryFindExpr expr eseq =
        eseq |> List.tryFind 
            (fun esi -> match esi with
                        | ExeSequenceItem(_, Some e) when expr = e -> true
                        | _ -> false)

type EvalResultT = {ExecUnitId: ExecUnitIdT; Storage: StorageTargetT; Shared: bool}
type EvalReqT = {Expr: ExprT; Storage: StorageTargetT option; Afterwards: EvalResultT -> unit}

type EvalReqsT = EvalReqT list

module EvalReq =
    let reqsCount er =
        List.length er.Requests

module EvalReqs =
    let addReq newExpr newReq (ers: EvalReqsT) =
        let rec addRequestRec ers =
            match ers with
            | ({Expr=rExpr} as req)::reqs when rExpr = newExpr ->
                {req with Requests = newReq::req.Requests} :: reqs
            | req::reqs -> req :: addRequestRec reqs
            | [] -> [{Expr=newExpr; Requests=[newReq]}]
        addRequestRec ers


let subExprEvalCount expr =
    let evalCount = Dictionary<ExprT, int>()
    let rec build expr =
        if evalCount.ContainsKey(expr) then
            evalCount.[expr] <- evalCount.[expr] + 1
        else
            evalCount.[expr] <- 1

        match expr with
        | Leaf _ -> ()
        | Unary(_, a) -> build a
        | Binary(_, a, b) -> build a; build b
    build expr

    fun subExpr ->
        if evalCount.ContainsKey(subExpr) then
            evalCount.[subExpr]
        else 
            0


let canWorkInPlace unaryOp = true

let canWorkInFirstPlace binaryOp = true

let canWorkInSecondPlace binaryOp = true


let buildSequence (sizeSymbolEnv: SymbolEnvT)  (expr: ExprT) =
//    let mutable storageCount = 0
//    let newStorage () : StorageT =
//        storageCount <- storageCount + 1
//        sprintf "s%d" storageCount

    let evalCnt = subExprEvalCount expr

    let mutable execUnits = []
    let mutable maxExecUnitId = 0
    let newExecUnit () =
        maxExecUnitId <- maxExecUnitId + 1
        {Id=maxExecUnitId; DependsOn=[]; Items=[]; Computes=None;}
    let submitExecUnit eu =
        execUnits <- eu :: execUnits

    let newStorageTarget shape =
        // TODO
        {Slot={Name="TODO"}}

    let mutable evalRequests : EvalReqT list = []
    let mutable evaluatedExprs : Map<ExprT, EvalResultT> = Map.empty

    let submitEvalRequest expr storage afterwards =
        evalRequests <- {Expr=expr; Storage=storage; Afterwards=afterwards} :: evalRequests

    let processEvalRequest () =   

        // find a request to process and target storage
        let erqToProcess, erqStorage, erqResult =
            // First, look if there are any expressions which are already computed.
            match evalRequests |> List.tryFind (fun erq -> evaluatedExprs |> Map.containsKey erq.Expr) with
            | Some computedErq -> computedErq, 
                                  Some evaluatedExprs.[computedErq.Expr].Storage, 
                                  Some evaluatedExprs.[computedErq.Expr]
            | None ->
                // if none, look if there is a group of request for the same expression whose requestors are all known.
                let erqsByExpr = evalRequests |> List.groupBy (fun erq -> erq.Expr)
                let _, erqsForExpr = erqsByExpr |> List.find (fun (expr, rs) -> List.length rs = evalCnt expr)

                // If a request from the group has a specified storage target, process it first.
                match List.tryFind (fun erq -> erq.Storage <> None) erqsForExpr with
                | Some erqWithStorage -> 
                    erqWithStorage, erqWithStorage.Storage, None
                | None -> 
                    // Otherwise process any (the first) request from the group.
                    erqsForExpr.[0], None, None
        
        /// stores the evaluation result and executes Afterwards functions of the requestor
        let completeEvalRequest result =
            evaluatedExprs <- evaluatedExprs |> Map.add erqToProcess.Expr result
            erqToProcess.Afterwards result

        match erqResult with
        | Some result ->
            // expr is already evaluated
            erqToProcess.Afterwards result
        | None ->
            // emit exec unit to evaluate expression
            match erqToProcess.Expr with
            | Leaf(op) ->
                // If no desired storage has been specified, we allocate a new one for this leaf.
                let targetStorage = 
                    match erqStorage with
                    | Some s -> s
                    | None -> newStorageTarget (shapeOf erqToProcess.Expr)

                // emit execution unit 
                let eu = {newExecUnit() with Items=[ExecLeaf(targetStorage, op)]}
                submitExecUnit eu

                completeEvalRequest {ExecUnitId=eu.Id; Storage=targetStorage; Shared=false}
            | Unary(op, aExpr) -> 
                // request aExpr to be evaluated directly into our storage, if we can work inplace.
                let aReqStorage = if canWorkInPlace op then erqStorage else None

                submitEvalRequest aExpr aReqStorage 
                    (fun aRes ->
                        // determine our definitive storage
                        let targetStorage =
                            match erqStorage with
                            | Some s -> s
                            | None when canWorkInPlace op && not aRes.Shared -> aRes.Storage
                            | None -> newStorageTarget (shapeOf erqToProcess.Expr)                               
                        let targetShared =
                            if targetStorage = aRes.Storage then aRes.Shared else false

                        // emit execution unit 
                        let eu = {newExecUnit() with Items=[ExecUnary(targetStorage, op, aRes.Storage)];
                                                     DependsOn=[aRes.ExecUnitId]}                                    
                        submitExecUnit eu

                        completeEvalRequest {ExecUnitId=eu.Id; Storage=targetStorage; Shared=targetShared}
                    )                                                                       
            | Binary(op, aExpr, bExpr) ->
                // request aExpr or bExpr to be evaluated directly into our storage, if we can work inplace.
                let aReqStorage, bReqStorage = 
                    match canWorkInFirstPlace op, canWorkInSecondPlace op with
                    | true, _ -> erqStorage, None
                    | _, true -> None, erqStorage
                    | false, false -> None, None

                // callback when aExpr and bExpr requests have been evaluated
                let mutable aRes = None
                let mutable bRes = None
                let maybeCompleted () =
                    match aRes, bRes with
                    | Some aRes, Some bRes ->
                        // determine our definitive storage
                        let targetStorage =
                            match erqStorage with
                            | Some s -> s
                            | None when canWorkInFirstPlace op && not aRes.Shared -> aRes.Storage
                            | None when canWorkInSecondPlace op && not bRes.Shared -> bRes.Storage
                            | None -> newStorageTarget (shapeOf erqToProcess.Expr)
                        let targetShared = 
                            (if targetStorage = aRes.Storage then aRes.Shared else false) ||
                            (if targetStorage = bRes.Storage then bRes.Shared else false)

                        // emit execution unit 
                        let eu = {newExecUnit() with Items=[ExecBinary(targetStorage, op, aRes.Storage, bRes.Storage)];
                                                     DependsOn=[aRes.ExecUnitId; bRes.ExecUnitId]}
                        submitExecUnit eu

                        completeEvalRequest {ExecUnitId=eu.Id; Storage=targetStorage; Shared=targetShared}
                    | _ -> ()    
                    
                submitEvalRequest aExpr aReqStorage (fun res -> aRes <- Some res)
                submitEvalRequest bExpr bReqStorage (fun res -> bRes <- Some res)



           




    ()



