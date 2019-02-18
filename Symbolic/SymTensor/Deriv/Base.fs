namespace SymTensor

open System.Diagnostics
open DeepNet.Utils
open SymTensor.Ops


[<AutoOpen>]
module DerivTypes =

    /// Jacobians for each variable
    type DerivT = {
        /// the number of elements of the function the derivative is taken of
        FunElems:   SizeSpec
        /// the Jacobians w.r.t. the variables occuring in the expression
        Jacobians:  Map<Var, BaseExpr>
    }


    type XChDeriv =
        | SingleChDeriv of BaseExpr
        | MultiChDeriv of Map<string, BaseExpr>


    //type IncomingDeriv =
    //    | FullDeriv of expr:BaseExpr
    //    | ChDeriv of channel:string * expr:BaseMultiChannelExpr


/// Derivative computation functions.
module Deriv = 

    let private add (x: BaseExpr) (y: BaseExpr) =
        (Expr x) + (Expr y) |> Expr.baseExpr

    /// merges two derivative maps
    let private merge (aGrads: DerivT) (bGrads: DerivT) : DerivT =
        if aGrads.FunElems <> bGrads.FunElems then
            failwithf "Cannot merge derivatives with different number of function elements: %A and %A."
                aGrads.FunElems bGrads.FunElems
        let jacs =
            (aGrads.Jacobians, bGrads.Jacobians)
            ||> Map.fold (fun m v vg -> 
                match Map.tryFind v m with
                | Some ovg -> m |> Map.add v (add vg ovg)
                | None -> m |> Map.add v vg) 
        {FunElems=aGrads.FunElems; Jacobians=jacs}

    /// empty derivatives for expression
    let private empty (expr: BaseExpr) =
        {FunElems=expr.NElems; Jacobians=Map.empty}

    /// Computes the derivatives of all arguments of an expression given the derivative of the expression.
    let rec private reverseDiffStep (expr: BaseExpr) (eg: BaseExpr) : Map<string, BaseExpr> =    
        // TODO: get single-channel op derivative
        failwith "TODO"
        
    /// Computes the derivatives of all arguments of a multi-channel op given the derivatives
    /// w.r.t. all channels of the multi-channel op.
    and private multiChannelDiffStep (mcOp: BaseMultiChannelExpr) (eg: Map<Channel, BaseExpr>) : Map<string, XChDeriv> =
        // TODO: get multi-channel op derivative
        failwith "TODO"

    /// Computes the derivatives of the specified expression w.r.t. all variables occuring in it.
    and computeWithRootJacobian (rootJacobian: XChDeriv) (rootExpr: BaseXChExpr) : DerivT =

        // build expression info 
        let exprInfo = BaseXChExprGroup [rootExpr]
        //let rootExpr = List.exactlyOne exprInfo.Exprs

        /// map from an expression to the sum of incoming Jacobians (for all its channels)
        let incomingJacobian = Dictionary<BaseXChExpr, XChDeriv> ()
        /// map from an expression (channel) to the set of dependants that transmitted Jacobian to the expression
        let receivedJacobiansFrom = Dictionary<BaseExprCh, HashSet<BaseXChExpr>> ()
        /// channels of multi-channel expression that have received Jacobian from all their dependants
        let exprChannelsWithFullJacobian = Dictionary<BaseMultiChannelExpr, HashSet<string>> ()
        /// expressions that have received Jacobians from all their dependants (for all channels)
        let exprsWithFullJacobian = Queue<BaseXChExpr> ()

        //let multiChannelOpJacobians = 
        //    Dictionary<BaseMultiChannelExpr, Dictionary<string, Expr>> (HashIdentity.Structural) 
        //let multiChannelOpsWithFullJacobians = Queue<BaseMultiChannelExpr> ()

        /// adds the specified Jacobian coming from `source` to `target`
        let transmitJacobian (source: BaseXChExpr) (target: BaseExprCh) (jacobian: BaseExpr) =
            let neededSources = exprInfo.Dependants target |> Set.ofSeq

            // add jacobian
            let targetExpr = BaseExprCh.asBaseXChExpr target
            incomingJacobian.[targetExpr] <- 
                match incomingJacobian.TryFind targetExpr, target with
                | None, BaseExprCh.Only _ -> 
                    SingleChDeriv jacobian
                | Some (SingleChDeriv j), BaseExprCh.Only _ -> 
                    SingleChDeriv (add j jacobian)
                | None, BaseExprCh.Ch (ch, _) -> 
                    MultiChDeriv (Map [ch, jacobian])
                | Some (MultiChDeriv js), BaseExprCh.Ch (ch, _) ->
                    match js |> Map.tryFind ch with
                    | Some j -> MultiChDeriv (js |> Map.add ch (add j jacobian))
                    | None -> MultiChDeriv (js |> Map.add ch jacobian)
                | _, _ -> failwith "Invalid single-/multi-channel combination for derivative."

            // add to received set
            if not (receivedJacobiansFrom.ContainsKey target) then
                receivedJacobiansFrom.[target] <- HashSet<_> (HashIdentity.Structural)
            if receivedJacobiansFrom.[target].Contains source then
                failwithf "Derivative from %A to %A was transmitted more than once." source target
            if not (neededSources.Contains source) then
                failwithf "%A received derivative from non-dependant %A." target source
            receivedJacobiansFrom.[target].Add source |> ignore

            // check if target has received all derivatives
            let receivedSources = receivedJacobiansFrom.[target] |> Set.ofSeq
            if receivedSources = neededSources then 
                match target with
                | BaseExprCh.Only target -> 
                    exprsWithFullJacobian.Enqueue (BaseXChExpr.SingleCh target)
                | BaseExprCh.Ch (ch, target) ->
                    if not (exprChannelsWithFullJacobian.ContainsKey target) then
                        exprChannelsWithFullJacobian.[target] <- HashSet<_> ()
                    exprChannelsWithFullJacobian.[target].Add ch |> ignore
                    if Set.ofSeq exprChannelsWithFullJacobian.[target] = Set.ofSeq target.Channels then
                        exprsWithFullJacobian.Enqueue (BaseXChExpr.MultiCh target)

        let transmitJacobians src jacobians =
            jacobians
            |> List.groupBy (fun (target, _) -> target)
            |> List.iter (fun (target, jacs) ->
                let jacSum = jacs |> List.map (fun (_, jac) -> jac) |> List.reduce (+)
                transmitJacobian src target jacSum)            

        let transmitMultiChannelOpJacobian mcOp channel jacobian =
            // add jacobian
            if not (multiChannelOpJacobians.ContainsKey mcOp) then
                multiChannelOpJacobians.[mcOp] <- Dictionary<Channel, Expr> (HashIdentity.Structural)
            let mcoj = multiChannelOpJacobians.[mcOp]
            mcoj.[channel] <- jacobian

            // check if multi-channel op has received Jacobians on all its channels
            let received = Set.ofSeq mcoj.Keys
            let needed = exprInfo.UsedChannels mcOp
            if received = needed then multiChannelOpsWithFullJacobians.Enqueue mcOp

        // set Jacobian of root node
        incomingJacobian.[rootExpr] <- rootJacobian
        exprsWithFullJacobian.Enqueue rootExpr

        // process Jacobians in loop
        let mutable varJacs = Map.empty
        while exprsWithFullJacobian.Count > 0 do

            let expr = exprsWithFullJacobian.Dequeue ()

            // propagate Jacobians
            match expr, incomingJacobian.[expr] with
            | BaseXChExpr.SingleCh sChExpr, SingleChDeriv exprDeriv ->
                let targets = sChExpr.Args
                let argDerivs = reverseDiffStep sChExpr exprDeriv
                for KeyValue(argName, argDeriv) in argDerivs do
                    transmitJacobian expr targets.[argName] argDeriv
                ()
                // |> transmitJacobians (Choice1Of2 expr)
            //| Nary (Channel (op, channel), es) ->
            //    transmitMultiChannelOpJacobian (op, es) channel incomingJacobian.[expr]
            //| _ ->
            //    reverseDiffStep expr incomingJacobian.[expr] |> transmitJacobians (Choice1Of2 expr)

            // extract variable Jacobians
            match expr with
            | Leaf (Var vs) -> varJacs <- varJacs |> Map.add vs incomingJacobian.[expr]
            | _ -> ()

            if multiChannelOpsWithFullJacobians.Count > 0 then
                let mcOp = multiChannelOpsWithFullJacobians.Dequeue ()
                let channelJacs = multiChannelOpJacobians.[mcOp] |> Map.ofDictionary               
                multiChannelDiffStep mcOp channelJacs |> transmitJacobians (Choice2Of2 mcOp)
        
        {
            FunElems  = rootJacobian.Shape.[0]
            Jacobians = varJacs
        }    

    /// computes the derivatives of the specified expression w.r.t. all variables occuring in it
    and compute (rootExpr: Expr) : DerivT =
        if Debug.TraceCompile then printfn "Computing derivatives..."
        let sw = Stopwatch.StartNew()
        let rootJac = Expr.shapeOf rootExpr |> ShapeSpec.nElem |> Expr.identityOfSameType rootExpr
        let deriv = computeWithRootJacobian rootJac rootExpr
        if Debug.Timing then printfn "Computing derivatives took %A" sw.Elapsed
        deriv

    /// extracts the Jacobian of the given VarSpecT
    and ofVarSpec var (deriv: DerivT) =
        match deriv.Jacobians |> Map.tryFind var with
        | Some d -> d
        | None when Debug.FailIfVarNotInDerivative -> 
            failwithf "the variable %A is not present in the expression" var
        | None -> 
            let varExpr = Expr.makeVar var
            Expr.zerosOfSameType varExpr [deriv.FunElems; Expr.nElems varExpr]

    /// extracts the Jacobian of the given variable
    and ofVar var deriv =
        ofVarSpec (Expr.extractVar var) deriv                  


