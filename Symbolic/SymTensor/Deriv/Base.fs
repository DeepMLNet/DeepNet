namespace SymTensor

open System.Diagnostics
open DeepNet.Utils
open SymTensor.Ops


/// Jacobians for each variable
type Deriv = {
    /// the number of elements of the function the derivative is taken of
    FunElems:   SizeSpec
    /// the Jacobians w.r.t. the variables occuring in the expression
    Jacobians:  Map<Var, Expr>
}


    //type XChDeriv =
    //    | SingleChDeriv of BaseExpr
    //    | MultiChDeriv of Map<string, BaseExpr>


    //type IncomingDeriv =
    //    | FullDeriv of expr:BaseExpr
    //    | ChDeriv of channel:string * expr:BaseMultiChannelExpr


/// Derivative computation functions.
module Deriv = 

    let private add (x: BaseExprCh) (y: BaseExprCh) =
        (Expr x) + (Expr y) |> Expr.baseExprCh

    /// merges two derivative maps
    //let private merge (aGrads: DerivT) (bGrads: DerivT) : DerivT =
    //    if aGrads.FunElems <> bGrads.FunElems then
    //        failwithf "Cannot merge derivatives with different number of function elements: %A and %A."
    //            aGrads.FunElems bGrads.FunElems
    //    let jacs =
    //        (aGrads.Jacobians, bGrads.Jacobians)
    //        ||> Map.fold (fun m v vg -> 
    //            match Map.tryFind v m with
    //            | Some ovg -> m |> Map.add v (add vg ovg)
    //            | None -> m |> Map.add v vg) 
    //    {FunElems=aGrads.FunElems; Jacobians=jacs}

    ///// empty derivatives for expression
    //let private empty (expr: BaseExprCh) =
    //    {FunElems=expr.NElems; Jacobians=Map.empty}

    /// Computes the derivatives of all arguments of an expression given the derivative of the expression.
    let private derivOp (expr: BaseExpr) (eg: Map<Ch, BaseExprCh>) : Map<BaseExprCh, BaseExprCh> =    
        // TODO: get single-channel op derivative
        failwith "TODO"        

    /// Computes the derivatives of the specified expression w.r.t. all variables occuring in it.
    let computeWithRootJacobian (rootJacobian: Map<Ch, BaseExprCh>) (rootExpr: BaseExpr) : Deriv =

        // build expression info 
        let exprInfo = BaseExprGroup [rootExpr]

        /// map from an expression to the sum of incoming Jacobians (for all its channels)
        let incomingJacobian = Dictionary<BaseExpr, Map<Ch, BaseExprCh>> ()
        /// map from an expression (channel) to the set of dependants that transmitted Jacobian to the expression
        let receivedJacobiansFrom = Dictionary<BaseExprCh, HashSet<BaseExpr>> ()
        /// channels of multi-channel expression that have received Jacobian from all their dependants
        let exprChannelsWithFullJacobian = Dictionary<BaseExpr, HashSet<Ch>> ()
        /// expressions that have received Jacobians from all their dependants (for all channels)
        let exprsWithFullJacobian = Queue<BaseExpr> ()

        /// adds the specified Jacobian for `target` coming from `source`.
        let transmitJacobian (source: BaseExpr) (target: BaseExprCh) (jacobian: BaseExprCh) =
            let neededSources = exprInfo.Dependants target |> Set.ofSeq

            // add jacobian
            incomingJacobian.[target.Expr] <- 
                match incomingJacobian.TryFind target.Expr with
                | None -> Map [target.Channel, jacobian]
                | Some js -> 
                    match js |> Map.tryFind target.Channel with
                    | Some j -> js |> Map.add target.Channel (add j jacobian)
                    | None -> js |> Map.add target.Channel jacobian

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
                if not (exprChannelsWithFullJacobian.ContainsKey target.Expr) then
                    exprChannelsWithFullJacobian.[target.Expr] <- HashSet<_> ()
                exprChannelsWithFullJacobian.[target.Expr].Add target.Channel |> ignore
                if Set.ofSeq exprChannelsWithFullJacobian.[target.Expr] = Set.ofSeq target.Expr.Channels then
                    exprsWithFullJacobian.Enqueue target.Expr

        // set Jacobian of root node
        incomingJacobian.[rootExpr] <- rootJacobian
        exprsWithFullJacobian.Enqueue rootExpr

        // process Jacobians in loop
        let mutable varJacs = Map.empty
        while exprsWithFullJacobian.Count > 0 do

            let expr = exprsWithFullJacobian.Dequeue ()

            // propagate Jacobians
            let argDerivs = derivOp expr incomingJacobian.[expr]
            for KeyValue(arg, argDeriv) in argDerivs do
                transmitJacobian expr arg argDeriv

            // extract variable Jacobian
            match Expr expr with
            | Expr.VarArg vs ->
                varJacs <- varJacs |> Map.add vs (Expr incomingJacobian.[expr].[Ch.Only])
            | _ -> ()

        {
            FunElems  = (rootJacobian |> Map.toSeq |> Seq.head |> snd |> BaseExprCh.shape).[0]
            Jacobians = varJacs
        }    

    /// computes the derivatives of the specified expression w.r.t. all variables occuring in it
    let compute (rootExpr: Expr) : Deriv =
        if Debug.TraceCompile then printfn "Computing derivatives..."
        let sw = Stopwatch.StartNew()
        let rootJacCh = rootExpr.Shape |> ShapeSpec.nElem |> Expr.identityOfType rootExpr.DataType
        let rootJac = Map [Ch.Only, rootJacCh.BaseExprCh]
        let deriv = computeWithRootJacobian rootJac rootExpr.BaseExpr
        if Debug.Timing then printfn "Computing derivatives took %A" sw.Elapsed
        deriv

    /// extracts the Jacobian of the given VarSpecT
    let ofVarSpec var (deriv: Deriv) =
        match deriv.Jacobians |> Map.tryFind var with
        | Some d -> d
        | None when Debug.FailIfVarNotInDerivative -> 
            failwithf "the variable %A is not present in the expression" var
        | None -> 
            let varExpr = Expr.makeVar var
            Expr.zerosOfType varExpr.DataType [deriv.FunElems; Expr.nElems varExpr]

    /// extracts the Jacobian of the given variable
    let ofVar var deriv =
        ofVarSpec (Expr.varArg var) deriv                  


