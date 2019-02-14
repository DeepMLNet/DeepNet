namespace SymTensor

open System.Diagnostics
open DeepNet.Utils


[<AutoOpen>]
module DerivTypes =

    /// Jacobians for each variable
    type DerivT = {
        /// the number of elements of the function the derivative is taken of
        FunElems:   SizeSpec
        /// the Jacobians w.r.t. the variables occuring in the expression
        Jacobians:  Map<Var, Expr>
    }


/// Derivative computation functions.
module Deriv =

    /// merges two derivative maps
    let private merge (aGrads: DerivT) (bGrads: DerivT) : DerivT =
        if aGrads.FunElems <> bGrads.FunElems then
            failwithf "cannot merge derivatives with different number of function elements: %A and %A"
                aGrads.FunElems bGrads.FunElems
        let jacs =
            (aGrads.Jacobians, bGrads.Jacobians)
            ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                        | Some ovg -> m |> Map.add v (vg + ovg)
                                        | None -> m |> Map.add v vg) 
        {FunElems=aGrads.FunElems; Jacobians=jacs}

    /// empty derivatives for expression
    let private empty expr =
        {FunElems=Expr.nElems expr; Jacobians=Map.empty}


    /// calculates the Jacobian of all arguments of an expression given the Jacobian of the expression
    let rec private reverseDiffStep (expr: Expr) (eg: Expr) : List<Expr * Expr> =    
        let exprShp = expr.Shape
        let funElems = eg.Shape.[0]  

        // check type and shape
        if expr.TypeName <> eg.TypeName then
            failwithf "Jacobian with type %A was specified for expression of type %A"
                eg.TypeName expr.TypeName
        if eg.Shape.[1] .<> expr.NElems then
            printfn "expr=\n%A" expr
            printfn "eg=\n%A" eg
            failwithf "Jacobian with %A wrt elements was specified for expression with %A elements"
                (Expr.shapeOf eg).[1] (ShapeSpec.nElem (Expr.shapeOf expr))

        // TODO: get op derivative
        failwith "TODO"
        
    /// computes the Jacobians of the arguments of a multi-channel op given the Jacobians
    /// w.r.t. all channels of the multi-channel op
    and private multiChannelDiffStep (mcOp: MultiChannelOpUsageT) (eg: Map<Channel, Expr>) : List<Expr * Expr> =
        // TODO: get op derivative
        failwith "TODO"

    /// computes the derivatives of the specified expression w.r.t. all variables occuring in it
    and computeWithRootJacobian (rootJacobian: Expr) (rootExpr: Expr) : DerivT =

        // build expression info and unify common subexpressions
        let exprInfo = ExprInfoT [rootExpr]
        let rootExpr = List.exactlyOne exprInfo.Exprs

        /// map from an expression to the sum of incoming Jacobians
        let incomingJacobian = Dictionary<Expr, Expr> (HashIdentity.Reference)
        /// map from an expression to the set of dependants that transmitted Jacobian to the expression
        let receivedJacobiansFrom = Dictionary<Expr, HashSet<Expr>> (HashIdentity.Reference)
        /// expressions that have received Jacobians from all their dependants
        let exprsWithFullJacobian = Queue<Expr> ()

        let multiChannelOpJacobians = 
            Dictionary<MultiChannelOpUsageT, Dictionary<Channel, Expr>> (HashIdentity.Structural) 
        let multiChannelOpsWithFullJacobians = Queue<MultiChannelOpUsageT> ()

        /// adds the specified Jacobian coming from `source` to `target`
        let transmitJacobian source target jacobian =
            let neededSources = exprInfo.Dependants target |> Set.ofSeq

            // add jacobian
            match incomingJacobian.TryFind target with
            | Some j -> incomingJacobian.[target] <- j + jacobian
            | None -> incomingJacobian.[target] <- jacobian

            // add to received set
            if not (receivedJacobiansFrom.ContainsKey target) then
                receivedJacobiansFrom.[target] <- HashSet<Expr> (HashIdentity.Structural)
            match source with
            | Choice1Of2 exprSource -> 
                if receivedJacobiansFrom.[target].Contains exprSource then
                    failwithf "Jacobian from %A to %A was already transmitted" exprSource target
                if not (neededSources.Contains exprSource) then
                    failwithf "%A received Jacobian from non-dependant %A" target exprSource
                receivedJacobiansFrom.[target].Add exprSource |> ignore
            | Choice2Of2 mcopSource ->
                neededSources
                |> Seq.filter (function 
                               | Nary (Channel (op, channel), es) when (op, es) = mcopSource -> true
                               | _ -> false)
                |> Seq.iter (fun src -> receivedJacobiansFrom.[target].Add src |> ignore)

            // check if target has received all Jacobians
            let receivedSources = receivedJacobiansFrom.[target] |> Set.ofSeq
            if receivedSources = neededSources then exprsWithFullJacobian.Enqueue target

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
        while exprsWithFullJacobian.Count > 0 || multiChannelOpsWithFullJacobians.Count > 0 do

            if exprsWithFullJacobian.Count > 0 then
                let expr = exprsWithFullJacobian.Dequeue ()

                // propagate Jacobians
                match expr with
                | Nary (Channel (op, channel), es) ->
                    transmitMultiChannelOpJacobian (op, es) channel incomingJacobian.[expr]
                | _ ->
                    reverseDiffStep expr incomingJacobian.[expr] |> transmitJacobians (Choice1Of2 expr)

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


