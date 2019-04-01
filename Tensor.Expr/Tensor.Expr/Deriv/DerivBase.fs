namespace Tensor.Expr

open System.Diagnostics
open DeepNet.Utils
open Tensor.Expr.Ops


/// Provides a derivative for an op.
type IDerivableOp =    
    /// Computes the derivative w.r.t. each argument given the derivative w.r.t. the op.
    ///
    /// `dOp` is the incoming derivative, i.e. the derivative with respect to this op.
    /// Assuming that N is the number of elements of the function the derivative is being taken and
    /// the output shape of this op is M1xM2x...xMD, the incoming derivative will be of shape
    /// NxM1xM2x...xMD.
    ///
    /// The outgoing derivatives should be of shape NxK1xK2x...xKD where K1xK2x...xKD is the
    /// shape of the respective argument.
    abstract Deriv: dOp:Map<Ch, UExpr> -> Map<Arg, UExpr>

    

/// Derivatives for each variable occuring in an expression.
type Deriv = private {
    /// The number of elements of the function the derivative is taken of.
    _FunElems:   Size
    /// The derivatives w.r.t. the variables occuring in the expression.
    /// They are of the shape _FunElems x (shape of variable).
    _WrtVar:     Map<VarName, UExpr>
} with

    static member private log = Log "Deriv"

    static member private add (x: BaseExprCh) (y: BaseExprCh) =
        (UExpr x) + (UExpr y) |> UExpr.baseExprCh


    /// Merges two derivative maps by summing derivatives for variables they both have in common.
    static member merge (a: Deriv) (b: Deriv) : Deriv =
        if a.FunElems <> b.FunElems then
            failwithf "Cannot merge derivatives with different number of function elements: %A and %A."
                      a.FunElems b.FunElems
        let mergeMap a b =
            (a, b)
            ||> Map.fold (fun m v vg -> 
                match Map.tryFind v m with
                | Some ovg -> m |> Map.add v (vg + ovg)
                | None -> m |> Map.add v vg) 
        {
            _FunElems=a.FunElems
            _WrtVar=mergeMap a._WrtVar b._WrtVar
        }


    /// Computes the derivatives of all arguments of an expression given the derivative of the expression.
    static member private derivOp (expr: BaseExpr) (dExpr: Map<Ch, BaseExprCh>) : Map<BaseExprCh, BaseExprCh> =    
        let deriver = 
            match OpExtender.tryGet<IDerivableOp> expr.Op with
            | Some d -> d
            | None -> failwithf "The op %A %A is not derivable." (expr.Op.GetType()) expr.Op
        let dExpr = dExpr |> Map.map (fun _ e -> UExpr e)

        let funElems = dExpr |> Map.toSeq |> Seq.head |> snd |> UExpr.shape |> List.head
        for KeyValue(ch, dCh) in dExpr do
            if not (Shape.equalIgnoringBc dCh.Shape (funElems :: expr.[ch].Shape)) then
                failwithf "Derivative for channel %A of op %A with shape %A has invalid shape %A."
                    ch expr.Op expr.[ch].Shape dCh.Shape

        let dArgs = deriver.Deriv dExpr |> Map.map (fun _ e -> e.BaseExprCh)

        for KeyValue(arg, dArg) in dArgs do
            if not (Shape.equalIgnoringBc dArg.Shape (funElems :: expr.Args.[arg].Shape)) then
                failwithf "Derivative for argument %A of op %A with shape %A has invalid shape %A."
                    arg expr.Op expr.Args.[arg].Shape dArg.Shape

        let dArgExprs =
            expr.Args
            |> Map.toSeq
            |> Seq.map (fun (arg, argExpr) -> 
                match dArgs |> Map.tryFind arg with
                | Some dArgExpr -> argExpr, dArgExpr
                | None -> 
                    failwithf "The op %A %A did not provide a derivative w.r.t. its argument %A." 
                        (expr.Op.GetType()) expr.Op arg)
            |> Map.ofSeq
        dArgExprs


    /// Computes the derivatives of the specified expression w.r.t. all variables occuring in it.
    static member baseCompute (rootDeriv: Map<Ch, BaseExprCh>) (rootExpr: BaseExpr) =
        
        // build expression info 
        let exprInfo = BaseExprGroup [rootExpr]

        /// map from an expression to the sum of incoming derivatives (for all its channels)
        let incomingDeriv = Dictionary<BaseExpr, Map<Ch, BaseExprCh>> ()
        /// map from an expression (channel) to the set of dependants that transmitted derivatives to the expression
        let receivedFrom = Dictionary<BaseExprCh, HashSet<BaseExpr>> ()
        /// channels of multi-channel expression that have received derivatives from all their dependants
        let exprChsWithFullDeriv = Dictionary<BaseExpr, HashSet<Ch>> ()
        /// expressions that have received derivatives from all their dependants (for all channels)
        let exprsWithFullDeriv = Queue<BaseExpr> ()

        /// adds the specified Jacobian for `target` coming from `source`.
        let transmitDeriv (source: BaseExpr) (target: BaseExprCh) (deriv: BaseExprCh) =
            let neededSources = exprInfo.Dependants target |> Set.ofSeq

            // add jacobian
            incomingDeriv.[target.Expr] <- 
                match incomingDeriv.TryFind target.Expr with
                | None -> Map [target.Channel, deriv]
                | Some js -> 
                    match js |> Map.tryFind target.Channel with
                    | Some j -> js |> Map.add target.Channel (Deriv.add j deriv)
                    | None -> js |> Map.add target.Channel deriv

            // add to received set
            if not (receivedFrom.ContainsKey target) then
                receivedFrom.[target] <- HashSet<_> (HashIdentity.Structural)
            if receivedFrom.[target].Contains source then
                failwithf "Derivative from %A to %A was transmitted more than once." source target
            if not (neededSources.Contains source) then
                failwithf "%A received derivative from non-dependant %A." target source
            receivedFrom.[target].Add source |> ignore

            // check if target has received all derivatives
            let receivedSources = receivedFrom.[target] |> Set.ofSeq
            if receivedSources = neededSources then 
                if not (exprChsWithFullDeriv.ContainsKey target.Expr) then
                    exprChsWithFullDeriv.[target.Expr] <- HashSet<_> ()
                exprChsWithFullDeriv.[target.Expr].Add target.Channel |> ignore
                if Set.ofSeq exprChsWithFullDeriv.[target.Expr] = Set.ofSeq target.Expr.Channels then
                    exprsWithFullDeriv.Enqueue target.Expr

        // set derivative of root node
        incomingDeriv.[rootExpr] <- rootDeriv
        exprsWithFullDeriv.Enqueue rootExpr

        // process derivatives in loop
        let mutable varDerivs = Map.empty
        while exprsWithFullDeriv.Count > 0 do
            // get op with computed derivative
            let expr = exprsWithFullDeriv.Dequeue ()
            let exprDeriv = incomingDeriv.[expr]

            match expr.Op with
            | :? VarArg as varArg ->
                // arrived at a variable: save its derivative
                varDerivs <- varDerivs |> Map.add varArg.Var.Name exprDeriv.[Ch.Default]
            | _ -> 
                // propagate derivative to arguments of op
                let argDerivs = Deriv.derivOp expr exprDeriv
                for KeyValue(arg, argDeriv) in argDerivs do
                    transmitDeriv expr arg argDeriv

        varDerivs


    /// Computes the derivative expression w.r.t. all variables occuring in it using the specified
    /// value for the derivative of the specified expression.
    static member computeWithRootDeriv (rootDeriv: UExpr) (rootExpr: UExpr) : Deriv =
        let funElems = rootDeriv.Shape.[0]
        let rootDerivShp = funElems :: rootExpr.Shape
        if not (Shape.equalIgnoringBc rootDerivShp rootDeriv.Shape) then
            failwithf "Expecting shape %A for root derivative of expression with shape %A, but got shape %A."
                      rootDerivShp rootExpr.Shape rootDeriv.Shape
        let rootDeriv = Map [Ch.Default, rootDeriv.BaseExprCh]

        Deriv.log.Info "Comptuing derivatives for %A with root derivative %A" rootExpr rootDeriv
        let sw = Stopwatch.StartNew()
        let varDerivs = Deriv.baseCompute rootDeriv rootExpr.BaseExpr
        Deriv.log.Info "Computing derivatives took %A" sw.Elapsed

        {
            _FunElems = funElems
            _WrtVar = varDerivs |> Map.map (fun _ deriv -> UExpr deriv)
        }    


    /// Computes the derivative expression w.r.t. all variables occuring in it.
    static member compute (rootExpr: UExpr) : Deriv =
        let funElems = Shape.nElem rootExpr.Shape 
        let rootJac = UExpr.identity rootExpr.DataType rootExpr.Dev funElems
        let rootDeriv = rootJac |> UExpr.reshape (funElems :: rootExpr.Shape)
        let deriv = Deriv.computeWithRootDeriv rootDeriv rootExpr
        deriv


    /// Computes the derivative expression w.r.t. all variables occuring in it.
    static member compute (rootExpr: Expr<'T>) =
        Deriv.compute rootExpr.Untyped


    /// Returns the derivative of the specified variable name.
    /// If the variable does not occur in the expression, None is returned.
    member this.Wrt (varName: VarName) =
        this._WrtVar |> Map.tryFind varName

    /// Returns the derivatives of the specified untyped variable.
    /// If the variable does not occur in the expression, zeros are returned.
    member this.Wrt (var: Var) = 
        match this._WrtVar |> Map.tryFind var.Name with
        | Some d -> d
        | None -> 
            UExpr.zeros var.DataType var.Dev [this.FunElems; Shape.nElem var.Shape]

    /// Returns the derivatives of the specified typed variable.
    /// If the variable does not occur in the expression, zeros are returned.
    member this.Wrt (var: Var<'T>) = 
        this.Wrt var.Untyped |> Expr<'T>

    /// Number of function elements.
    member this.FunElems = this._FunElems

