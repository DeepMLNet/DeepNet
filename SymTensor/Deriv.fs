namespace SymTensor

open Basics

[<AutoOpen>]
module DerivTypes =
    open Expr

    /// Jacobians for each variable
    type DerivT = {
        /// the expression the derivative was calculated of
        Expr:       ExprT
        /// the Jacobians w.r.t. the variables occuring in the expression
        Jacobians:  Map<VarSpecT, ExprT>
    }


/// derivative calculation
module Deriv =
    open Expr

    /// merges two derivative maps
    let private merge (aGrads: DerivT) (bGrads: DerivT) : DerivT =
        if aGrads.Expr <> bGrads.Expr then
            failwith "derivatives must belong to same expression"
        let jacs =
            (aGrads.Jacobians, bGrads.Jacobians)
            ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                        | Some ovg -> m |> Map.add v (vg + ovg)
                                        | None -> m |> Map.add v vg) 
        {Expr=aGrads.Expr; Jacobians=jacs}

    /// empty derivatives for expression
    let private empty expr =
        {Expr=expr; Jacobians=Map.empty}

    /// reverse accumulation autodifferentiation of an expression
    let rec private reverseDiffStep (baseExpr: ExprT) (expr: ExprT) (eg: ExprT) : DerivT =    
        let rds = reverseDiffStep baseExpr
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
                (shapeOf eg).[1] (ShapeSpec.nElem (shapeOf expr))

        /// expands the second dimension of the the Jacobian into the shape of this expression
        let egExpanded =
            eg |> reshape (funElems :: (shapeOf expr))

        /// flattens all but the first dimension into one dimension
        let collapse g =
            let wrtElems = (shapeOf g).[1..] |> ShapeSpec.nElem
            g |> reshape [funElems; wrtElems]

        /// total derivates given op's derivates
        let totalDerivates es des =
            (empty baseExpr, List.zip es des)
            ||> List.fold (fun totGrad (e, de) ->
                let eGrad = rds e de
                merge totGrad eGrad)

        /// logic op failure
        let failLogic op =
            failwithf "cannot calculate derivative of logic or comparison operation %A" op

        // useful numbers
        let zero = Expr.zeroOfSameType expr
        let one = Expr.oneOfSameType expr
        let two = Expr.twoOfSameType expr
        let scalar = Expr.scalarOfSameType expr
        let zeros = Expr.zerosOfSameType expr

        match expr with
        | Leaf(op) ->                  
            match op with
            | ScalarConst _ -> empty baseExpr
            | SizeValue _ -> empty baseExpr
            | Identity _ -> empty baseExpr
            | Var v -> {empty baseExpr with Jacobians=Map [v, eg]}

        | Unary(op, a) ->
            match op with
            | Negate -> -eg |> rds a
            | Abs -> egExpanded * padLeft (signt a) |> collapse |> rds a
            | SignT -> empty baseExpr
            | Log -> egExpanded * padLeft (a ** (-one)) |> collapse |> rds a
            | Log10 -> eg |> rds (log a / log (scalar 10))
            | Exp -> egExpanded * padLeft (exp a) |> collapse |> rds a
            | Sin -> egExpanded * padLeft (cos a) |> collapse |> rds a
            | Cos -> egExpanded * padLeft (-sin a) |> collapse |> rds a
            | Tan -> egExpanded * padLeft (one + (tan a)**two) |> collapse |> rds a
            | Asin -> egExpanded * padLeft (one / sqrtt (one - a**two)) |> collapse |> rds a
            | Acos -> egExpanded * padLeft (-one / sqrtt (one - a**two)) |> collapse |> rds a
            | Atan -> egExpanded * padLeft (one / (one + a**two)) |> collapse |> rds a
            | Sinh -> egExpanded * padLeft (cosh a) |> collapse |> rds a
            | Cosh -> egExpanded * padLeft (sinh a) |> collapse |> rds a
            | Tanh -> egExpanded * padLeft (one - (tanh a)**two) |> collapse |> rds a
            | Sqrt -> egExpanded * padLeft (one / (two * sqrtt a)) |> collapse |> rds a
            | Ceil -> empty baseExpr
            | Floor -> empty baseExpr
            | Round -> empty baseExpr
            | Truncate -> empty baseExpr
            
            | Not -> failLogic op

            | Diag (ax1, ax2) -> egExpanded |> diagMatAxis (ax1 + 1) (ax2 + 1) |> collapse |> rds a
            | DiagMat (ax1, ax2) -> egExpanded |> diagAxis (ax1 + 1) (ax2 + 1) |> collapse |> rds a
            | Invert -> -(padLeft expr.T) .* egExpanded .* (padLeft expr.T) |> collapse |> rds a
            | PermuteAxes perm -> 
                let backPerm = Permutation.invert perm
                let egePerm = 
                    0 :: List.map (fun p -> p + 1) backPerm
                egExpanded |> permuteAxes egePerm |> collapse |> rds a

            | Subtensor srs ->
                let agExpanded = zeros (funElems :: (shapeOf a))
                setSubtensor agExpanded.[SRSAll :: srs] egExpanded
                |> collapse 
                |> rds a
            | Reshape ss -> eg |> rds a
            | DoBroadcast ss -> 
                let mutable egUnbroadcasted = egExpanded
                for ax, (eSize, aSize) in List.indexed (List.zip ss (shapeOf a)) do
                    match eSize, aSize with
                    | SizeSpecT.Broadcast, SizeSpecT.Broadcast -> ()
                    | _, SizeSpecT.Broadcast ->
                        egUnbroadcasted <- egUnbroadcasted |> sumKeepingAxis (ax + 1)
                    | _ -> ()
                egUnbroadcasted |> collapse |> rds a
            | Sum -> eg |> enableBroadcast 1 |> broadcast (funElems :: ShapeSpec.flatten (shapeOf a)) 
                        |> collapse |> rds a
            | SumAxis ax -> 
                let eeg = egExpanded 
                let bca = eeg |> reshape (shapeOf eeg |> ShapeSpec.insertBroadcastAxis (ax + 1))
                let ael = (shapeOf a).[ax]
                let bc = bca |> broadcast (shapeOf bca |> ShapeSpec.set (ax + 1) ael)
                bc |> collapse |> rds a
            | StoreToVar _ -> eg |> rds a

            | NullifyJacobian ->
                Expr.zerosLike eg |> rds a
            | AssumeJacobian jac ->
                let jacBc =
                    match eg.Shape.[0], jac.Shape.[0] with
                    | fl, jl when fl = jl -> jac
                    | fl, jl when jl = SizeSpec.broadcastable ->
                        jac |> Expr.broadcast [fl; jac.Shape.[1]]
                    | _ -> 
                        failwithf "cannot broadcast specified Jacobian of shape %A to required 
                                   Jacobian shape %A" jac.Shape eg.Shape
                jacBc |> rds a

            | Print _ -> eg |> rds a
            | Dump _ -> eg |> rds a
            | Annotated _ -> eg |> rds a
            | CheckFinite name ->
                eg |> checkFinite (sprintf "(partial) Jacobian wrt %s" name) |> rds a

        | Binary(op, a, b) ->
            let inline (.+) da db = totalDerivates [a; b] [da; db]

            match op with            
            | Add -> eg .+ eg
            | Substract -> eg .+ (-eg)
            | Multiply -> ((egExpanded * (padLeft b)) |> collapse) .+
                          ((egExpanded * (padLeft a)) |> collapse)
            | Divide -> eg |> rds (a * b ** (-one))
            | Modulo -> 
                failwith "Modulo gradient is broken"
                eg .+ (padLeft (-truncate (a / b)) |> collapse) 
            | Power -> (egExpanded * padLeft (b * a**(b - one)) |> collapse) .+ 
                       (egExpanded * padLeft (a**b * log a) |> collapse)
            
            | MaxElemwise -> eg |> rds (ifThenElse (a >>>> b) a b)
            | MinElemwise -> eg |> rds (ifThenElse (a <<<< b) a b)

            | Equal
            | Less
            | LessEqual
            | Greater
            | GreaterEqual
            | NotEqual
                -> failLogic op

            | And 
            | Or 
                -> failLogic op

            | IfThenElse cond ->
                let egZeros = zerosLike egExpanded
                let da = ifThenElse (padLeft cond) egExpanded egZeros |> collapse
                let db = ifThenElse (padLeft cond) egZeros egExpanded |> collapse
                da .+ db

            | Dot -> 
                /// Jacobian of y = m .* x wrt x
                let mxWrtX (m: ExprT) x y dy =
                    let xShp, yShp, dyShp = shapeOf x, shapeOf y, shapeOf dy
                    let nd = ShapeSpec.nDim xShp
                    let batchShp = xShp.[0..nd-3]
                    let batchElems = ShapeSpec.nElem batchShp
                    let xSmplShp, ySmplShp = xShp.[nd-2..], yShp.[nd-2..]
                    let funElems = dyShp.[0]
                    let dyMat = dy |> swapDim 0 1 |> reshape (batchShp @ [ySmplShp.[0]; ySmplShp.[1] * funElems])
                    let dxMat = m.T .* dyMat
                    let dx = dxMat |> reshape [batchElems * xSmplShp.[0] * xSmplShp.[1]; funElems] |> swapDim 1 0
                    dx

                // Jacobian wrt b
                let db = mxWrtX a b expr eg

                // calculate Jacobian wrt a by transposing expression and resulting Jacobian
                let aShp = shapeOf a
                let nd = ShapeSpec.nDim aShp
                let batchShp = aShp.[0..nd-3]
                let egT = egExpanded.T |> collapse
                let daT = mxWrtX (b.T) (a.T) (expr.T) egT
                let da = daT |> reshape ([funElems] @ batchShp @ [aShp.[nd-1]; aShp.[nd-2]]) |> transpose |> collapse

                da .+ db
            | TensorProduct -> failwith "not implemented"
            | SetSubtensor sr ->
                let bgExpanded = egExpanded.[SRSAll::sr]
                let agExpanded = setSubtensor egExpanded.[SRSAll::sr] (zerosLike bgExpanded)
                (agExpanded |> collapse) .+ (bgExpanded |> collapse)

        | Nary(op, es) ->
            match op with
            | Elements (resShape, elemExpr) ->
                let desElemExprs = ElemExprDeriv.buildDerivElemExpr elemExpr resShape (es.Length)
                let des = 
                    List.zip es desElemExprs
                    |> List.map (fun (e, deElemExpr) -> 
                        let deShp = funElems :: (shapeOf e)
                        let deArgs = es @ [egExpanded]
                        Expr.elements deShp deElemExpr deArgs |> collapse)
                totalDerivates es des
            | Interpolate ip -> 
                match ip.Mode with
                | InterpolateLinearaly ->
                    let des = 
                        [for d=0 to es.Length-1 do
                            let ipd = ip |> Interpolator.getDerivativeOfInterpolator d 
                            yield egExpanded * padLeft (Expr.interpolate ipd es)]
                    totalDerivates es des
                | InterpolateToLeft -> empty baseExpr

            | ExtensionOp eop -> eop.Deriv eg es |> totalDerivates es                
            | Discard -> failwith "cannot propagate derivative thorugh Discard op"


    /// computes the derivatives of the specified expression w.r.t. all variables occuring in it
    let compute (expr: ExprT) : DerivT =
        let eg = shapeOf expr |> ShapeSpec.nElem |> identityOfSameType expr
        reverseDiffStep expr expr eg

    /// extracts the Jacobian of the given variable
    let ofVar var (deriv: DerivT) =
        match deriv.Jacobians |> Map.tryFind (extractVar var) with
        | Some d -> d
        | None when Debug.FailIfVarNotInDerivative -> 
            failwithf "the variable %A is not present in the expression" (extractVar var)
        | None -> Expr.zerosOfSameType var [Expr.nElems deriv.Expr ; Expr.nElems var]
            
        



