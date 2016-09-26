namespace SymTensor

open Basics

[<AutoOpen>]
module DerivTypes =
    open Expr

    /// map containing the Jacobian for each variable
    type DerivT = Map<VarSpecT, ExprT>


module Deriv =
    open Expr

    /// merges to derivative maps
    let private merge (aGrads: DerivT) (bGrads: DerivT) : DerivT =
        (aGrads, bGrads)
        ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                    | Some ovg -> m |> Map.add v (vg + ovg)
                                    | None -> m |> Map.add v vg) 

    /// reverse accumulation autodifferentiation of an expression
    let rec reverseDiffStep (expr: ExprT) (eg: ExprT) : DerivT =    
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
            (Map.empty, List.zip es des)
            ||> List.fold (fun totGrad (e, de) ->
                let eGrad = reverseDiffStep e de
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
            | Zeros _ -> Map.empty
            | ScalarConst _ -> Map.empty
            | SizeValue _ -> Map.empty
            | Identity _ -> Map.empty
            | Var v -> Map.empty |> Map.add v eg

        | Unary(op, a) ->
            match op with
            | Negate -> -eg |> reverseDiffStep a
            | Abs -> egExpanded * padLeft (signt a) |> collapse |> reverseDiffStep a
            | SignT -> Map.empty
            | Log -> egExpanded * padLeft (a ** (-one)) |> collapse |> reverseDiffStep a
            | Log10 -> eg |> reverseDiffStep (log a / log (scalar 10))
            | Exp -> egExpanded * padLeft (exp a) |> collapse |> reverseDiffStep a
            | Sin -> egExpanded * padLeft (cos a) |> collapse |> reverseDiffStep a
            | Cos -> egExpanded * padLeft (-sin a) |> collapse |> reverseDiffStep a
            | Tan -> egExpanded * padLeft (one + (tan a)**two) |> collapse |> reverseDiffStep a
            | Asin -> egExpanded * padLeft (one / sqrtt (one - a**two)) |> collapse |> reverseDiffStep a
            | Acos -> egExpanded * padLeft (-one / sqrtt (one - a**two)) |> collapse |> reverseDiffStep a
            | Atan -> egExpanded * padLeft (one / (one + a**two)) |> collapse |> reverseDiffStep a
            | Sinh -> egExpanded * padLeft (cosh a) |> collapse |> reverseDiffStep a
            | Cosh -> egExpanded * padLeft (sinh a) |> collapse |> reverseDiffStep a
            | Tanh -> egExpanded * padLeft (one - (tanh a)**two) |> collapse |> reverseDiffStep a
            | Sqrt -> egExpanded * padLeft (one / (two * sqrtt a)) |> collapse |> reverseDiffStep a
            | Ceil -> Map.empty
            | Floor -> Map.empty
            | Round -> Map.empty
            | Truncate -> Map.empty
            
            | Not -> failLogic op

            | Diag (ax1, ax2) -> egExpanded |> diagMatAxis (ax1 + 1) (ax2 + 1) |> collapse |> reverseDiffStep a
            | DiagMat (ax1, ax2) -> egExpanded |> diagAxis (ax1 + 1) (ax2 + 1) |> collapse |> reverseDiffStep a
            | Invert -> -(padLeft expr.T) .* egExpanded .* (padLeft expr.T) |> collapse |> reverseDiffStep a
            | PermuteAxes perm -> 
                let backPerm = Permutation.invert perm
                let egePerm = 
                    0 :: List.map (fun p -> p + 1) backPerm
                egExpanded |> permuteAxes egePerm |> collapse |> reverseDiffStep a

            | Subtensor srs ->
                let agExpanded = zeros (funElems :: (shapeOf a))
                setSubtensor agExpanded.[SRSAll :: srs] egExpanded
                |> collapse 
                |> reverseDiffStep a
            | Reshape ss -> eg |> reverseDiffStep a
            | DoBroadcast ss -> 
                let mutable egUnbroadcasted = egExpanded
                for ax, (eSize, aSize) in List.indexed (List.zip ss (shapeOf a)) do
                    match eSize, aSize with
                    | SizeSpecT.Broadcast, SizeSpecT.Broadcast -> ()
                    | _, SizeSpecT.Broadcast ->
                        egUnbroadcasted <- egUnbroadcasted |> sumKeepingAxis (ax + 1)
                    | _ -> ()
                egUnbroadcasted |> collapse |> reverseDiffStep a
            | Sum -> eg |> enableBroadcast 1 |> broadcast (funElems :: ShapeSpec.flatten (shapeOf a)) 
                        |> collapse |> reverseDiffStep a
            | SumAxis ax -> 
                let eeg = egExpanded 
                let bca = eeg |> reshape (shapeOf eeg |> ShapeSpec.insertBroadcastAxis (ax + 1))
                let ael = (shapeOf a).[ax]
                let bc = bca |> broadcast (shapeOf bca |> ShapeSpec.set (ax + 1) ael)
                bc |> collapse |> reverseDiffStep a
            | StoreToVar _ -> eg |> reverseDiffStep a

            | NullifyJacobian ->
                Expr.zerosLike eg |> reverseDiffStep a
            | AssumeJacobian jac ->
                let jacBc =
                    match eg.Shape.[0], jac.Shape.[0] with
                    | fl, jl when fl = jl -> jac
                    | fl, jl when jl = SizeSpec.broadcastable ->
                        jac |> Expr.broadcast [fl; jac.Shape.[1]]
                    | _ -> 
                        failwithf "cannot broadcast specified Jacobian of shape %A to required 
                                   Jacobian shape %A" jac.Shape eg.Shape
                jacBc |> reverseDiffStep a

            | Print _ -> eg |> reverseDiffStep a
            | Dump _ -> eg |> reverseDiffStep a
            | Annotated _ -> eg |> reverseDiffStep a
            | CheckFinite name ->
                eg |> checkFinite (sprintf "(partial) Jacobian wrt %s" name) |> reverseDiffStep a

        | Binary(op, a, b) ->
            let inline (.+) da db = totalDerivates [a; b] [da; db]

            match op with            
            | Add -> eg .+ eg
            | Substract -> eg .+ (-eg)
            | Multiply -> ((egExpanded * (padLeft b)) |> collapse) .+
                          ((egExpanded * (padLeft a)) |> collapse)
            | Divide -> eg |> reverseDiffStep (a * b ** (-one))
            | Modulo -> 
                failwith "Modulo gradient is broken"
                eg .+ (padLeft (-truncate (a / b)) |> collapse) 
            | Power -> (egExpanded * padLeft (b * a**(b - one)) |> collapse) .+ 
                       (egExpanded * padLeft (a**b * log a) |> collapse)
            
            | MaxElemwise -> eg |> reverseDiffStep (ifThenElse (a >>>> b) a b)
            | MinElemwise -> eg |> reverseDiffStep (ifThenElse (a <<<< b) a b)

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
                | InterpolateToLeft -> Map.empty

            | ExtensionOp eop -> eop.Deriv eg es |> totalDerivates es                
            | Discard -> failwith "cannot propagate derivative thorugh Discard op"


    /// reverse accumulation autodifferentiation of an expression
    let compute (expr: ExprT) : DerivT =
        let eg = shapeOf expr |> ShapeSpec.nElem |> identityOfSameType expr
        reverseDiffStep expr eg

    /// extracts the Jacobian of the given variable
    let ofVar var (varDiffs: DerivT) =
        match varDiffs |> Map.tryFind (extractVar var) with
        | Some d -> d
        | None -> failwithf "the variable %A is not present in the expression" (extractVar var)



