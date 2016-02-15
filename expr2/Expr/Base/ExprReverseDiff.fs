module ExprReverseDiff

open Shape
open Op

/// map containing the Jacobian for each variable
type VarDiffs = Map<VarSpecT, ExprT>

/// reverse accumulation autodifferentiation of an expression
let rec reverseDiffStep (expr: ExprT) (eg: ExprT) : VarDiffs =    
    let exprShp = shapeOf expr
    let funElems = (shapeOf eg).[0]  

    //printfn "expr=%A" expr
    if (shapeOf eg).[1] <> ShapeSpec.nElem (shapeOf expr) then
        failwithf "gradient with %A wrt elements was specified for expression with %A elements"
            (shapeOf eg).[1] (ShapeSpec.nElem (shapeOf expr))

    /// expands the second dimension of the the Jacobian into the shape of this expression
    let egExpanded =
        eg |> reshape ((shapeOf eg).[0] :: (shapeOf expr))

    /// flattens all but the first dimension into one dimension
    let collapse g =
        let wrtElems = (shapeOf g).[1..] |> ShapeSpec.nElem
        g |> reshape [funElems; wrtElems]

    match expr with
    | Leaf(op) ->                  
        match op with
        | Zeros ss -> Map.empty
        | ScalarConst _ -> Map.empty
        | TensorConst (_, ss) -> Map.empty
        | DiagonalOne ss -> Map.empty
        | Var v -> Map.empty |> Map.add v eg

    | Unary(op, a) ->
        match op with
        | Negate -> -eg |> reverseDiffStep a
        | Log -> egExpanded * (padLeft a) ** -1.0f |> collapse |> reverseDiffStep a
        | Exp -> egExpanded * (padLeft expr) |> collapse |> reverseDiffStep a
        | SwapDim (ax1, ax2) -> egExpanded |> swapDim (ax1 + 1) (ax2 + 1) |> collapse |> reverseDiffStep a
        | Reshape ss -> eg |> reverseDiffStep a
        | Broadcast ss -> 
            let mutable egUnbroadcasted = egExpanded
            for ax, (eSize, aSize) in List.indexed (List.zip ss (shapeOf a)) do
                match eSize, aSize with
                | Shape.Broadcast, Shape.Broadcast -> ()
                | _, Shape.Broadcast ->
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
        | Annotated ano -> eg |> reverseDiffStep a

    | Binary(op, a, b) ->
        let inline (.+) da db =   
            let aGrads, bGrads = reverseDiffStep a da, reverseDiffStep b db         
            Map.fold (fun m v vg -> match Map.tryFind v m with
                                    | Some ovg -> m |> Map.add v (vg + ovg)
                                    | None -> m |> Map.add v vg) 
                aGrads bGrads

        match op with            
        | Add -> eg .+ eg
        | Substract -> eg .+ (-eg)
        | Multiply -> ((egExpanded * (padLeft b)) |> collapse) .+
                      ((egExpanded * (padLeft a)) |> collapse)
        | Divide -> eg |> reverseDiffStep (a * b ** -1.0f)
        | Power -> (egExpanded * padLeft (b * a**(b-1.0f)) |> collapse) .+ 
                   (egExpanded * padLeft (a**b * log a) |> collapse)
        | Dot -> 
            /// Jacobian of y = m .* x wrt x
            let mxWrtX m x y dy =
                let xShp, yShp, dyShp = shapeOf x, shapeOf y, shapeOf dy
                let funElems = dyShp.[0]
                let dyMat = dy |> swapDim 0 1 |> reshape [yShp.[0]; yShp.[1] * funElems]
                let dxMat = m**T .* dyMat
                let dx = dxMat |> reshape [xShp.[0] * xShp.[1]; funElems] |> swapDim 1 0
                dx

            // Jacobian wrt b
            let db = mxWrtX a b expr eg

            // calculate Jacobian wrt a by transposing expression and resulting Jacobian
            let aShp = shapeOf a
            let egT = egExpanded |> swapDim 1 2 |> collapse
            let daT = mxWrtX (b**T) (a**T) (expr**T) egT
            let da = daT |> reshape [funElems; aShp.[1]; aShp.[0]] |> swapDim 1 2 |> collapse

            da .+ db
        | TensorProduct -> failwith "not implemented"

    | Nary(op, es) ->
        match op with
        | Discard -> failwith "cannot propagte gradient thorugh discard"


/// reverse accumulation autodifferentiation of an expression
let reverseDiff expr =
    let eg = shapeOf expr |> ShapeSpec.nElem |> idMatrix
    reverseDiffStep expr eg

/// extracts the Jacobian of the given variable
let diffOf var (varDiffs: VarDiffs) =
    varDiffs.[extractVar var]

