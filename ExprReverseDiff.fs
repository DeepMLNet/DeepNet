module ExprReverseDiff

open Shape
open Op


/// reverse accumulation autodifferentiation of an expression
let rec reverseDiff (expr: Expr) (eg: Expr) : Map<VarSpecT, Expr> =    
    let exprShp = shapeOf expr
    let funElems = (shapeOf eg).[0]  

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
        | Identity ss -> Map.empty
        | Var v -> Map.empty |> Map.add v eg

    | Unary(op, a) ->
        match op with
        | Negate -> -eg |> reverseDiff a
        | Log -> egExpanded * (padLeft a) ** -1. |> collapse |> reverseDiff a
        | Exp -> egExpanded * (padLeft expr) |> collapse |> reverseDiff a
        | SwapDim (ax1, ax2) -> egExpanded |> swapDim (ax1 + 1) (ax2 + 1) |> collapse |> reverseDiff a
        | Reshape ss -> eg |> reverseDiff a
        | Broadcast ss -> egExpanded |> broadcast (funElems :: ss) |> collapse |> reverseDiff a
        | Sum -> eg |> enableBroadcast 1 |> broadcast (funElems :: ShapeSpec.flatten (shapeOf a)) 
                    |> collapse |> reverseDiff a
        | SumAxis ax -> 
            let eeg = egExpanded 
            let bca = eeg |> reshape (shapeOf eeg |> ShapeSpec.insertBroadcastAxis (ax + 1))
            let ael = (shapeOf a).[ax]
            let bc = bca |> broadcast (shapeOf bca |> ShapeSpec.set (ax + 1) ael)
            bc |> collapse |> reverseDiff a
        | Annotated ano -> eg |> reverseDiff a

    | Binary(op, a, b) ->
        let inline (.+) aGrads bGrads =            
            Map.fold (fun m v vg -> match Map.tryFind v m with
                                    | Some ovg -> m |> Map.add v (vg + ovg)
                                    | None -> m |> Map.add v vg) 
                aGrads bGrads

        match op with            
        | Add -> (eg |> reverseDiff a) .+ (eg |> reverseDiff b)
        | Substract -> (eg |> reverseDiff a) .+ (-eg |> reverseDiff b)
        | Multiply -> ((egExpanded * (padLeft b)) |> collapse |> reverseDiff a) .+
                      ((egExpanded * (padLeft a)) |> collapse |> reverseDiff b)
        | Divide -> eg |> reverseDiff (a * b ** -1.)
        | Power -> (egExpanded * padLeft (a**(b-1.)) |> collapse |> reverseDiff a) .+ 
                   (egExpanded * padLeft (a**b * log a) |> collapse |> reverseDiff b)
        | Dot -> 
            let mxWrtX m x y dy =
                let xShp, yShp, dyShp = shapeOf x, shapeOf y, shapeOf dy
                let funElems = dyShp.[0]
                let dyMat = dy |> swapDim 0 1 |> reshape [yShp.[0]; yShp.[1] * funElems]
                let dxMat = m**T .* dyMat
                let dx = dxMat |> reshape [xShp.[0] * xShp.[1]; funElems] |> swapDim 1 0
                dx

            let bShp = shapeOf b
            let bg = mxWrtX a b expr eg
            let ag = mxWrtX (b**T) (a**T) expr eg 
                     |> reshape [funElems; bShp.[1]; bShp.[0]] |> swapDim 1 2 |> collapse

            (ag |> reverseDiff a) .+ (bg |> reverseDiff b)
        | TensorProduct -> failwith "not implemented"

