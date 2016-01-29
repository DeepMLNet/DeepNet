module ExprReverseDiff

open Shape
open Op


/// reverse accumulation autodifferentiation of an expression
let rec reverseDiff (expr: Expr) (eg: Expr) : Map<VarSpecT, Expr> =    
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
            let sa, sb = shapeOf a, shapeOf b
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 1, 1 -> eg |> reverseDiff (sum(a * b))
                | 2, 1 -> (eg .* (b %* (idMatrix sa.[0] sa.[0])) |> reverseDiff a) .+
                          (eg .* a |> reverseDiff b)
                | 2, 2 when sa.[1] = sb.[0] ->  // TODO: fix gradient wrt a
                    (eg .* ((b ** T) %* (idMatrix sa.[0] sa.[0])) |> reverseDiff a) .+
                    (eg .* ((idMatrix sb.[1] sb.[1]) %* a) |> reverseDiff b)
                | _ -> failshape op sa sb 
        | TensorProduct -> failwith "not implemented"
        // (gaExpanded %* b) .+ (a %* gbExpanded) |> collapse

