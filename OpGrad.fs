module OpGrad

open Shape
open Op


let rec grad wrt expr =    
    // We assume that all operands have compatible size. 
    // For elementwise operations we assume that a and b are already broadcasted
    // to have the *same* size.
    let wrtShape = shapeOf wrt
    let wrtElems = ShapeSpec.nElem wrtShape
    let zeroGrad ss = zeroMatrix (ShapeSpec.nElem ss) wrtElems
    let isZero g = 
        match g with 
        | Leaf(Zeros _) | Unary(Annotated _, Leaf(Zeros _)) -> true 
        | _ -> false
   
    let subgrad x = 
        let g = grad wrt x
        let sg = shapeOf g
        if ShapeSpec.nDim sg <> 2 then
            failwithf "gradient must have two dimensions but it has shape %A" sg
        if not (sg.[1] .= wrtElems) then
            failwithf "gradient second dimensions must have %A elements (same as wrt) but it has %A elements" wrtElems sg.[1]
        g

    /// flattens all but the last dimension into one dimension
    let collapse g =
        let sg = shapeOf g
        let funElems = sg.[0 .. (ShapeSpec.nDim sg - 2)] |> ShapeSpec.nElem
        let wrtElems = sg.[ShapeSpec.nDim sg - 1]
        g |> reshape [funElems; wrtElems]

    match expr with
    | expr when expr = wrt ->
        idMatrix wrtElems wrtElems
    | expr when not (contains wrt expr) ->
        zeroGrad (shapeOf expr)

    | Leaf(op) ->                  
        match op with
        | Zeros ss -> zeroGrad ss
        | ScalarConst _ -> zeroGrad ShapeSpec.scalar
        | TensorConst (_, ss) -> zeroGrad ss
        | Identity ss -> zeroGrad ss
        | Var v -> zeroGrad (VarSpec.shape v)                  

    | Unary(op, a) ->
        let ga = subgrad a
        let sa = shapeOf a
        let sga = shapeOf ga
        let gaExpanded = ga |> reshape (sa @ [wrtElems])

        match op with
        | Negate -> -ga
        | Log -> a ** -1. * ga
        | Exp -> exp a * ga
        | SwapDim (ax1, ax2) -> gaExpanded |> swapDim ax1 ax2 |> collapse
        | Reshape ss -> ga 
        | Broadcast ss -> gaExpanded |> broadcast (ss @ [wrtElems]) |> collapse
        | Sum -> ga |> sumAxis 0 |> collapse
        | SumAxis ax -> gaExpanded |> sumAxis ax |> collapse
        | Annotated ano -> ga |> annotate ano 

    | Binary(op, a, b) ->
        let ga, gb = subgrad a, subgrad b
        let sa, sb = shapeOf a, shapeOf b
        let sga, sgb = shapeOf ga, shapeOf gb
        let gaExpanded = ga |> reshape (sa @ [wrtElems])
        let gbExpanded = gb |> reshape (sb @ [wrtElems])

        let inline (.+) gaDep gbDep =            
            if isZero gb then gaDep
            elif isZero ga then gbDep
            else gaDep + gbDep

        match op with            
        | Add -> ga .+ gb
        | Substract -> ga - gb
        | Multiply -> ga * (padRight b) .+ (padRight a) * gb 
        | Divide -> subgrad (a * b ** -1.)
        | Power -> padRight (a**(b-1.) * b) * ga .+ padRight (a**b * log a) * gb
        | Dot -> 
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 1, 1 -> subgrad (sum(a * b))
                | 2, 1 -> (b %* (idMatrix sa.[0] sa.[0])) .* ga .+ a .* gb
                | 2, 2 when sa.[1] = sb.[0] ->  // TODO: fix gradient wrt a
                    ((b ** T) %* (idMatrix sa.[0] sa.[0])) .* ga .+ ((idMatrix sb.[1] sb.[1]) %* a) .* gb
                | _ -> failshape op sa sb 
        | TensorProduct -> (gaExpanded %* b) .+ (a %* gbExpanded) |> collapse
    |> annotate (GradOf expr) 

