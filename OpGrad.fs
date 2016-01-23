module OpGrad

open Shape
open Op


let rec grad op wrt =    
    let g =
        // We assume that all operands have compatible size. 
        // For elementwise operations we assume that a and b are already broadcasted
        // to have the *same* size.
        let subgrad x = 
            let g = grad x wrt
            if shapeOf g |> ShapeSpec.nDim <> 2 then
                failwithf "gradient must have two dimensions but it has shape %A" (shapeOf g)
            g
        let constGrad ss = Zeros [ShapeSpec.nElem ss; ShapeSpec.nElem (shapeOf (Var wrt))]
        match op with        
        | Add(a, b) -> Add(subgrad a, subgrad b)
        | Substract(a, b) -> Substract(subgrad a, subgrad b)
        | Multiply(a, b) -> Add(Multiply(a, subgrad b), Multiply(b, subgrad a))
        | Divide(a, b) -> subgrad (Multiply(a, Power(b, ScalarConst -1.0))) 
        | Power(a, b) -> Add(Multiply(Multiply(Power(a, Substract(b, ScalarConst 1.0)), b), subgrad a),
                             Multiply(Multiply(Power(a, b), Log a), subgrad b))
        | Negate a -> Negate (subgrad a)
        | Log a -> Multiply(Power(a, ScalarConst -1.0), subgrad a)
        | Exp a -> Multiply(Exp a, subgrad a)
        | Dot(a, b) -> 
            let sa, sb = shapeOf a, shapeOf b
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 1, 1 -> subgrad (Sum(Multiply(a, b)))
                | 2, 1 -> 
                    Add(Dot(TensorProduct(b, Identity (ShapeSpec.matrix sa.[0] sa.[0])), // wrt a
                            subgrad a), 
                        Dot(a, subgrad b)) // wrt b
                | 2, 2 when sa.[1] = sb.[0] -> 
                    Add(Dot(TensorProduct(SwapDim(0, 1, b), Identity (ShapeSpec.matrix sa.[0] sa.[0])), // wrt a
                            subgrad a),
                        Dot(TensorProduct(Identity (ShapeSpec.matrix sb.[1] sb.[1]), a), // wrt b
                            subgrad b))
                | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        | TensorProduct(a, b) ->
            let ga, gb = subgrad a, subgrad b
            let sa, sb = shapeOf a, shapeOf b
            let sga, sgb = shapeOf ga, shapeOf gb
            let g = Add(TensorProduct(Reshape(sa @ [sga.[1]], ga), b),
                        TensorProduct(a, Reshape(sb @ [sgb.[1]], gb)))
            let sg = shapeOf g            
            Reshape(sg.[0 .. (ShapeSpec.nDim sg) - 1] @ [sga.[1]], g)            
        | SwapDim (ax1, ax2, a) ->
            let g = subgrad a
            let sg, sa = shapeOf g, shapeOf a
            Reshape(sg, SwapDim(ax1, ax2, Reshape(sa @ [sg.[1]], g)))
        | Reshape (ss, a) ->
            let g = subgrad a
            let sg, sa = shapeOf g, shapeOf a
            Reshape([ShapeSpec.nElem ss; sg.[1]], Reshape(ss @ [sg.[1]], Reshape(sa @ [sg.[1]], g)))
        | Broadcast (ss, a) ->
            let g = subgrad a
            let sg, sa = shapeOf g, shapeOf a
            Reshape([ShapeSpec.nElem ss; sg.[1]], Broadcast(ss @ [sg.[1]], Reshape(sa @ [sg.[1]], g)))
        | Sum a -> 
            let ga = subgrad a
            let sga = shapeOf ga
            Reshape([SizeSpec.one; sga.[1]], SumAxis(0, ga))
        | SumAxis (ax, a) -> 
            let ga = subgrad a
            let sa, sga = shapeOf a, shapeOf ga
            SumAxis(ax, Reshape(sa @ [sga.[1]], ga)) 
        | Zeros ss -> constGrad ss
        | ScalarConst _ -> constGrad ShapeSpec.scalar
        | TensorConst (_, ss) -> constGrad ss
        | Identity ss -> constGrad ss
        | Var v -> 
            let sv = shapeOf (Var v)
            if v = wrt then                 
                Identity [ShapeSpec.nElem sv; ShapeSpec.nElem sv]
            else 
                constGrad sv
        | Annotated(a, ano) -> Annotated(subgrad a, ano)
    Annotated(g, GradOf op) |> checkAndAdaptShapes

