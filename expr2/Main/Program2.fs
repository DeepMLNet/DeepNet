open ArrayNDNS

open ExprNS




[<EntryPoint>]
let main argv = 

    let v1 = Expr.var "v1" (ShapeSpec.vector (SizeSpec.fix 3))
    let v2 = Expr.var "v2" (ShapeSpec.vector (SizeSpec.fix 3))
    
    let v1val : ArrayND.ArrayNDT<float> = ArrayNDHost.ones [3]
    let v2val = ArrayNDHost.scalar 3.0 |> ArrayND.padLeft |> ArrayND.broadcastToShape [3]
    let v3val = -v2val
    let v4val = v1val + v3val

    printfn "v1val: %A" v1val.[[0]]
    printfn "v2val: %A" v2val.[[0]]
    printfn "v3val: %A" v3val.[[0]]
    printfn "v4val: %A" v4val.[[0]]
    
    0
