open Xunit

open Shape
open Op
open OpGrad
open OpEval


let printExpr label expr =
    printfn "%s :=\n%A\nshape of %s: %A\n" label expr label (shapeOf expr)

let printVal label value =
    printfn "%s =\n%A\nshape of %s: %A\n" label value label (NDArray.shape value)

type LinearRegression = {a: Expr; b: Expr; x: Expr; t: Expr;
                         Pred: Expr; Loss: Expr}  
let linearRegression () =
    let a = var "a" [symbol "M"; symbol "N"]
    let b = var "b" [symbol "M"]
    let x = var "x" [symbol "N"]
    let t = var "t" [symbol "M"]

    let pred = a.*x + b
    let smplLoss = (pred - t)**2.0
    let loss = sum smplLoss

    {a=a; b=b; x=x; t=t; Pred=pred; Loss=loss}

type LinearRegressionGradient = {LossWrtA: Expr; LossWrtB: Expr; LossWrtX: Expr; LossWrtT: Expr}
let linearRegressionGradient (lr: LinearRegression) =
    {LossWrtA = grad (extractVar lr.a) lr.Loss;
     LossWrtB = grad (extractVar lr.b) lr.Loss;
     LossWrtX = grad (extractVar lr.x) lr.Loss;
     LossWrtT = grad (extractVar lr.t) lr.Loss;}

let linearRegressionValueEnv (lr: LinearRegression) =
    let m, n = 5, 3
    let aVal = NDArray.zeros [m; n]
    let bVal = NDArray.zeros [m]
    let xVal = NDArray.zeros [n]
    let tVal = NDArray.zeros [m]
    let env = 
        VarEnv.empty
        |> VarEnv.add lr.a aVal
        |> VarEnv.add lr.b bVal
        |> VarEnv.add lr.x xVal
        |> VarEnv.add lr.t tVal
    env

[<Fact>]
let ``Build linear regression`` () =
    let lr = linearRegression ()
    printExpr "pred" lr.Pred
    printExpr "loss" lr.Loss

[<Fact>]
let ``Eval linear regression`` () =
    let lr = linearRegression ()
    let env = linearRegressionValueEnv lr
    printVal "pred" (eval env lr.Pred)
    printVal "loss" (eval env lr.Loss)

[<Fact>]
let ``Gradient of linear regression`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionGradient lr
    printExpr "lossWrtA" lrg.LossWrtA
    printExpr "lossWrtB" lrg.LossWrtB
    printExpr "lossWrtX" lrg.LossWrtX  
    printExpr "lossWrtT" lrg.LossWrtT

[<Fact>]
let ``Eval gradient of linear regression`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionGradient lr
    let env = linearRegressionValueEnv lr
    printVal "lossWrtA" (eval env lrg.LossWrtA)
    printVal "lossWrtB" (eval env lrg.LossWrtB)
    printVal "lossWrtX" (eval env lrg.LossWrtX) 
    printVal "lossWrtT" (eval env lrg.LossWrtT)
       

[<EntryPoint>]
let main argv = 
    ``Build linear regression`` ()
    ``Eval linear regression`` ()
    //``Gradient of linear regression`` ()
    ``Eval gradient of linear regression`` ()
    0
