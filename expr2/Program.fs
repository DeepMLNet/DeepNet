open Xunit

open Shape
open Op
open ExprForwardDiff
open ExprReverseDiff
open OpEval
open ExprEvalSequencer


let printExpr label expr =
    printfn "%s :=\n%A\nshape of %s: %A\n" label expr label (shapeOf expr)

let printVal label value =
    printfn "%s =\n%A\nshape of %s: %A\n" label value label (NDArray.shape value)

type LinearRegression = {a: ExprT; b: ExprT; x: ExprT; t: ExprT;
                         Pred: ExprT; Loss: ExprT}  
let linearRegression () =
    let a = var "a" [symbol "M"; symbol "N"]
    let b = var "b" [symbol "M"]
    let x = var "x" [symbol "N"]
    let t = var "t" [symbol "M"]

    let pred = a.*x + b
    let smplLoss = (pred - t)**2.0
    let loss = sum smplLoss

    {a=a; b=b; x=x; t=t; Pred=pred; Loss=loss}

type LinearRegressionGradient = {LossWrtA: ExprT; LossWrtB: ExprT; LossWrtX: ExprT; LossWrtT: ExprT}
let linearRegressionForwardGradient (lr: LinearRegression) =
    {LossWrtA = grad lr.a lr.Loss;
     LossWrtB = grad lr.b lr.Loss;
     LossWrtX = grad lr.x lr.Loss;
     LossWrtT = grad lr.t lr.Loss;}

let linearRegressionReverseGradient (lr: LinearRegression) =
    let d = reverseDiff lr.Loss
    {LossWrtA = diffOf lr.a d;
     LossWrtB = diffOf lr.b d;
     LossWrtX = diffOf lr.x d;
     LossWrtT = diffOf lr.t d;}

let linearRegressionEvalEnv (lr: LinearRegression) =
    let m, n = 3, 2
    let aVal = NDArray.identity [m; n]
    let bVal = NDArray.zeros [m]
    let xVal = NDArray.ones [n]
    let tVal = NDArray.ones [m]
    let varEnv = 
        VarEnv.empty
        |> VarEnv.add lr.a aVal
        |> VarEnv.add lr.b bVal
        |> VarEnv.add lr.x xVal
        |> VarEnv.add lr.t tVal
    EvalEnv.fromVarEnvAndExpr varEnv lr.Loss

[<Fact>]
let ``Build linear regression`` () =
    let lr = linearRegression ()
    printExpr "pred" lr.Pred
    printExpr "loss" lr.Loss

[<Fact>]
let ``Eval linear regression`` () =
    let lr = linearRegression ()
    let env = linearRegressionEvalEnv lr
    printVal "pred" (eval env lr.Pred)
    printVal "loss" (eval env lr.Loss)

[<Fact>]
let ``Forward gradient of linear regression`` () =
    let lr = linearRegression ()   
    printfn "Forward:"
    let fg = linearRegressionForwardGradient lr
    printExpr "lossWrtA" fg.LossWrtA
    printExpr "lossWrtB" fg.LossWrtB
    printExpr "lossWrtX" fg.LossWrtX  
    printExpr "lossWrtT" fg.LossWrtT

[<Fact>]
let ``Reverse gradient of linear regression`` () =
    let lr = linearRegression ()  
    printfn "Reverse:"
    let rg = linearRegressionReverseGradient lr
    printExpr "lossWrtA" rg.LossWrtA
    printExpr "lossWrtB" rg.LossWrtB
    printExpr "lossWrtX" rg.LossWrtX  
    printExpr "lossWrtT" rg.LossWrtT

[<Fact>]
let ``Eval forward gradient of linear regression`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionForwardGradient lr
    let env = linearRegressionEvalEnv lr
    printfn "Forward gradient:"
    printVal "lossWrtA" (eval env lrg.LossWrtA)
    printVal "lossWrtB" (eval env lrg.LossWrtB)
    printVal "lossWrtX" (eval env lrg.LossWrtX) 
    printVal "lossWrtT" (eval env lrg.LossWrtT)

[<Fact>]
let ``Eval reverse gradient of linear regression`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr
    printfn "Reverse gradient:"
    printVal "lossWrtA" (eval env lrg.LossWrtA)
    printVal "lossWrtB" (eval env lrg.LossWrtB)
    printVal "lossWrtX" (eval env lrg.LossWrtX) 
    printVal "lossWrtT" (eval env lrg.LossWrtT)


[<Fact>]
let ``Check forward gradient of linear regression`` () =
    let lr = linearRegression ()
    let env = linearRegressionEvalEnv lr
    printfn "delta lossWrtA = %f" (NumGrad.exprGradDiff env lr.a lr.Loss)
    printfn "delta lossWrtB = %f" (NumGrad.exprGradDiff env lr.b lr.Loss)
    printfn "delta lossWrtX = %f" (NumGrad.exprGradDiff env lr.x lr.Loss)
    printfn "delta lossWrtT = %f" (NumGrad.exprGradDiff env lr.t lr.Loss)

[<Fact>]
let ``Check reverse gradient of linear regression`` () =
    let lr = linearRegression ()
    let env = linearRegressionEvalEnv lr
    DiffCheck.checkReverseDiff env lr.Loss
    printfn "linear regression gradient checked"

let printList execSeq =
    for i, item in List.indexed execSeq do
        printfn "%d. %A" (i+1) item

let printStreams streams =
    for i, stream in List.indexed streams do
        printfn "==============================================="
        printfn "stream %d:" i
        printList stream

[<Fact>]
let ``Build execution sequence of linear regression`` () =
    let lr = linearRegression ()
    let env = linearRegressionEvalEnv lr
    
    let exeSeq, eRes = exprToExecUnits env.SizeSymbolEnv lr.Loss
    //printfn "linear regression exec sequence:\n%A" exeSeq

    let exeStreams = execUnitsToStreamCommands exeSeq
    printfn "linear regression exec streams:"
    printStreams exeStreams

    let cudaCalls = generateCalls exeStreams
    printfn "linear regression CUDA calls:"
    printList cudaCalls


[<Fact>]
let ``Build execution sequence of linear regression gradient`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr

    let exeSeq, eRes = exprToExecUnits env.SizeSymbolEnv lrg.LossWrtA
    //printfn "linear regression wrt A exec sequence:\n%A" exeSeq

    let exeStreams = execUnitsToStreamCommands exeSeq
    printfn "linear regression wrt A exec streams:"
    printStreams exeStreams

    let cudaCalls = generateCalls exeStreams
    printfn "linear regression wrt A CUDA calls:"
    printList cudaCalls


[<EntryPoint>]
let main argv = 
    //CudaCodeGen.testMe ()
    ``Build linear regression`` ()
    //``Eval linear regression`` ()
    //``Reverse gradient of linear regression`` ()
    //``Eval forward gradient of linear regression`` ()
    //``Eval reverse gradient of linear regression`` ()
    //``Check gradient of linear regression`` ()
    //``Check reverse gradient of linear regression`` ()
    ``Build execution sequence of linear regression`` ()
    ``Build execution sequence of linear regression gradient`` ()
    0
