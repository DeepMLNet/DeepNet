open Xunit

open Basics.Cuda
open ArrayNDNS
open SymTensor
open SymTensor.Compiler
open SymTensor.Compiler.Cuda



let printExpr label expr =
    printfn "%s :=\n%A\nshape of %s: %A\n" label expr label (Expr.shapeOf expr)

let printVal label value =
    printfn "%s =\n%A\nshape of %s: %A\n" label value label (ArrayND.shape value)

type ExprTs = ExprT<single>

type LinearRegression = {a: ExprTs; b: ExprTs; x: ExprTs; t: ExprTs;
                         predOut: ExprTs; lossOut: ExprTs;
                         lossWrtAOut: ExprTs; lossWrtBOut: ExprTs; lossWrtXOut: ExprTs; lossWrtTOut: ExprTs;
                         Pred: ExprTs; Loss: ExprTs}  
let linearRegression () =
    let a : ExprTs = Expr.var "a" [SizeSpec.symbol "M"; SizeSpec.symbol "N"]
    let b : ExprTs = Expr.var "b" [SizeSpec.symbol "M"]
    let x : ExprTs = Expr.var "x" [SizeSpec.symbol "N"]
    let t : ExprTs = Expr.var "t" [SizeSpec.symbol "M"]

    let predOut = Expr.var "predOut" [SizeSpec.symbol "M"]
    let lossOut = Expr.var "lossOut" []
    let lossWrtAOut = Expr.var "lossWrtAOut" [SizeSpec.fix 1; (SizeSpec.symbol "M") * (SizeSpec.symbol "N")]
    let lossWrtBOut = Expr.var "lossWrtBOut" [SizeSpec.fix 1; SizeSpec.symbol "M"]
    let lossWrtXOut = Expr.var "lossWrtXOut" [SizeSpec.fix 1; SizeSpec.symbol "N"]
    let lossWrtTOut = Expr.var "lossWrtTOut" [SizeSpec.fix 1; SizeSpec.symbol "M"]

    let pred = a.*x + b
    let smplLoss = (pred - t)**2.0f
    let loss = Expr.sum smplLoss

    {a=a; b=b; x=x; t=t; Pred=pred; Loss=loss;
     predOut=predOut; lossOut=lossOut;
     lossWrtAOut=lossWrtAOut; lossWrtBOut=lossWrtBOut; lossWrtXOut=lossWrtXOut; lossWrtTOut=lossWrtTOut}

type LinearRegressionGradient = {LossWrtA: ExprTs; LossWrtB: ExprTs; LossWrtX: ExprTs; LossWrtT: ExprTs}

let linearRegressionReverseGradient (lr: LinearRegression) =
    let d = Deriv.compute lr.Loss
    {LossWrtA = Deriv.ofVar lr.a d;
     LossWrtB = Deriv.ofVar lr.b d;
     LossWrtX = Deriv.ofVar lr.x d;
     LossWrtT = Deriv.ofVar lr.t d;}

let linearRegressionEvalEnv (lr: LinearRegression) =
    let m, n = 3, 2
    let aVal = ArrayNDHost.ones [m; n]
    let bVal = ArrayNDHost.zeros [m]
    let xVal = ArrayNDHost.ones [n]
    let tVal = ArrayNDHost.ones [m]
    let predOutVal = ArrayNDHost.zeros [m]
    let lossOutVal = ArrayNDHost.zeros []
    let lossWrtAVal = ArrayNDHost.zeros [1; m*n]
    let lossWrtBVal = ArrayNDHost.zeros [1; m]
    let lossWrtXVal = ArrayNDHost.zeros [1; n]
    let lossWrtTVal = ArrayNDHost.zeros [1; m]
    let varEnv = 
        VarEnv.empty
        |> VarEnv.add lr.a aVal
        |> VarEnv.add lr.b bVal
        |> VarEnv.add lr.x xVal
        |> VarEnv.add lr.t tVal
        |> VarEnv.add lr.predOut predOutVal
        |> VarEnv.add lr.lossOut lossOutVal
        |> VarEnv.add lr.lossWrtAOut lossWrtAVal
        |> VarEnv.add lr.lossWrtBOut lossWrtBVal
        |> VarEnv.add lr.lossWrtXOut lossWrtXVal
        |> VarEnv.add lr.lossWrtTOut lossWrtTVal
    EvalEnv.create varEnv (Seq.singleton lr.Loss)

let linearRegressionCompileEnv (lr: LinearRegression) =
    let varLocs =
        [lr.a |> Expr.extractVar :> IVarSpec, LocHost;
         lr.b |> Expr.extractVar :> IVarSpec, LocHost;
         lr.x |> Expr.extractVar :> IVarSpec, LocHost;
         lr.t |> Expr.extractVar :> IVarSpec, LocHost;
         lr.predOut |> Expr.extractVar :> IVarSpec, LocHost;
         lr.lossOut |> Expr.extractVar :> IVarSpec, LocHost;
         lr.lossWrtAOut |> Expr.extractVar :> IVarSpec, LocHost;
         lr.lossWrtBOut |> Expr.extractVar :> IVarSpec, LocHost;
         lr.lossWrtXOut |> Expr.extractVar :> IVarSpec, LocHost;
         lr.lossWrtTOut |> Expr.extractVar :> IVarSpec, LocHost;]
        |> Map.ofList
    {CudaCompileEnvT.VarStorLoc = varLocs}


[<Fact>]
let ``Build linear regression`` () =
    let lr = linearRegression ()
    printExpr "pred" lr.Pred
    printExpr "loss" lr.Loss

[<Fact>]
let ``Eval linear regression`` () =
    let lr = linearRegression ()
    let env = linearRegressionEvalEnv lr
    printVal "pred" (Eval.evalWithEvalEnv env lr.Pred)
    printVal "loss" (Eval.evalWithEvalEnv env lr.Loss)

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
let ``Eval reverse gradient of linear regression`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr
    printfn "Reverse gradient:"
    printVal "lossWrtA" (Eval.evalWithEvalEnv env lrg.LossWrtA)
    printVal "lossWrtB" (Eval.evalWithEvalEnv env lrg.LossWrtB)
    printVal "lossWrtX" (Eval.evalWithEvalEnv env lrg.LossWrtX) 
    printVal "lossWrtT" (Eval.evalWithEvalEnv env lrg.LossWrtT)

[<Fact>]
let ``Check reverse gradient of linear regression`` () =
    let lr = linearRegression ()
    let env = linearRegressionEvalEnv lr
    DerivCheck.checkReverseDiff env lr.Loss
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
    let cenv = linearRegressionCompileEnv lr
    
    let exeSeq, eRes, memAllocs = CudaExecUnit.exprToCudaExecUnits cenv env.SizeSymbolEnv (UExpr.toUExpr lr.Loss)
    printfn "linear regression exec sequence:\n%A" exeSeq

    let exeStreams, strmCnt = CudaStreamSeq.execUnitsToStreams exeSeq
    printfn "linear regression exec streams:"
    printStreams exeStreams

    let cudaCalls, krnlCache = CudaRecipe.generateCalls exeStreams
    printfn "linear regression CUDA calls:"
    printList cudaCalls


[<Fact>]
let ``Build execution sequence of linear regression gradient`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr
    let cenv = linearRegressionCompileEnv lr

    let exeSeq, eRes, memAllocs = CudaExecUnit.exprToCudaExecUnits cenv env.SizeSymbolEnv (UExpr.toUExpr lrg.LossWrtA)
    //printfn "linear regression wrt A exec sequence:\n%A" exeSeq

    let exeStreams, strmCnt = CudaStreamSeq.execUnitsToStreams exeSeq
    printfn "linear regression wrt A exec streams:"
    printStreams exeStreams

    let cudaCalls, krnlCache = CudaRecipe.generateCalls exeStreams
    printfn "linear regression wrt A CUDA calls:"
    printList cudaCalls


[<Fact>]
let ``Build CUDA recipe for linear regression gradient`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr
    let cenv = linearRegressionCompileEnv lr

    let recipe = CudaRecipe.build cenv env.SizeSymbolEnv (UExpr.toUExpr lrg.LossWrtA)
    printfn "%A" recipe

    ()


let ``Evaluate linear regression using CUDA`` () =
    let lr = linearRegression ()
    let env = linearRegressionEvalEnv lr
    let cenv = linearRegressionCompileEnv lr

    let allWrtsSaved = 
        Expr.discard [lr.Pred |> Expr.storeToVar lr.predOut;
                      lr.Loss |> Expr.storeToVar lr.lossOut;]

    //printfn "%A" allWrtsSaved

    let recipe = CudaRecipe.build cenv env.SizeSymbolEnv (UExpr.toUExpr allWrtsSaved)
    use cudaExpr = new CudaExprWorkspace(recipe)
    use lockedVarEnv = new VarEnvReg(env.VarEnv)

    cudaExpr.Eval(Map.empty, env.VarEnv)

    printVal "pred" (VarEnv.get lr.predOut env.VarEnv)
    printVal "loss" (VarEnv.get lr.lossOut env.VarEnv)


[<Fact>]
let ``Evaluate linear regression gradient using CUDA`` () =
    let lr = linearRegression ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr
    let cenv = linearRegressionCompileEnv lr

    let allWrtsSaved = 
        Expr.discard [lrg.LossWrtA |> Expr.storeToVar lr.lossWrtAOut;
                      lrg.LossWrtB |> Expr.storeToVar lr.lossWrtBOut;
                      lrg.LossWrtX |> Expr.storeToVar lr.lossWrtXOut;
                      lrg.LossWrtT |> Expr.storeToVar lr.lossWrtTOut]

    //printfn "%A" allWrtsSaved

    let recipe = CudaRecipe.build cenv env.SizeSymbolEnv (UExpr.toUExpr allWrtsSaved)
    use cudaExpr = new CudaExprWorkspace(recipe)
    use lockedVarEnv = new VarEnvReg(env.VarEnv)

    cudaExpr.Eval(Map.empty, env.VarEnv)

    printVal "lossWrtA" (VarEnv.get lr.lossWrtAOut env.VarEnv)
    printVal "lossWrtB" (VarEnv.get lr.lossWrtBOut env.VarEnv)
    printVal "lossWrtX" (VarEnv.get lr.lossWrtXOut env.VarEnv)
    printVal "lossWrtT" (VarEnv.get lr.lossWrtTOut env.VarEnv)




[<EntryPoint>]
let main argv = 
    CudaSup.printInfo ()

    //``Build linear regression`` ()
    //``Reverse gradient of linear regression`` ()
    //``Check reverse gradient of linear regression`` ()
    //``Build execution sequence of linear regression`` ()
    //``Build execution sequence of linear regression gradient`` ()
    //``Build CUDA recipe for linear regression gradient`` ()
    
    ``Eval linear regression`` ()
    ``Evaluate linear regression using CUDA`` ()

    ``Eval reverse gradient of linear regression`` ()
    ``Evaluate linear regression gradient using CUDA`` ()
    
    
    CudaSup.shutdown ()
    0
