module CompilerPipeline

open System.Diagnostics
open System.IO
open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers
open TestUtils



let printExpr label expr =
    printfn "%s :=\n%A\nshape of %s: %A\n" label expr label (Expr.shapeOf expr)

let printVal label value =
    printfn "%s =\n%A\nshape of %s: %A\n" label value label (ArrayND.shape value)

let printList execSeq =
    for i, item in List.indexed execSeq do
        printfn "%d. %A" (i+1) item

let printStreams streams =
    for i, stream in List.indexed streams do
        printfn "==============================================="
        printfn "stream %d:" i
        printList stream


type LinearRegression<'T> = {
    a: ExprT<'T>; 
    b: ExprT<'T>; 
    x: ExprT<'T>; 
    t: ExprT<'T>;
    predOut: ExprT<'T>; 
    lossOut: ExprT<'T>;
    lossWrtAOut: ExprT<'T>; 
    lossWrtBOut: ExprT<'T>; 
    lossWrtXOut: ExprT<'T>; 
    lossWrtTOut: ExprT<'T>;
    Pred: ExprT<'T>; 
    Loss: ExprT<'T>;
}  

let inline linearRegression<'T> () =
    let a : ExprT<'T> = Expr.var "a" [SizeSpec.symbol "M"; SizeSpec.symbol "N"]
    let b : ExprT<'T> = Expr.var "b" [SizeSpec.symbol "M"]
    let x : ExprT<'T> = Expr.var "x" [SizeSpec.symbol "N"]
    let t : ExprT<'T> = Expr.var "t" [SizeSpec.symbol "M"]

    let predOut = Expr.var "predOut" [SizeSpec.symbol "M"]
    let lossOut = Expr.var "lossOut" []
    let lossWrtAOut = Expr.var "lossWrtAOut" [SizeSpec.fix 1; (SizeSpec.symbol "M") * (SizeSpec.symbol "N")]
    let lossWrtBOut = Expr.var "lossWrtBOut" [SizeSpec.fix 1; SizeSpec.symbol "M"]
    let lossWrtXOut = Expr.var "lossWrtXOut" [SizeSpec.fix 1; SizeSpec.symbol "N"]
    let lossWrtTOut = Expr.var "lossWrtTOut" [SizeSpec.fix 1; SizeSpec.symbol "M"]

    let pred = a.*x + b
    let smplLoss = (pred - t) ** (Expr.scalar (conv<'T> 2.0))
    let loss = Expr.sum smplLoss

    {a=a; b=b; x=x; t=t; Pred=pred; Loss=loss;
     predOut=predOut; lossOut=lossOut;
     lossWrtAOut=lossWrtAOut; lossWrtBOut=lossWrtBOut; lossWrtXOut=lossWrtXOut; lossWrtTOut=lossWrtTOut}

type LinearRegressionGradient<'T> = {
    LossWrtA: ExprT<'T>; 
    LossWrtB: ExprT<'T>; 
    LossWrtX: ExprT<'T>; 
    LossWrtT: ExprT<'T>;
}

let linearRegressionReverseGradient (lr: LinearRegression<'T>) =
    let d = Deriv.compute lr.Loss
    {LossWrtA = Deriv.ofVar lr.a d;
     LossWrtB = Deriv.ofVar lr.b d;
     LossWrtX = Deriv.ofVar lr.x d;
     LossWrtT = Deriv.ofVar lr.t d;}

let linearRegressionEvalEnv (lr: LinearRegression<'T>) =
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
    EvalEnv.create varEnv

let linearRegressionCompileEnv (lr: LinearRegression<'T>) =
    let varLocs =
        [lr.a |> UVarSpec.ofExpr, LocHost;
         lr.b |> UVarSpec.ofExpr, LocHost;
         lr.x |> UVarSpec.ofExpr, LocHost;
         lr.t |> UVarSpec.ofExpr, LocHost;
         lr.predOut |> UVarSpec.ofExpr, LocHost;
         lr.lossOut |> UVarSpec.ofExpr, LocHost;
         lr.lossWrtAOut |> UVarSpec.ofExpr, LocHost;
         lr.lossWrtBOut |> UVarSpec.ofExpr, LocHost;
         lr.lossWrtXOut |> UVarSpec.ofExpr, LocHost;
         lr.lossWrtTOut |> UVarSpec.ofExpr, LocHost;]
        |> Map.ofList
    {CudaCompileEnvT.VarStorLoc = varLocs}




[<Fact>]
let ``Build linear regression`` () =
    let lr = linearRegression<single> ()
    printExpr "pred" lr.Pred
    printExpr "loss" lr.Loss

[<Fact>]
let ``Linear regression symbol size inference`` () =
    let lr = linearRegression<single> ()
    let env = linearRegressionEvalEnv lr

    printfn "Inferred symbol sizes with variable values:"
    let vsenv = VarEnv.inferSymSizes env.VarEnv
    SymSizeEnv.dump vsenv


[<Fact>]
let ``Eval linear regression`` () =
    let lr = linearRegression<single> ()
    let env = linearRegressionEvalEnv lr
    printfn "CPU: pred=\n%A" (HostEval.eval env lr.Pred)
    printfn "CPU: loss=\n%A" (HostEval.eval env lr.Loss)


[<Fact>]
let ``Reverse gradient of linear regression`` () =
    let lr = linearRegression<double> ()  
    printfn "Reverse:"
    let rg = linearRegressionReverseGradient lr
    printExpr "lossWrtA" rg.LossWrtA
    printExpr "lossWrtB" rg.LossWrtB
    printExpr "lossWrtX" rg.LossWrtX  
    printExpr "lossWrtT" rg.LossWrtT

[<Fact>]
let ``Eval reverse gradient of linear regression`` () =
    let lr = linearRegression<double> ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr
    printfn "Reverse gradient:"
    printfn "CPU: lossWrtA=\n%A" (HostEval.eval env lrg.LossWrtA)
    printfn "CPU: lossWrtB=\n%A" (HostEval.eval env lrg.LossWrtB)
    printfn "CPU: lossWrtX=\n%A" (HostEval.eval env lrg.LossWrtX) 
    printfn "CPU: lossWrtT=\n%A" (HostEval.eval env lrg.LossWrtT)

[<Fact>]
let ``Check reverse gradient of linear regression`` () =
    let lr = linearRegression<double> ()
    let env = linearRegressionEvalEnv lr
    DerivCheck.checkReverseDiff env lr.Loss
    printfn "linear regression gradient checked"



[<Fact>]
let ``Build execution sequence of linear regression`` () =
    let lr = linearRegression<single> ()
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
    let lr = linearRegression<single> ()
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
    let lr = linearRegression<single> ()
    let lrg = linearRegressionReverseGradient lr
    let env = linearRegressionEvalEnv lr
    let cenv = linearRegressionCompileEnv lr

    let recipe = CudaRecipe.build cenv env.SizeSymbolEnv (UExpr.toUExpr lrg.LossWrtA)
    printfn "%A" recipe

    ()


let ``Evaluate linear regression using CUDA`` () =
    let lr = linearRegression<single> ()
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

    printfn "GPU: pred=\n%A" (VarEnv.get lr.predOut env.VarEnv)
    printfn "GPU: loss=\n%A" (VarEnv.get lr.lossOut env.VarEnv)


[<Fact>]
let ``Evaluate linear regression gradient using CUDA`` () =
    let lr = linearRegression<single> ()
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

    printfn "GPU: lossWrtA =\n%A" (VarEnv.get lr.lossWrtAOut env.VarEnv)
    printfn "GPU: lossWrtB =\n%A" (VarEnv.get lr.lossWrtBOut env.VarEnv)
    printfn "GPU: lossWrtX =\n%A" (VarEnv.get lr.lossWrtXOut env.VarEnv)
    printfn "GPU: lossWrtT =\n%A" (VarEnv.get lr.lossWrtTOut env.VarEnv)




