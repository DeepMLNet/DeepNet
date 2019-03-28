namespace global

open System.Diagnostics
open System.IO
open Xunit
open FsUnit.Xunit
open Xunit.Abstractions

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.ML
open TestUtils


type LinearRegression (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let dataDir = Util.assemblyDir + "/TestData/"
    let mnistPath = dataDir + "MNIST/"
    let batch = 100L

    let buildLossExpr device batch = 
        let mc = Context.root device / "LinReg"

        // symbolic sizes
        let batchSize  = SizeSym "BatchSize"
        let nInput     = SizeSym "nInput"
        let nTarget    = SizeSym "nTarget"

        // model parameters
        let rng = System.Random 123
        let pars = 
            LinearRegression.pars (mc / "Layer1") rng 
                (LinearRegression.HyperPars<single>.standard (Size.sym nInput) (Size.sym nTarget))
     
        // input / output variables
        let input =  Var<single> (mc / "Input",  [Size.sym batchSize; Size.sym nInput])
        let target = Var<single> (mc / "Target", [Size.sym batchSize; Size.sym nTarget])

        // expressions
        let loss = LinearRegression.loss pars (Expr input) (Expr target)

        loss, pars, input, target, batchSize, nInput, nTarget

    let buildParSetInst device batch =
        let loss, pars, input, target, batchSize, nInput, nTarget = buildLossExpr device batch

        // instantiate model
        let sizeEnv = Map [
            batchSize, Size.fix batch        
            nInput, Size.fix 5L
            nTarget, Size.fix 10L
        ]
        let ps = ParSet.fromExprs ContextPath.root [loss]
        let mi = ps |> ParSet.inst (ContextPath.root / "Store") sizeEnv 
        let loss = mi.Use loss

        let rng = System.Random 123
        let inputVal = HostTensor.randomNormal rng (0.0f, 1.0f) [batch; 5L]
        let targetVal = HostTensor.randomNormal rng (1.0f, 0.5f) [batch; 10L]
        let varEnv = VarEnv.ofSeq [
            input, Tensor.transfer device inputVal
            target, Tensor.transfer device targetVal
        ]
        let varEnv = mi.Use varEnv

        loss, input, target, ps, mi, varEnv

    let buildOptFn device batch =
        let loss, input, target, ps, mi, varEnv = buildParSetInst device batch

        // optimizer (with parameters)
        let optCfg = {Opt.GradientDescent.Cfg.Step=1e-3}
        let opt = Opt.GradientDescent.make (optCfg, loss, mi)

        loss, opt, varEnv

    [<Fact>]
    let ``Loss expression`` () =
        runOnAllDevs output (fun ctx ->
            let loss, pars, input, target, batchSize, nInput, nTarget = buildLossExpr ctx.Dev batch
            printfn "Loss:\n%s" (loss.ToString())
        )
    
    [<Fact>]
    let ``Loss derivatives`` () =
        runOnAllDevs output (fun ctx ->
            let loss, pars, input, target, batchSize, nInput, nTarget = buildLossExpr ctx.Dev batch
            let d = Deriv.compute loss
            printfn "Deriv wrt. input:\n%s" ((d.Wrt input).ToString())
            printfn "Deriv wrt. target:\n%s" ((d.Wrt target).ToString())
            printfn "Deriv wrt. weights:\n%s" ((d.Wrt pars.Weights).ToString())
        )

    [<Fact>]
    let ``Build ParSet instance`` () =
        runOnAllDevs output (fun ctx ->
            let loss, input, target, ps, mi, varEnv = buildParSetInst ctx.Dev batch            
            printfn "Loss: %A" loss
            printfn "ParSet: %A" ps
            printfn "ParSetInst: %A" mi
        )

    [<Fact>]
    let ``Evaluate loss with random data`` () =
        runOnAllDevs output (fun ctx ->
            let loss, input, target, ps, mi, varEnv = buildParSetInst ctx.Dev batch            

            printfn "Filling parameters with ones..."
            for KeyValue(_, pi) in mi.ParInsts do
                pi.Data.FillOnes()

            printfn "VarEnv:\n%A" varEnv

            let lossVal = loss |> Expr.eval varEnv
            printfn "Loss value: %A" lossVal

            let d = Deriv.compute loss
            printfn "Deriv wrt. input:\n%s" ((d.Wrt input |> mi.Use |> Expr.eval varEnv).Full)
            printfn "Deriv wrt. target:\n%s" ((d.Wrt target |> mi.Use |> Expr.eval varEnv).Full)
        )

    [<Fact>]
    let ``Perform 100 optimization steps`` () =
        runOnAllDevs output (fun ctx ->
            let loss, opt, varEnv = buildOptFn ctx.Dev batch 
            let step = opt.Step |> EvalUpdateBundle.addExpr loss

            let initial = loss |> Expr.eval varEnv |> Tensor.value
            for i in 1..100 do
                let res = step |> EvalUpdateBundle.exec varEnv
                printfn "%d: loss=%f" i (res.Get loss |> Tensor.value)
            let final = loss |> Expr.eval varEnv |> Tensor.value

            assert (final <= initial * 0.99f)
        )