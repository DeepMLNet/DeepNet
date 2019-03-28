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



type NeuralNetOnMNIST (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let dataDir = Util.assemblyDir + "/TestData/"
    let mnistPath = dataDir + "MNIST/"

    let build device batch = 
        let mc = Context.root device / "NeuralNetModel"

        // symbolic sizes
        let batchSize  = SizeSym "BatchSize"
        let nInput     = SizeSym "nInput"
        let nTarget    = SizeSym "nTarget"

        // model parameters
        let rng = System.Random 123
        let pars = NeuralLayer.pars (mc / "Layer1") rng 
                    {NeuralLayer.HyperPars.standard (Size.sym nInput) (Size.sym nTarget) with ActFunc=ActFunc.SoftMax}
     
        // input / output variables
        let input =  Var<single> (mc / "Input",  [Size.sym batchSize; Size.sym nInput])
        let target = Var<single> (mc / "Target", [Size.sym batchSize; Size.sym nTarget])

        // set strides
        //mc.SetStride input (TensorLayout.cStride [batch; 784L])
        //mc.SetStride target (TensorLayout.cStride [batch; 10L])

        // expressions
        let pred = NeuralLayer.pred pars (Expr input)
        let loss = LossLayer.loss LossLayer.CrossEntropy pred (Expr target)
        printfn "loss is:\n%A" loss

        // instantiate model
        let sizeEnv = Map [
            batchSize, Size.fix batch        
            nInput, Size.fix 784L
            nTarget, Size.fix 10L
        ]
        let ps = ParSet.fromExprs ContextPath.root [pred; loss]
        let mi = ps |> ParSet.inst (ContextPath.root / "Store") sizeEnv 
        let pred = mi.Use pred
        let loss = mi.Use loss

        // optimizer (with parameters)
        let optCfg = {Opt.GradientDescent.Cfg.Step=1e-3}
        let opt = Opt.GradientDescent.make (optCfg, loss, mi)

        // compile functions
        let lossFun = ExprFunc.make loss |> ExprFunc.add mi |> ExprFunc.arg2 input target
        let optFun = ExprFunc.make opt.Step |> ExprFunc.add mi |> ExprFunc.arg2 input target
    
        lossFun, optFun, optCfg

    let getMnist device samples =
        let cut (x: Tensor<_>) =
            match samples with
            | Some samples -> x.[0L .. samples-1L, *]
            | None -> x

        let mnist = Loader.Mnist.loadRaw mnistPath
        let tstImgs =  
            mnist.TstImgs
            |> Tensor.reshape [mnist.TstImgs.Shape.[0]; Remainder]
            |> cut
            |> Tensor<_>.transfer device
        let tstLbls =  
            mnist.TstLbls
            |> cut
            |> Tensor<_>.transfer device
        tstImgs, tstLbls

    let train device samples iters = 
        let tstImgs, tstLbls = getMnist device (Some samples)
        let lossFun, optFun, optCfg = build device samples
        let initialLoss = lossFun tstImgs tstLbls |> Tensor.value
        printfn "Initial loss: %f" initialLoss
        for itr = 0 to iters-1 do
            optFun tstImgs tstLbls |> ignore
            printfn "%d: %f" itr (lossFun tstImgs tstLbls |> Tensor.value)
        let finalLoss = lossFun tstImgs tstLbls |> Tensor.value
        printfn "Final loss: %f" finalLoss
        initialLoss, finalLoss

    [<Fact>]
    let ``MNIST loads`` () =
        let sw = Stopwatch.StartNew()
        getMnist HostTensor.Dev None |> ignore
        printfn "MNIST load time: %A" sw.Elapsed

    [<Fact>]
    let ``MNIST transfers to GPU`` () =
        let sw = Stopwatch.StartNew()
        getMnist CudaTensor.Dev None |> ignore
        printfn "MNIST load time: %A" sw.Elapsed

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Neural net compiles for GPU`` () =
        let sw = Stopwatch.StartNew()
        build CudaTensor.Dev 10000L |> ignore
        printfn "Model build time: %A" sw.Elapsed

    [<Fact>]
    let ``Loss decreases during training on CPU`` () =
        let sw = Stopwatch.StartNew()
        let initialLoss, finalLoss = train HostTensor.Dev 1000L 50
        finalLoss |> should lessThan (initialLoss - 0.001f)
        printfn "Model build and train time: %A" sw.Elapsed

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Loss decreases during training on GPU`` () =
        let sw = Stopwatch.StartNew()
        let initialLoss, finalLoss = train CudaTensor.Dev 1000L 50
        finalLoss |> should lessThan (initialLoss - 0.001f)
        printfn "Model build and train time: %A" sw.Elapsed

    //[<Fact>]
    //[<Trait("Category", "Skip_CI")>]
    //let ``CPU and GPU have same trace during training`` () =
    //    let diffs = compareTraces (fun dev -> train dev 10L 1 |> ignore) false
    //    diffs |> should equal 0

    