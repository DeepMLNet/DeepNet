module NeuralNetOnMNIST

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


let mnistPath = Util.assemblyDirectory + "../../../../Data/MNIST"


let build device batch = 
    let mc = ModelBuilder<single> "NeuralNetModel"

    // symbolic sizes
    let batchSize  = mc.Size "BatchSize"
    let nInput     = mc.Size "nInput"
    let nTarget    = mc.Size "nTarget"

    // model parameters
    let pars = NeuralLayer.pars (mc.Module "Layer1") 
                {NeuralLayer.defaultHyperPars with
                  NInput=nInput; NOutput=nTarget; TransferFunc=NeuralLayer.Tanh}
     
    // input / output variables
    let input =  mc.Var "Input"  [batchSize; nInput]
    let target = mc.Var "Target" [batchSize; nTarget]

    // set sizes
    mc.SetSize batchSize batch
    mc.SetSize nInput 784
    mc.SetSize nTarget 10

    // instantiate model
    let mi = mc.Instantiate (device, canDelay=false)

    // expressions
    let pred = NeuralLayer.pred pars input
    let loss = LossLayer.loss LossLayer.MSE pred target
    printfn "loss is:\n%A" loss

    // optimizer (with parameters)
    let opt = GradientDescent<single> (loss, mi.ParameterVector, device)
    let optCfg = {GradientDescent.Step=1e-3f}
    opt.PublishLoc mi

    // compile functions
    let lossFun = mi.Func loss |> arg2<single, single, _> input target
    let optFun = mi.Func (opt.Minimize) |> opt.Use |> arg2<single, single, _> input target
    
    lossFun, optFun, optCfg, opt.InitialState optCfg mi.ParameterValues

let getMnist device samples =
    let cut (x: ArrayNDT<_>) =
        match samples with
        | Some samples -> x.[0..samples-1, *]
        | None -> x

    let mnist = Mnist.loadRaw mnistPath
    let tstImgs =  
        mnist.TstImgs
        |> ArrayND.reshape [mnist.TstImgs.Shape.[0]; -1]
        |> cut
        |> post device
    let tstLbls =  
        mnist.TstLbls
        |> cut
        |> post device
    tstImgs, tstLbls

let train device samples iters = 
    let tstImgs, tstLbls = getMnist device (Some samples)
    let lossFun, optFun, optCfg, optState = build device samples
    let initialLoss = lossFun tstImgs tstLbls |> ArrayND.value
    printfn "Initial loss: %f" initialLoss
    for itr = 0 to iters-1 do
        optFun tstImgs tstLbls optCfg optState |> ignore
        //printfn "%d: %f" itr (lossFun tstImgs tstLbls |> ArrayND.value)
    let finalLoss = lossFun tstImgs tstLbls |> ArrayND.value
    printfn "Final loss: %f" finalLoss
    initialLoss, finalLoss

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``MNIST loads`` () =
    let sw = Stopwatch.StartNew()
    getMnist DevCuda None |> ignore
    printfn "MNIST load time: %A" sw.Elapsed

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Neural net compiles for GPU`` () =
    let sw = Stopwatch.StartNew()
    build DevCuda 10000 |> ignore
    printfn "Model build time: %A" sw.Elapsed

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Loss decreases during training on GPU`` () =
    let sw = Stopwatch.StartNew()
    let initialLoss, finalLoss = train DevCuda 1000 50
    finalLoss |> should lessThan (initialLoss - 0.001f)
    printfn "Model build and train time: %A" sw.Elapsed

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``CPU and GPU have same trace during training`` () =
    let diffs = compareTraces (fun dev -> train dev 10 1 |> ignore) false
    diffs |> should equal 0

    