(*** hide ***)
#load "../../DeepNet.fsx"

(**
Learning MNIST
==============

In this example we will show how to learn MNIST classification using a two-layer feed-forward network.

*)
open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers

(**
First we write a function that loads MNIST.
*)

let mnistPath = __SOURCE_DIRECTORY__ + "../../Data/MNIST"

let mnist = Mnist.load mnistPath
let trnImgs =
    mnist.TrnImgs
    |> ArrayND.reorderAxes [2; 0; 1]
    |> ArrayND.reshape [-1; (ArrayND.shape mnist.TrnImgs).[0]]
    |> ArrayNDCuda.toDev
let trnLbls =
    mnist.TrnLbls
    |> ArrayND.reorderAxes [1; 0]
    |> ArrayNDCuda.toDev
let tstImgs =
    mnist.TstImgs
    |> ArrayND.reorderAxes [2; 0; 1]
    |> ArrayND.reshape [-1; (ArrayND.shape mnist.TstImgs).[0]]
    |> ArrayNDCuda.toDev
let tstLbls =
    mnist.TstLbls
    |> ArrayND.reorderAxes [1; 0]
    |> ArrayNDCuda.toDev

(**
Then we execute it.
*)

let sampleLimit = None
let iters = 10

let mc = ModelBuilder<single> "NeuralNetModel"

// symbolic sizes
let batchSize  = mc.Size "BatchSize"
let nInput     = mc.Size "nInput"
let nTarget    = mc.Size "nTarget"

// model parameters
let hpars = { NeuralLayer.NInput=nInput; NeuralLayer.NOutput=nTarget; NeuralLayer.TransferFunc=NeuralLayer.Tanh }
let pars = NeuralLayer.pars (mc.Module "Layer1") hpars

// input / output variables
let input =  mc.Var "Input"  [nInput;  batchSize]
let target = mc.Var "Target" [nTarget; batchSize]

// infer sizes and variable locations from dataset
mc.UseTmplVal input tstImgs
mc.UseTmplVal target tstLbls
printfn "inferred sizes: %A" mc.SymSizeEnv
printfn "inferred locations: %A" mc.VarLocs

// instantiate model
let mi = mc.Instantiate DevCuda

// expressions
let pred = NeuralLayer.pred pars input
let loss = LossLayer.loss LossLayer.MSE pred target

// compile functions
let opt = GradientDescent (loss, mi.ParameterVector, DevCuda)
let optFun = mi.Func opt.Minimize |> opt.Use |> arg2 input target
let lossFun = mi.Func (loss) |> arg2 input target

// calculate test loss on MNIST
let tstLoss = lossFun tstImgs tstLbls
printfn "Test loss on MNIST=%A" tstLoss

//opt.PublishLoc mi

printfn "Optimizing..."

for itr = 0 to iters do
    optFun trnImgs trnLbls |> ignore
    let l = lossFun tstImgs tstLbls
    printfn "Loss afer %d iterations: %A" itr l


(**
Done.
*)
