module ModelTest

open System.IO
open System.Diagnostics

open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers
open TestUtils


let ``Demo neural net`` device =
    let sampleLimit = None
    let iters = 10

    // MNIST dataset
    let tstImgs, tstLbls = NeuralNetOnMNIST.getMnist device sampleLimit

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
    let dLoss = mi.WrtParameters loss

    //printfn "loss:\n%A" loss
    //printfn "dLoss:\n%A" dLoss

    // compile functions
    let lossFun = mi.Func (loss) |> arg2 input target
    let dLossFun = mi.Func (dLoss) |> arg2 input target

    // calculate test loss on MNIST
    let tstLoss = lossFun tstImgs tstLbls
    printfn "Test loss on MNIST=%A" tstLoss

    let opt = GradientDescent.minimize {Step=1e-3f} loss mi.ParameterVector
    let optFun = mi.Func opt |> arg2 input target
    
    printfn "Optimizing..."
    
    for itr = 0 to iters do
        optFun tstImgs tstLbls |> ignore
        let l = lossFun tstImgs tstLbls
        printfn "Loss afer %d iterations: %A" itr l

    ()

     

[<EntryPoint>]
let main argv = 
    Basics.Cuda.CudaSup.init ()
    Basics.Cuda.CudaSup.printInfo()
   
    ``Demo neural net`` DevCuda

    Basics.Cuda.CudaSup.shutdown ()
    0

