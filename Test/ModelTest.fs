module ModelTest

open Basics
open ArrayNDNS

open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets


let mnist = Mnist.load @"..\..\..\Data\MNIST"


let ``Test neural net`` () =
    let mc = ModelBuilder<single> "NeuralNetModel"
    
    // symbolic sizes
    let batchSize  = mc.Size "BatchSize"
    let nInput     = mc.Size "nInput"
    let nTarget    = mc.Size "nTarget"

    // model parameters
    let pars = NeuralLayer.pars (mc.Module "Layer1") nInput nTarget
    
    // input / output variables
    let input =  mc.Var "Input"  [nInput;  batchSize]
    let target = mc.Var "Target" [nTarget; batchSize]

    let mc = mc.ParametersComplete ()

    // expressions
    let loss = NeuralLayer.loss pars input target |> mc.Subst
    //let dLoss = mc.WrtParameters loss

    //let loss = Optimizer.optimize loss
    //let dLoss = Optimizer.optimize dLoss

    //printfn "loss:\n%A" loss
    //printfn "dLoss:\n%A" dLoss

    // MNIST dataset
    let tstImgs =  
        mnist.TstImgs
        |> ArrayND.reorderAxes [2; 0; 1] 
        |> ArrayND.reshape [-1; (ArrayND.shape mnist.TstImgs).[0]]
    let tstLbls =  
        mnist.TstLbls
        |> ArrayND.reorderAxes [1; 0] 

    // infer sizes and variable locations from dataset
    mc.UseTmplVal input tstImgs     
    mc.UseTmplVal target tstLbls

    printfn "inferred sizes: %A" mc.SymSizeEnv
    printfn "inferred locations: %A" mc.VarLocs

    // instantiate model
    let mi = mc.Instantiate DevHost

    // compile functions
    let lossFun = mi.Func (loss) |> arg2 input target
    //let dLossFun = mi.Func (dLoss) |> arg2 input target

    // calcualte test loss on MNIST
    let tstLoss = lossFun tstImgs tstLbls
    printfn "Test loss on MNIST=%A" tstLoss

    let opt = Optimizers.gradientDescent {Step=1e-3f} loss mc.ParameterSet.Flat
    
    let optFun = mi.Func opt |> arg2 input target
    
    printfn "Optimizing..."
    for itr = 0 to 100 do
        optFun tstImgs tstLbls |> ignore
        let l = lossFun tstImgs tstLbls
        printfn "Loss afer %d iterations: %A" itr l



    ()


let ``Test Autoencoder`` () =
    let mc = ModelBuilder<single> "Autoencoder"
    
    // symbolic sizes
    let batchSize  = mc.Size "BatchSize"
    let nInput     = mc.Size "nInput"
    
    let pars = Autoencoder.pars (mc.Module "Autoencoder1") {NVisible=nInput; NLatent=mc.Fix 50; Tied=false}
    
    let input  =     mc.Var "Input"  [nInput; batchSize]

    let loss = Autoencoder.loss pars input 

    printfn "Autoencoder:\n%A" loss
    let dloss = Deriv.compute loss
    ()
    //printfn "%A" dloss



[<EntryPoint>]
let main argv = 
    
    ``Test neural net`` ()

    //``Test Autoencoder`` ()

    // need code to load data and perform regression
        

    
    0
