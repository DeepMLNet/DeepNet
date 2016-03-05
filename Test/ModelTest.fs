module ModelTest

open Basics
open ArrayNDNS

open SymTensor
open Models
open Datasets


let mnist = Mnist.load @"C:\Local\surban\dev\fexpr\Data\MNIST"


let ``Test neural net`` () =
    let mc = MC "NeuralNet"
    
    let input : ExprT<single> =     mc.Var "Input"  ["nInput"; "BatchSize"]
    let target =                    mc.Var "Target" ["nTarget"; "BatchSize"]

    let pars = NeuralLayer.parsFlexible (mc.Module "Layer1")
    let loss = NeuralLayer.loss pars input target

    //printfn "NeuralNet:\n%A" loss

    printfn "Parameters: %A" mc

    // compute loss of neural net on MNIST
    //let lossFun = Func.make onHost loss |> arg2 input target
    // Parameeter set must be passed
    // and individual parameters must be replaced with slices to the parameterset.
    // IMO it is best to keep individual parameter names as long as possible and substitute last.
    // optimizer also needs that thing
    // so separate substitute step?


    let lossFun = Func.make onHost loss |> arg2 input target

    // reorder MNIST
    let tstImgs =  
        mnist.TstImgs
        |> ArrayND.reorderAxes [2; 0; 1] 
        |> ArrayND.reshape [-1; (ArrayND.shape mnist.TstImgs).[0]]
    let tstLbls =  
        mnist.TstLbls
        |> ArrayND.reorderAxes [1; 0] 

    //let tstLoss = lossFun tstImgs tstLbls
    //printfn "Test loss on MNIST=%A" tstLoss

    let dloss = Deriv.compute loss
    ()
    //printfn "%A" dloss


let ``Test Autoencoder`` () =
    let mc = MC "Autoencoder"
    
    let input : ExprT<single> =     mc.Var "Input"  ["nInput"; "BatchSize"]

    let pars = Autoencoder.pars (mc.Module "Autoencoder1") {NLatent=50; Tied=false}
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
