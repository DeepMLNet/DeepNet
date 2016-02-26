module ModelTest

open Basics
open ArrayNDNS

open SymTensor
open Models


let ``Test neural net`` () =
    let mc = MC "NeuralNet"
    
    let input : ExprT<single> =     mc.Var "Input"  ["nInput"; "BatchSize"]
    let target =                    mc.Var "Target" ["nTarget"; "BatchSize"]

    let pars = NeuralLayer.parsFlexible (mc.Module "Layer1")
    let loss = NeuralLayer.loss pars input target

    printfn "NeuralNet:\n%A" loss
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

    ``Test Autoencoder`` ()

    // need code to load data and perform regression
        

    
    0
