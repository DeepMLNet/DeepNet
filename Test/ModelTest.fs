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



let ``Test slice`` () =
    let ary : ArrayNDT<single> = ArrayNDHost.ones [5; 7; 4]
    //printfn "ary=\n%A" ary

    let slc1 = ary.[0..1, 1..3, 2..4]
    printfn "slc1=\n%A" slc1

    let slc1b = ary.[0..1, 1..3, *]
    printfn "slc1b=\n%A" slc1b

    let slc2 = ary.[1, 1..3, 2..4]
    printfn "slc2=\n%A" slc2

    let ary2 : ArrayNDT<single> = ArrayNDHost.ones [5; 4]
    //printfn "ary2=\n%A" ary2

    let slc3 = ary2.[SpecialNewAxis, 1..3, 2..4]
    printfn "slc3=\n%A" slc3

    let slc4 = ary2.[SpecialFill, 1..3, 2..4]
    printfn "slc4=\n%A" slc4


[<EntryPoint>]
let main argv = 
    
    ``Test slice`` ()

    //``Test neural net`` ()

    //``Test Autoencoder`` ()

    // need code to load data and perform regression
        

    
    0
