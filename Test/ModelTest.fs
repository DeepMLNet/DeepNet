module ModelTest

open Basics
open ArrayNDNS

open SymTensor
open Models


[<EntryPoint>]
let main argv = 
    let mc = MC "Test"
    
    let input : ExprT<single> =     mc.Var "Input"  ["nInput"; "BatchSize"]
    let target =                    mc.Var "Target" ["nTarget"; "BatchSize"]

    let pars = NeuralLayer.parsFlexible (mc.Module "NeuralLayer1")
    let loss = NeuralLayer.loss pars input target

    let senv = Expr.inferSymSizes loss
    SymSizeEnv.dump senv

    printfn "%A" loss


    let dloss = Deriv.compute loss

    //printfn "%A" dloss
    

    // need code to load data and perform regression
        

    
    0
