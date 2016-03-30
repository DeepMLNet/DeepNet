namespace Models

open SymTensor



module LossLayer =

    type Measures =
        | MSE 
        | BinaryCrossEntropy

    let loss lm (pred: ExprT<'T>) (target: ExprT<'T>) =
        match lm with
        | MSE -> 
            (pred - target) ** Expr.two<'T>()
            |> Expr.mean
        | BinaryCrossEntropy ->
            -(target * log pred + (Expr.one<'T>() - target) * log (Expr.one<'T>() - pred))
            |> Expr.mean


module NeuralLayer = 

    type TransferFuncs =
        | Tanh
        | Identity

    type HyperPars = {
        NInput:         SizeSpecT
        NOutput:        SizeSpecT
        TransferFunc:   TransferFuncs
    }

    type Pars<'T> = {
        Weights:        ExprT<'T>
        Bias:           ExprT<'T>
        HyperPars:      HyperPars
    }

    let pars (mc: ModelBuilder<_>) hp = {
        Weights   = mc.Param "Weights"     [hp.NOutput; hp.NInput]
        Bias      = mc.Param "Bias"        [hp.NOutput]
        HyperPars = hp
    }

    let pred pars input =
        let activation = pars.Weights .* input + pars.Bias
        match pars.HyperPars.TransferFunc with
        | Tanh     -> tanh activation
        | Identity -> activation


module MLP =

    type HyperPars = {
        Layers:         NeuralLayer.HyperPars list
        LossMeasure:    LossLayer.Measures
    }

    type Pars<'T> = {
        Layers:         NeuralLayer.Pars<'T> list
        HyperPars:      HyperPars
    }

    let pars (mc: ModelBuilder<_>) (hp: HyperPars) = {
        Layers = hp.Layers 
                 |> List.mapi (fun idx nhp -> 
                    NeuralLayer.pars (mc.Module (sprintf "Layer%d" idx)) nhp)
        HyperPars = hp
    }

    let pred (pars: Pars<'T>) input =
        (input, pars.Layers)
        ||> List.fold (fun inp p -> NeuralLayer.pred p inp)
        
    let loss pars input target =
        LossLayer.loss pars.HyperPars.LossMeasure (pred pars input) target




//module Autoencoder =
//
//    type Pars<'T> = {
//        InLayer:    NeuralLayer.Pars<'T>;
//        OutLayer:   NeuralLayer.Pars<'T>;
//    }
//
//    type HyperPars = {
//        NVisible:   SizeSpecT;
//        NLatent:    SizeSpecT;
//        Tied:       bool;
//    }
//
//    let pars (mc: ModelBuilder<_>) hp =
//        let p =
//            {InLayer   = NeuralLayer.pars (mc.Module "InLayer") hp.NVisible hp.NLatent;
//             OutLayer  = NeuralLayer.pars (mc.Module "OutLayer") hp.NLatent hp.NVisible;}
//        if hp.Tied then
//            {p with OutLayer = {p.OutLayer with Weights = p.InLayer.Weights.T}}
//        else p
//
//    let latent pars input =
//        NeuralLayer.pred pars.InLayer input
//
//    let recons pars input =
//        let hidden = latent pars input
//        NeuralLayer.pred pars.OutLayer hidden
//
//    let loss pars (input: ExprT<'T>) = 
//        let recons = recons pars input
//        let diff = (recons - input) ** Expr.two<'T>()
//        Expr.sum diff
//
