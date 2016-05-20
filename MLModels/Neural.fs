namespace Models

open Basics
open ArrayNDNS
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
        Weights:        ExprT<'T> ref
        Bias:           ExprT<'T> ref
        HyperPars:      HyperPars
    }

    let internal initWeights seed (shp: int list) : ArrayNDHostT<'T> = 
        let fanOut = shp.[0] |> float
        let fanIn = shp.[1] |> float
        let r = 4.0 * sqrt (6.0 / (fanIn + fanOut))
        let rng = System.Random seed
        
        rng.SeqDouble(-r, r)
        |> Seq.map conv<'T>
        |> ArrayNDHost.ofSeqWithShape shp
        
    let internal initBias seed (shp: int list) : ArrayNDHostT<'T> =
        Seq.initInfinite (fun _ -> conv<'T> 0)
        |> ArrayNDHost.ofSeqWithShape shp

    let pars (mc: ModelBuilder<_>) hp = {
        Weights   = mc.Param ("Weights", [hp.NOutput; hp.NInput], initWeights)
        Bias      = mc.Param ("Bias",    [hp.NOutput],            initBias)
        HyperPars = hp
    }

    let pred pars input =
        let activation = !pars.Weights .* input + !pars.Bias
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
