module Models

open SymTensor



module LinearRegression =

    type Pars<'T> = {
        Weights:    ExprT<'T>;
    }

    let pars (nOut: int) (mc: MC) =
        {Weights = mc.Var "weights"     [nOut; ">nIn"]} 
        
    let pred (pars: Pars<'T>) (input: ExprT<'T>) =
        pars.Weights .* input

    let loss pars (input: ExprT<'T>) (target: ExprT<'T>) =
        let pred = pred pars input
        let diff = (pred - target) ** Expr.two<'T>()
        Expr.sum diff



module NeuralLayer = 

    type Pars<'T> = {
        Weights:    ExprT<'T>;
        Bias:       ExprT<'T>;
    }

    let pars (mc: MC) (nOut: int) =
        {Weights = mc.Var "weights"     [nOut; ">nIn"];
         Bias    = mc.Var "bias"        [nOut];}

    let parsFlexible (mc: MC) =
        {Weights = mc.Var "weights"     [">nOut"; ">nIn"];
         Bias    = mc.Var "bias"        [">nOut"];}

    let pred pars input =
        tanh (pars.Weights .* input + pars.Bias)

    let loss pars (input: ExprT<'T>) (target: ExprT<'T>) =
        let pred = pred pars input
        let diff = (pred - target) ** Expr.two<'T>()
        Expr.sum diff



module Autoencoder =

    type Pars<'T> = {
        InLayer:    NeuralLayer.Pars<'T>;
        OutLayer:   NeuralLayer.Pars<'T>;
    }

    type HyperPars = {
        NLatent:    int;
        Tied:       bool;
    }

    let pars (mc: MC) hyperPars =
        let p =
            {InLayer   = NeuralLayer.pars (mc.Module "InLayer") hyperPars.NLatent;
             OutLayer  = NeuralLayer.parsFlexible (mc.Module "OutLayer");}
        if hyperPars.Tied then
            {p with OutLayer = {p.OutLayer with Weights = p.InLayer.Weights.T}}
        else p

    let latent pars input =
        NeuralLayer.pred pars.InLayer input

    let recons pars input =
        let hidden = latent pars input
        NeuralLayer.pred pars.OutLayer hidden

    let loss pars (input: ExprT<'T>) = 
        let recons = recons pars input
        let diff = (recons - input) ** Expr.two<'T>()
        Expr.sum diff

