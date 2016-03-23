namespace Models

open SymTensor


module NeuralLayer = 

    type Pars<'T> = {
        Weights:    ExprT<'T>;
        Bias:       ExprT<'T>;
    }

    let pars (mc: ModelBuilder<_>) nIn nOut =
        {Weights = mc.Param "Weights"     [nOut; nIn];
         Bias    = mc.Param "Bias"        [nOut];}

    let parsFromInput (mc: ModelBuilder<_>) input nOut =
        pars mc (Expr.shapeOf input).[0] nOut

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
        NVisible:   SizeSpecT;
        NLatent:    SizeSpecT;
        Tied:       bool;
    }

    let pars (mc: ModelBuilder<_>) hp =
        let p =
            {InLayer   = NeuralLayer.pars (mc.Module "InLayer") hp.NVisible hp.NLatent;
             OutLayer  = NeuralLayer.pars (mc.Module "OutLayer") hp.NLatent hp.NVisible;}
        if hp.Tied then
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

