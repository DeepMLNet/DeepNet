namespace Models

open Tensor.Utils
open Tensor
open SymTensor



/// A layer that calculates the loss between predictions and targets using a difference metric.
module LossLayer =

    /// Difference metrics.
    type Measures =
        /// mean-squared error 
        | MSE 
        /// binary cross entropy
        | BinaryCrossEntropy
        /// multi-class cross entropy
        | CrossEntropy
        /// soft-max followed by multi-class cross entropy
        /// (use with identity transfer function in last layer)
        | SoftMaxCrossEntropy

    /// Returns an expression for the loss given the loss measure `lm`, the predictions
    /// `pred` and the target values `target`.
    /// If the multi-class cross entropy loss measure is used then
    /// pred.[smpl, cls] must be the predicted probability that the sample
    /// belong to class cls and target.[smpl, cls] must be 1 if the sample
    /// actually belongs to class cls and 0 otherwise.
    let loss lm (pred: ExprT) (target: ExprT) =
        // pred   [smpl, cls]
        // target [smpl, cls]
        let one = Expr.oneOfSameType pred
        let two = Expr.twoOfSameType pred
        match lm with
        | MSE -> 
            (pred - target) ** two
            |> Expr.mean
        | BinaryCrossEntropy ->
            -(target * log pred + (one - target) * log (one - pred))
            |> Expr.mean
        | CrossEntropy ->
            -target * log pred |> Expr.sumAxis 1 |> Expr.mean
        | SoftMaxCrossEntropy ->
            let c = pred |> Expr.maxKeepingAxis 1
            let logProb = pred - c - log (Expr.sumKeepingAxis 1 (exp (pred - c)))             
            -target * logProb |> Expr.sumAxis 1 |> Expr.mean


/// A layer of neurons (perceptrons).
module NeuralLayer = 

    /// Neural layer hyper-parameters.
    type HyperPars = {
        /// number of inputs
        NInput:             SizeSpecT
        /// number of outputs
        NOutput:            SizeSpecT
        /// transfer (activation) function
        TransferFunc:       ActivationFunc
        /// weights trainable
        WeightsTrainable:   bool
        /// bias trainable
        BiasTrainable:      bool
        /// l1 regularization weight
        L1Regularization:   single option
        /// l2 regularization weight
        L2Regularization:   single option
    }

    let defaultHyperPars = {
        NInput              = SizeSpec.fix 0L
        NOutput             = SizeSpec.fix 0L
        TransferFunc        = Tanh
        WeightsTrainable    = true
        BiasTrainable       = true
        L1Regularization    = None
        L2Regularization    = None
    }


    /// Neural layer parameters.
    type Pars<'T> = {
        /// expression for the weights
        Weights:        ExprT 
        /// expression for the biases
        Bias:           ExprT 
        /// hyper-parameters
        HyperPars:      HyperPars
    }

    let internal initWeights seed (shp: int64 list) : Tensor<'T> = 
        let rng = System.Random seed
        let fanOut = shp.[0] |> float
        let fanIn = shp.[1] |> float
        let r = 4.0 * sqrt (6.0 / (fanIn + fanOut)) 
        rng.UniformTensor (conv<'T> -r, conv<'T> r) shp
        
    let internal initBias seed (shp: int64 list) : Tensor<'T> =
        HostTensor.zeros shp

    /// Creates the parameters for the neural-layer in the supplied
    /// model builder `mb` using the hyper-parameters `hp`.
    /// The weights are initialized using random numbers from a uniform
    /// distribution with support [-r, r] where
    /// r = 4 * sqrt (6 / (hp.NInput + hp.NOutput)).
    /// The biases are initialized to zero.
    let pars (mb: ModelBuilder<_>) hp = {
        Weights   = mb.Param ("Weights", [hp.NOutput; hp.NInput], initWeights)
        Bias      = mb.Param ("Bias",    [hp.NOutput],            initBias)
        HyperPars = hp
    }

    /// Returns an expression for the output (predictions) of the
    /// neural layer with parameters `pars` given the input `input`.
    /// If the soft-max transfer function is used, the normalization
    /// is performed over axis 0.
    let pred pars (input: ExprT) =
        // weights [outUnit, inUnit]
        // bias    [outUnit]
        // input   [smpl, inUnit]
        // pred    [smpl, outUnit]
        let weights = 
            if pars.HyperPars.WeightsTrainable then pars.Weights 
            else Expr.assumeZeroDerivative pars.Weights
        let bias = 
            if pars.HyperPars.BiasTrainable then pars.Bias
            else Expr.assumeZeroDerivative pars.Bias
        let act = input .* weights.T + bias
        ActivationFunc.apply pars.HyperPars.TransferFunc act

    /// Calculates sum of all regularization terms of this layer.
    let regularizationTerm pars  =
        let weights = pars.Weights
        if pars.HyperPars.WeightsTrainable then
            let l1reg =
                match pars.HyperPars.L1Regularization with
                | Some f    -> f * Regularization.l1Regularization weights
                | None      -> Expr.zeroOfSameType weights
            let l2reg =
                match pars.HyperPars.L2Regularization with
                | Some f    -> f * Regularization.l1Regularization weights
                | None      -> Expr.zeroOfSameType weights
            l1reg + l2reg
        else 
            Expr.zeroOfSameType weights

/// A neural network (multi-layer perceptron) of multiple 
/// NeuralLayers and one LossLayer on top.
module MLP =

    /// MLP hyper-parameters.
    type HyperPars = {
        /// a list of the hyper-parameters of the neural layers
        Layers:         NeuralLayer.HyperPars list
        /// the loss measure
        LossMeasure:    LossLayer.Measures
    }

    /// MLP parameters.
    type Pars<'T> = {
        /// a list of the parameters of the neural layers
        Layers:         NeuralLayer.Pars<'T> list
        /// hyper-parameters
        HyperPars:      HyperPars
    }

    /// Creates the parameters for the neural network in the supplied
    /// model builder `mb` using the hyper-parameters `hp`.
    /// See `NeuralLayer.pars` for documentation about the initialization.
    let pars (mb: ModelBuilder<'T>) (hp: HyperPars) : Pars<'T> = {
        Layers = hp.Layers 
                 |> List.mapi (fun idx nhp -> 
                    NeuralLayer.pars (mb.Module (sprintf "Layer%d" idx)) nhp)
        HyperPars = hp
    }

    /// Returns an expression for the output (predictions) of the
    /// neural network with parameters `pars` given the input `input`.
    let pred (pars: Pars<'T>) input =
        (input, pars.Layers)
        ||> List.fold (fun inp p -> NeuralLayer.pred p inp)

    /// Returns an expression for the loss of the
    /// neural network with parameters `pars` given the input `input` and
    /// target values `target`.       
    let loss pars input target =
        LossLayer.loss pars.HyperPars.LossMeasure (pred pars input) target

    /// Calculates sum of all regularization terms of this model.
    let regualrizationTerm pars input=
        (Expr.zeroOfSameType input, pars.Layers)
        ||> List.fold (fun reg p -> NeuralLayer.regularizationTerm p)
        

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
