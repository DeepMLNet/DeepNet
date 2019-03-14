namespace Tensor.Expr.ML

open DeepNet.Utils
open Tensor
open Tensor.Expr


/// A layer that calculates the loss between predictions and targets using a difference metric.
module LossLayer =

    /// Difference metrics.
    type Measure =
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
    let loss lm (pred: Expr<'T>) (target: Expr<'T>) =
        // pred   [smpl, cls]
        // target [smpl, cls]
        let one = Expr.scalar pred.Dev (conv<'T> 1)
        let two = Expr.scalar pred.Dev (conv<'T> 2)
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
    type HyperPars<'T> = {
        /// number of inputs
        NInput:             Size
        /// number of outputs
        NOutput:            Size
        /// transfer (activation) function
        ActFunc:            ActFunc
        /// l1 regularization weight
        L1Regul:            float 
        /// l2 regularization weight
        L2Regul:            float 
    } with 
        static member standard nInput nOutput : HyperPars<'T> = {
            NInput     = nInput
            NOutput    = nOutput
            ActFunc    = ActFunc.Tanh
            L1Regul    = 0.0
            L2Regul    = 0.0
        }


    /// Neural layer parameters.
    type Pars<'T> = {
        /// weights
        Weights:        Var<'T>
        /// bias
        Bias:           Var<'T>
        /// hyper-parameters
        HyperPars:      HyperPars<'T>
    }

    let internal initWeights (rng: System.Random) (weights: Tensor<'T>) = 
        let fanOut = weights.Shape.[0] |> float
        let fanIn = weights.Shape.[1] |> float
        let r = 4.0 * sqrt (6.0 / (fanIn + fanOut))
        let weightsHost = 
            HostTensor.randomUniform rng (conv<'T> -r, conv<'T> r) weights.Shape
        weights.TransferFrom weightsHost
        
    let internal initBias (rng: System.Random) (bias: Tensor<'T>) =
        bias.FillZeros ()

    /// Creates the parameters for the neural-layer in the supplied
    /// model builder `mb` using the hyper-parameters `hp`.
    /// The weights are initialized using random numbers from a uniform
    /// distribution with support [-r, r] where
    /// r = 4 * sqrt (6 / (hp.NInput + hp.NOutput)).
    /// The biases are initialized to zero.
    let pars (ctx: Context) rng (hp: HyperPars<'T>) = {
        Weights     = Var<'T> (ctx / "Weights", [hp.NOutput; hp.NInput]) |> Var<_>.toPar (initWeights rng)
        Bias        = Var<'T> (ctx / "Bias",    [hp.NOutput])            |> Var<_>.toPar (initBias rng)
        HyperPars   = hp
    }

    /// Returns an expression for the output (predictions) of the
    /// neural layer with parameters `pars` given the input `input`.
    /// If the soft-max transfer function is used, the normalization
    /// is performed over axis 0.
    let pred (pars: Pars<'T>) (input: Expr<'T>) =
        // weights [outUnit, inUnit]
        // bias    [outUnit]
        // input   [smpl, inUnit]
        // pred    [smpl, outUnit]
        let act = input .* (Expr pars.Weights).T + Expr pars.Bias
        ActFunc.apply pars.HyperPars.ActFunc act

    /// The regularization term for this layer.
    let regul (pars: Pars<'T>) =
        let l1reg = Regul.lRegul pars.HyperPars.L1Regul 1.0 (Expr pars.Weights)
        let l2reg = Regul.lRegul pars.HyperPars.L2Regul 2.0 (Expr pars.Weights)
        l1reg + l2reg




/// A neural network (multi-layer perceptron) of multiple 
/// NeuralLayers and one LossLayer on top.
module MLP =

    /// MLP hyper-parameters.
    type HyperPars<'T> = {
        /// a list of the hyper-parameters of the neural layers
        Layers:         NeuralLayer.HyperPars<'T> list
        /// the loss measure
        LossMeasure:    LossLayer.Measure
    }

    /// MLP parameters.
    type Pars<'T> = {
        /// a list of the parameters of the neural layers
        Layers:         NeuralLayer.Pars<'T> list
        /// hyper-parameters
        HyperPars:      HyperPars<'T>
    }

    /// Creates the parameters for the neural network in the supplied
    /// model builder `mb` using the hyper-parameters `hp`.
    /// See `NeuralLayer.pars` for documentation about the initialization.
    let pars (ctx: Context) rng (hp: HyperPars<'T>) : Pars<'T> = {
        Layers = hp.Layers 
                 |> List.mapi (fun idx nhp -> 
                        NeuralLayer.pars (ctx / sprintf "Layer%d" idx) rng nhp)
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
    let loss (pars: Pars<'T>) input target =
        LossLayer.loss pars.HyperPars.LossMeasure (pred pars input) target

    /// Calculates sum of all regularization terms of this model.
    let regualrizationTerm (pars: Pars<'T>) (input: Expr<'T>) =
        (input.Scalar 0, pars.Layers)
        ||> List.fold (fun reg p -> reg + NeuralLayer.regul p)
        

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
