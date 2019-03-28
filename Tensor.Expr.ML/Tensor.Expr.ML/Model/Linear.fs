namespace Tensor.Expr.ML

open DeepNet.Utils
open Tensor
open Tensor.Expr


/// Linear regression.
module LinearRegression =

    /// Linear regression hyperparameters.
    type HyperPars<'T> = {
        /// number of inputs
        NInput:     Size
        /// number of outputs
        NOutput:    Size
        /// l1 regularization weight
        L1Regul:    float 
        /// l2 regularization weight
        L2Regul:    float 
    } with

        /// Default hyper-parameters.
        static member standard nInput nOutput = {
            NInput = nInput
            NOutput = nOutput
            L1Regul = 0.0
            L2Regul = 0.0
        }

    /// Linear regression parameters.
    type Pars<'T> = {
        /// weights
        Weights:    Var<'T> 
        /// hyper-parameters
        HyperPars:  HyperPars<'T>
    }

    let internal initWeights (rng: System.Random) (weights: Tensor<'T>) = 
        let r = 1e-5
        let weightsHost = 
            HostTensor.randomUniform rng (conv<'T> -r, conv<'T> r) weights.Shape
        weights.TransferFrom weightsHost

    /// Creates the linear regression parameters.
    let pars (ctx: Context) rng (hp: HyperPars<'T>) = {
        Weights     = Var<'T> (ctx / "Weights", [hp.NOutput; hp.NInput]) |> Var<_>.toPar (initWeights rng)
        HyperPars   = hp
    }
        
    /// The predictions of the linear regression.
    let pred (pars: Pars<'T>) (input: Expr<'T>) =
        // input [smpl, inUnit]
        // pred  [smpl, outInit]
        input .* (Expr pars.Weights).T        

    /// The regularization terms.
    let regul (pars: Pars<'T>) =
        let l1reg = Regul.lRegul pars.HyperPars.L1Regul 1.0 (Expr pars.Weights)
        let l2reg = Regul.lRegul pars.HyperPars.L2Regul 2.0 (Expr pars.Weights)
        l1reg + l2reg

    /// The loss (mean-squared error plus regularization terms).
    let loss pars (input: Expr<'T>) (target: Expr<'T>) =
        let mse = 
            (pred pars input - target) ** input.Scalar 2
            |> Expr.sumAxis 1
            |> Expr.mean
        mse + regul pars
