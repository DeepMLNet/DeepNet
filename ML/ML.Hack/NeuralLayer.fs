namespace Models.Neural

open DeepNet.Utils
open Tensor
open Tensor.Expr



/// Activation functions.
type ActivationFunc =
    /// tanh transfer function
    | Tanh
    /// sigmoid transfer function
    | Sigmoid        
    /// soft-max transfer function
    | SoftMax
    /// logarithm of soft-max transfer function
    | LogSoftmax
    /// rectifier unit transfer function
    | Relu
    /// no transfer function
    | Identity
with 

    /// identity activation function
    static member id (x: Expr<'T>) =
        x

    /// tanh activation function
    static member tanh (x: Expr<'T>) =
        tanh x

    /// sigmoid activation function
    static member sigmoid (x: Expr<'T>) =
        let one = Expr<_>.scalar x.Dev (conv<'T> 1)
        let two = Expr<_>.scalar x.Dev (conv<'T> 2)
        (tanh (x / two) + one) / two

    /// Soft-max activation function.
    /// The second dimension enumerates the possible classes.
    static member softmax (x: Expr<'T>) =
        let c = x |> Expr.maxKeepingAxis 1
        let y = exp (x - c)
        y / Expr.sumKeepingAxis 1 y

    /// Natural logarithm of soft-max activation function.
    /// The second dimension enumerates the possible classes.
    static member logSoftmax (x: Expr<'T>) =
        let c = x |> Expr<_>.maxKeepingAxis 1
        x - c - log (Expr<_>.sumKeepingAxis 1 (exp (x - c))) 

    /// Rectifier Unit function: max(x, 0)
    static member relu (x: Expr<'T>) =
        x |> Expr.zerosLike |> Expr.maxElemwise x

    /// applies the specified activation function
    static member apply af (x: Expr<'T>) =
        match af with
        | Tanh       -> ActivationFunc.tanh x
        | Sigmoid    -> ActivationFunc.sigmoid x
        | SoftMax    -> ActivationFunc.softmax x
        | LogSoftmax -> ActivationFunc.logSoftmax x
        | Relu       -> ActivationFunc.relu x
        | Identity   -> ActivationFunc.id x

/// Regularization expressions.
module Regularization =

    let lqRegularization (weights:Expr) (q:int) =
        Expr.sum (abs(weights) *** (single q))

    let l1Regularization weights =
        lqRegularization weights 1

    let l2Regularization weights =
        lqRegularization weights 1



/// Neural layer hyper-parameters.
type HyperPars = {
    /// number of inputs
    NInput:             SizeSpec
    /// number of outputs
    NOutput:            SizeSpec
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


