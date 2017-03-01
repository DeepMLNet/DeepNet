namespace Models

open ArrayNDNS
open SymTensor


module Util = 

    /// Only allows the derivative to pass into `expr` when `trainable` is true.
    let gradGate trainable expr = 
        if trainable then expr
        else Expr.assumeZeroDerivative expr


[<AutoOpen>]
module ActivationFuncTypes = 

    /// Transfer (activation) functions.
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


/// Activation function expressions.
module ActivationFunc = 

    /// identity activation function
    let id (x: ExprT) =
        x

    /// tanh activation function
    let tanh (x: ExprT) =
        tanh x

    /// sigmoid activation function
    let sigmoid (x: ExprT) =
        let one = Expr.scalarOfSameType x 1
        let two = Expr.scalarOfSameType x 2
        (tanh (x / two) + one) / two

    /// Soft-max activation function.
    /// The second dimension enumerates the possible classes.
    let softmax (x: ExprT) =
        let c = x |> Expr.maxKeepingAxis 1
        let y = exp (x - c)
        y / Expr.sumKeepingAxis 1 y

    /// Natural logarithm of soft-max activation function.
    /// The second dimension enumerates the possible classes.
    let logSoftmax (x: ExprT) =
        let c = x |> Expr.maxKeepingAxis 1
        x - c - log (Expr.sumKeepingAxis 1 (exp (x - c))) 

    /// Rectifier Unit function: max(x, 0)
    let relu (x: ExprT) =
        x |> Expr.zerosLike |> Expr.maxElemwise x

    /// applies the specified activation function
    let apply af x =
        match af with
        | Tanh       -> tanh x
        | Sigmoid    -> sigmoid x
        | SoftMax    -> softmax x
        | LogSoftmax -> logSoftmax x
        | Relu       -> relu x
        | Identity   -> id x

/// Regularization expressions.
module Regularization =

    let lqRegularization (weights:ExprT) (q:int) =
        Expr.sum (abs(weights) *** (single q))

    let l1Regularization weights =
        lqRegularization weights 1

    let l2Regularization weights =
        lqRegularization weights 1

/// Expressions concernred with Normal Distriutions
module NormalDistribution =
    
    /// PDF of standard normal distribution
    let pdf (x:ExprT) (mu:ExprT) (cov:ExprT)=
        let fact = 1.0f / sqrt( 2.0f * (single System.Math.PI)*cov)
        fact * exp( - ((x - mu) *** 2.0f) / (2.0f * cov))
    
    /// Computes approximate gaussian error 
    /// with maximum approximation error of 1.2 * 10 ** -7
    let gaussianErrorFkt (x:ExprT) = 
        
        let t = 1.0f/  (1.0f + 0.5f * abs(x))
        let sum = -1.26551233f + 1.00002368f * t + 0.37409196f * t *** 2.0f +
                   0.09678418f * t *** 3.0f - 0.18628806f * t *** 4.0f + 0.27886807f * t *** 5.0f -
                   1.13520398f * t *** 6.0f + 1.48851587f * t *** 7.0f - 0.82215223f * t *** 8.0f +
                   0.17087277f * t *** 9.0f
        let tau = t * exp(-x *** 2.0f + sum)
        Expr.ifThenElse (x>>==0.0f) (1.0f - tau) (tau - 1.0f)
    
    ///CDF of standard normal distribution
    let cdf (x:ExprT) (mu:ExprT) (cov:ExprT) =
        (1.0f + gaussianErrorFkt((x- mu) / sqrt(2.0f * cov))) / 2.0f
    
    /// Normalizes 
    let normalize (x:ExprT) =
        let mean = Expr.mean x
        let cov = (Expr.mean (x * x)) - (mean * mean)
        let stdev = sqrt cov
        let zeroCov = x - (Expr.reshape [SizeSpec.broadcastable] mean)
        let nonzeroCov = (x - (Expr.reshape [SizeSpec.broadcastable] mean)) / (Expr.reshape [SizeSpec.broadcastable] stdev)
        Expr.ifThenElse (cov ==== (Expr.zeroOfSameType cov)) zeroCov nonzeroCov