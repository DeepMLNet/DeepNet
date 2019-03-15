namespace Tensor.Expr.ML

open DeepNet.Utils
open Tensor
open Tensor.Expr



/// Activation functions.
[<RequireQualifiedAccess>]
type ActFunc =
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
        (tanh (x / x.Scalar 2) + x.Scalar 1) / x.Scalar 2

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
        let zeros = Expr.zeros x.Dev x.Shape
        Expr.maxElemwise zeros x

    /// applies the specified activation function
    static member apply actFunc (x: Expr<'T>) =
        match actFunc with
        | Tanh       -> ActFunc.tanh x
        | Sigmoid    -> ActFunc.sigmoid x
        | SoftMax    -> ActFunc.softmax x
        | LogSoftmax -> ActFunc.logSoftmax x
        | Relu       -> ActFunc.relu x
        | Identity   -> ActFunc.id x



/// Regularization expressions.
module Regul =

    let lRegul (fac: float) (q: float) (w: Expr<'T>) =
        if fac <> 0.0 then 
            w.Scalar fac * Expr.sum (abs w *** w.Scalar q)
        else
            w.Scalar 0



/// Expressions concernred with Normal Distriutions
module NormalDistribution =
    
    /// PDF of standard normal distribution
    let pdf (x: Expr<'T>) (mu: Expr<'T>) (cov: Expr<'T>) =
        let fact = cov.Scalar 1 / sqrt( cov.Scalar 2 * cov.Scalar System.Math.PI * cov )
        fact * exp( - ((x - mu) *** cov.Scalar 2) / (cov.Scalar 2 * cov))
    
    /// Computes approximate gaussian error 
    /// with maximum approximation error of 1.2 * 10 ** -7
    let gaussianErrorFkt (x:Expr<'T>) = 
        let t = x.Scalar 1 / (x.Scalar 1 + x.Scalar 0.5 * abs x)
        let sum = -x.Scalar 1.26551233 + 
                   x.Scalar 1.00002368 * t + 
                   x.Scalar 0.37409196 * t *** x.Scalar 2 +
                   x.Scalar 0.09678418 * t *** x.Scalar 3 - 
                   x.Scalar 0.18628806 * t *** x.Scalar 4 + 
                   x.Scalar 0.27886807 * t *** x.Scalar 5 - 
                   x.Scalar 1.13520398 * t *** x.Scalar 6 + 
                   x.Scalar 1.48851587 * t *** x.Scalar 7 - 
                   x.Scalar 0.82215223 * t *** x.Scalar 8 +
                   x.Scalar 0.17087277 * t *** x.Scalar 9
        let tau = t * exp(-x *** x.Scalar 2 + sum)
        Expr.ifThenElse (x >>== x.Scalar 0) (x.Scalar 1 - tau) (tau - x.Scalar 1)
    
    /// CDF of standard normal distribution
    let cdf (x:Expr<'T>) (mu:Expr<'T>) (cov:Expr<'T>) =
        (x.Scalar 1 + gaussianErrorFkt( (x - mu) / sqrt(x.Scalar 2 * cov))) / x.Scalar 2
    
    /// Normalizes 
    let normalize (x:Expr<'T>) =
        let mean = Expr.mean x
        let cov = (Expr.mean (x * x)) - (mean * mean)
        let stdev = sqrt cov
        let zeroCov = x - (Expr.reshape [Size.broadcastable] mean)
        let nonzeroCov = (x - (Expr.reshape [Size.broadcastable] mean)) / (Expr.reshape [Size.broadcastable] stdev)
        Expr.ifThenElse (cov ==== x.Scalar 0) zeroCov nonzeroCov
