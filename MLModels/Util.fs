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
        one / (one + exp (-x))

    /// Soft-max activation function.
    /// The second dimension enumerates the possible classes.
    let softmax (x: ExprT) =
        // x[smpl, class]
        exp x / (Expr.sumKeepingAxis 1 (exp x))

    /// applies the specified activation function
    let apply af x =
        match af with
        | Tanh     -> tanh x
        | Sigmoid  -> sigmoid x
        | SoftMax  -> softmax x
        | Identity -> id x


