namespace Tensor.Expr.ML

open Tensor


/// A data sample consisting of an input and target tensor.
type InpTgtSmpl<'T> = {
    /// the input 
    Input:  Tensor<'T>
    /// the target 
    Target: Tensor<'T>
}
