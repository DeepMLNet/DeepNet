namespace Datasets

open Basics
open Tensor
open Util


/// A data sample consisting of an input and target array.
type InputTargetSampleT = {
    /// the input array
    Input:  Tensor<single>
    /// the target array
    Target: Tensor<single>
}
