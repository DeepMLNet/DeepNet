namespace Datasets

open Basics
open ArrayNDNS
open Util


/// A data sample consisting of an input and target array.
type InputTargetSampleT = {
    /// the input array
    Input:  ArrayNDT<single>
    /// the target array
    Target: ArrayNDT<single>
}
