namespace Tensor


/// Special constants that can be passed instead of indicies or parameter values or be returned from methods.
[<AutoOpen>]
module TensorVal =

    /// For slicing: inserts a new axis of size one.
    let NewAxis = SpecialIdx.NewAxis

    /// For slicing: fills all remaining axes with size one. 
    /// Cannot be used together with NewAxis.
    let Fill = SpecialIdx.Fill

    /// For reshape: remainder, so that number of elements stays constant.
    let Remainder = SpecialIdx.Remainder
    
    /// For search: value was not found.
    let NotFound = SpecialIdx.NotFound

    /// All elements.
    let RngAll = SpecialRng.RngAll

    /// Indicates that the dimension is unmasked, i.e. equals specifying a tensor filled with trues. 
    let NoMask = SpecialMask.NoMask

    