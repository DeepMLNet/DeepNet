namespace Tensor


/// <summary>Special constants that can be passed instead of indicies or parameter values or be returned 
/// from methods.</summary>
[<AutoOpen>]
module TensorVal =

    /// <summary>For slicing: inserts a new axis of size one.</summary>
    let NewAxis = SpecialIdx.NewAxis

    /// <summary>For slicing: fills all remaining axes with size one.</summary> 
    /// <remarks>Cannot be used together with <see cref="NewAxis"/>.</remarks>
    let Fill = SpecialIdx.Fill

    /// <summary>For reshape: remainder, so that number of elements stays constant.</summary>
    let Remainder = SpecialIdx.Remainder
    
    /// <summary>For search: value was not found.</summary>
    let NotFound = SpecialIdx.NotFound

    /// <summary>Indicates that the dimension is unmasked, i.e. equals specifying a tensor filled with trues.</summary>
    let NoMask = SpecialMask.NoMask

    