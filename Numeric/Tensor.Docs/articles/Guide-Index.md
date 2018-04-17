# Index functions

Index functions are functions for working with indices of tensors.

## Sequence of all indices

The [Tensor.allIdx](xref:Tensor.Tensor`1.allIdx*) function returns a sequence of indices that sequentially enumerate all elements within the specified tensor.

```fsharp
let a = HostTensor.zeros<int> [2L; 3L]
let s = Tensor.allIdx a
// s = seq [[0L; 0L]; [0L; 1L]; [0L; 2L]; [1L; 0L]; [1L; 1L]; [1L; 2L]]
```

## Indices of maximum and minimum

The [Tensor.argMax](xref:Tensor.Tensor`1.argMax*) and [Tensor.argMin](xref:Tensor.Tensor`1.argMin*) return the index of the element with the highest or lowest value within the tensor.
Using them on an empty tensor raises an exception.

```fsharp
let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
                             [5.0; 6.0; 7.0; 8.0]]
let b = Tensor.argMax a
// b = [1L; 3L]
```

The above example finds the index of the largest element of the matrix `a`, which is *8.0* at position *1,3*.

As with reduction operations, there exist variants of these functions that work along a specified axis of the tensor.
The [Tensor.argMaxAxis](xref:Tensor.Tensor`1.argMaxAxis*) and [Tensor.argMinAxis](xref:Tensor.Tensor`1.argMin*) find the highest or lowest elements along the specified axis and return them as a tensor as indices.

```fsharp
let b = Tensor.argMaxAxis 1 a
// b = [3L; 3L]
```

The above example finds the maximum values along each row of matrix `a`, which are located at *0,3* and *1,3*.

## Find index of specified value

The [Tensor.tryFind](xref:Tensor.Tensor`1.tryFind*) function searches for the specified value and returns the index of its first occurence as an option value.

```fsharp
let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
                             [5.0; 6.0; 7.0; 3.0]]
let b = Tensor.tryFind 3.0 a
// b = Some [0L; 2L]
```

If you want to find the first occurence of a value along a specific axis, use the [Tensor.findAxis](xref:Tensor.Tensor`1.findAxis*) function.

```fsharp
let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
                             [5.0; 6.0; 7.0; 3.0]]
let b = Tensor.findAxis 3.0 1 a
// b = [2L; 3L]
```

If the specified value is not present, this function returns the special value [NotFound](xref:Tensor.Tensor`1.NotFound) instead.

## Indices of all true values

For boolean tensors, the [Tensor.trueIdx](xref:Tensor.Tensor`1.trueIdx*) function returns all indices corresponding to `true` entries in the tensor.

```fsharp
let a = HostTensor.ofList2D [[true; false; true; false]
                             [false; true; true; false]]
let b = Tensor.trueIdx a
// b = [[0L; 0L]
//      [0L; 2L]
//      [1L; 1L]
//      [1L; 2L]]
```

## Gathering and scattering by index

The gather and scatter operations use a source tensor and a tensor of indices to build a new tensor.
They are useful for building lookup tables.

### Gather

The [Tensor.gather](xref:Tensor.Tensor`1.gather*) function gathers elements from a source tensor using the indices specified in the index tensor.

In the following example, we gather the elements with the indices *1,3*, *2,1*, *0,0* and *0,3* from the matrix `src` and store them in the new tensor `g`.

```fsharp
let src = HostTensor.ofList2D [[0.0; 0.1; 0.2; 0.3]
                               [1.0; 1.1; 1.2; 1.3]
                               [2.0; 2.1; 2.2; 2.3]]
let i0 = HostTensor.ofList [1L; 2L; 0L; 0L]
let i1 = HostTensor.ofList [3L; 1L; 0L; 3L]
let g = Tensor.gather [Some i0; Some i1] src
// g = [1.3000    2.1000    0.0000    0.3000]
```

If any index is out of range, an exception is raised.

You can also specify `None` instead of an index tensor for a particular dimension to assume an identity mapping.
The following example demonstrates this by selecting the indices *0,3*, *1,1* and *2,0* from the matrix `src`.

```fsharp
let j1 = HostTensor.ofList [3L; 1L; 0L]
let g2 = Tensor.gather [None; Some j1] src
// g2 = [0.3000    1.1000    2.0000]
```

### Scatter

The [Tensor.scatter](xref:Tensor.Tensor`1.scatter*) function can be thought of as the inverse operation of gathering.
It takes the elements from the source tensor and writes them into the locations specified by the index tensor.
If the same index occurs multiple times, all elements written to it are summed.
If a location of the target tensor never occurs, its value will be zero.

The following example sums the first row of matrix `src` into element *0,3* of result tensor `s`.
It further swaps rows *1* and *2* of matrix `src`.
Since row *3* of result tensor `s` is not referenced, its values are all zero.
```fsharp
let src = HostTensor.ofList2D [[0.0; 0.1; 0.2; 0.3]
                               [1.0; 1.1; 1.2; 1.3]
                               [2.0; 2.1; 2.2; 2.3]]
let i0 = HostTensor.ofList2D [[0L; 0L; 0L; 0L]
                              [2L; 2L; 2L; 2L]
                              [1L; 1L; 1L; 1L]]
let i1 = HostTensor.ofList2D [[3L; 3L; 3L; 3L]
                              [0L; 1L; 2L; 3L]
                              [0L; 1L; 2L; 3L]]
let s = Tensor.scatter [Some i0; Some i1] [4L; 4L] src
// s =
//     [[   0.0000    0.0000    0.0000    0.6000]
//      [   2.0000    2.1000    2.2000    2.3000]
//      [   1.0000    1.1000    1.2000    1.3000]
//      [   0.0000    0.0000    0.0000    0.0000]]
```

