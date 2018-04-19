# Shape operations
The shape of a tensor specifies the highest valid index for each dimension.


## Getting the shape
The shape of a tensor can be accessed using the [Shape](xref:Tensor.Tensor`1.Shape*) property.
It is returned as an F# list.
The rank (number of dimensions) can be accessed using the [NDims](xref:Tensor.Tensor`1.NDims*) property.
The number of elements can be accessed using the [NElems](xref:Tensor.Tensor`1.NElems*) property.

The following example shows the basic usage of these properties.
```fsharp
printfn "a has shape %A, rank %d and %d elements." a.Shape a.NDims a.NElems
// a has shape [7L; 5L], rank 2 and 35 elements.
```

## Reshaping
Reshaping changes the shape of the tensor while keeping the number of elements constant.

For example consider the *4x4* matrix `b`, that is created as follows.
```fsharp
let b = HostTensor.init [4L; 4L] (fun [|y; x|] -> 4 * int y + int x)
// b =
//    [[   0    1    2    3]
//     [   4    5    6    7]
//     [   8    9   10   11]
//     [  12   13   14   15]]
```

We can use the [Tensor.reshape](xref:Tensor.Tensor`1.reshape*) function to transform this matrix into a vector of length *16*.
```fsharp
let b1 = Tensor.reshape [16L] b
// b1 = [   0    1    2    3    4    5 ...   12   13   14   15]
```

We can also specify the special identifier [Remainder](xref:Tensor.TensorVal.Remainder()) for the new size of at most one dimension.
In this case, its size will be chosen automatically (so that the number of elements does not change).
In the following example tensor `b` is reshaped into a three dimensional tensor of shape *4x2x2*.
```fsharp
let b2 = Tensor.reshape [4L; 2L; Remainder] b
// b2 =
//    [[[   0    1]
//      [   2    3]]
//     [[   4    5]
//      [   6    7]]
//     [[   8    9]
//      [  10   11]]
//     [[  12   13]
//      [  14   15]]]
```


### View or copy?
If the tensor to reshape has row-major memory layout, then [Tensor.reshape](xref:Tensor.Tensor`1.reshape*) creates a new view into the existing tensor.
Otherwise the tensor is copied during the reshape operation.

If you need to ensure that no copy is performed, i.e. the original and reshaped tensor share the same memory, use the [Tensor.reshapeView](xref:Tensor.Tensor`1.reshapeView*) function instead.
It will raise an error, if creating a reshaped view of the original tensor is impossible.


## Reordering axes and transposing
The [Tensor.swapDim](xref:Tensor.Tensor`1.swapDim*) function creates a new view of a tensor with the given dimensions swapped.
For example, the following code transpose the matrix `b`.
```fsharp
let b3 = Tensor.swapDim 0 1 b
// b3 =
//    [[   0    4    8   12]
//     [   1    5    9   13]
//     [   2    6   10   14]
//     [   3    7   11   15]]
```

The original and tensor with swapped axes share the same memory and modifications made to one of them will affect the other one.

A matrix can also be transposed using the [Tensor.transpose](xref:Tensor.Tensor`1.transpose*) function or the [T](xref:Tensor.Tensor`1.T*) property, i.e. `b.T` means transpose of matrix `b`.

The [Tensor.permuteAxes](xref:Tensor.Tensor`1.permuteAxes*) function can reorder axes arbitrarily.
It takes a list (of length equal to the rank of the tensor) with each element specifying the new position of the corresponding axis.
The list must be a permutation of the axes indices, i.e. duplicating or leaving out axes is not permitted.
For the rank three tensor `b2` from above, the following example code creates the view `b4` with shape *2x2x4*.
```fsharp
let b4 = Tensor.permuteAxes [2; 0; 1] b2
// b4.Shape = [2L; 2L; 4L]
```

It is important to understand that each list entry specifies where the axis *moves to*, not where it is coming from.
Thus, in this example, axis 0 becomes axis 2, axis 1 becomes axis 0 and axis 2 becomes axis 1.

## Adding axes
The [Tensor.padLeft](xref:Tensor.Tensor`1.padLeft*) and [Tensor.padRight](xref:Tensor.Tensor`1.padRight*) functions add a new axis of size one on the left or right respectively.

If you need to add an axis at another position, use the slicing operator with the special [NewAxis](xref:Tensor.TensorVal.NewAxis()) identifier.
The following example creates a view of shape *2x2x1x4*.

```fsharp
let b5 = b4.[*, *, NewAxis, *]
// b5.Shape = [2L; 2L; 1L; 4L]
```

## Broadcasting
An axis of size one can be repeated multiple times with the same value.
This is called broadcasting.

Consider the *1x4* matrix `c` created using the following code.
```fsharp
let c = HostTensor.init [1L; 4L] (fun [|_; i|] -> int i)
// c = [[   0    1    2    3]]
```

We can use the [Tensor.broadcastTo](xref:Tensor.Tensor`1.broadcastTo*) function to obtain a tensor with the first dimension repeated 3 times.
```fsharp
let c1 = Tensor.broadcastTo [3L; 4L] c
// c1 =
//    [[   0    1    2    3]
//     [   0    1    2    3]
//     [   0    1    2    3]]
```

Broadcasting creates a view of the original tensor, thus the repeated dimensions do not use additional memory and changing the broadcasted view will also change the original as well as all indices of a broadcasted dimension.
This is demonstrated by the following example.

```fsharp
c1.[[1L; 1L]] <- 11
// c1 =
//    [[   0   11    2    3]
//     [   0   11    2    3]
//     [   0   11    2    3]]
// c = [[   0   11    2    3]]
```

### Automatic broadcasting

Broadcasting is also performed automatically when performing element-wise operations between two tensors of different, but compatible, shapes.
This will be explained in the section about [tensor operations](Guide-Operations.md) of the guide.
