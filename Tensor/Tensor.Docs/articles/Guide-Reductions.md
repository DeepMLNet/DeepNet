# Reduction operations

Reduction functions are function that reduce the number of dimensions of a tensor by combining multiple elements into a single element.
Summation is an example of such an operation, since it takes multiple input values and outputs a single value.

## Summation
The [Tensor.sum](xref:Tensor.Tensor`1.sum*) function computes the sum of all elements of a tensor and returns it as a primitive value.
```fsharp
let s1 = Tensor.sum f
// s1 = 66.0
```

If you want the result to be returned as a scalar tensor instead of a primitive value, use the [Tensor.sumTensor](xref:Tensor.Tensor`1.sumTensor*) function instead.
This is useful for tensors stored on the GPU if the result is used for furhter computation, since it avoids the transfer back to host memory.

Often it is necessary to compute many sums in parallel, for example it might be interesting to compute the sums of all columns of a matrix.
For this purpose, the tensor library provides the [Tensor.sumAxis](xref:Tensor.Tensor`1.sumAxis*) function.
The following example illustrates its usage.

```fsharp
let g = HostTensor.init [4L; 4L] (fun [|y; x|] -> 4 * int y + int x)
// g =
//    [[   0    1    2    3]
//     [   4    5    6    7]
//     [   8    9   10   11]
//     [  12   13   14   15]]
let s2 = Tensor.sumAxis 0 g
// s2 = [  24   28   32   36]
```

This computed the sums of all columns of the matrix, thus resulting in a vector.

In general, the result tensor of a reduction function that ends in `Axis` (i.e. [Tensor.sumAxis](xref:Tensor.Tensor`1.sumAxis*), [Tensor.productAxis](xref:Tensor.Tensor`1.productAxis*), etc.) has one dimension less than the input tensor.

## Product
Likewise [Tensor.product](xref:Tensor.Tensor`1.product*) and [Tensor.productAxis](xref:Tensor.Tensor`1.productAxis*) compute the product of the elements of a tensor.

## Maximum and minimum
The [Tensor.min](xref:Tensor.Tensor`1.min*) and [Tensor.max](xref:Tensor.Tensor`1.max*) compute the minimum and maximum of a tensor and return a primitive value.
Analogously [Tensor.minAxis](xref:Tensor.Tensor`1.minAxis*) and [Tensor.maxAxis](xref:Tensor.Tensor`1.maxAxis*) compute the minimum and maximum over the given axis.

```fsharp
let m2 = Tensor.maxAxis 0 g
// m2 = [  12   13   14   15]
```

## Mean and variance
The [Tensor.mean](xref:Tensor.Tensor`1.mean*) and [Tensor.var](xref:Tensor.Tensor`1.var*) compute the emperical mean and variance of a tensor.
Variants computing the mean and variance along the specified axis are [Tensor.meanAxis](xref:Tensor.Tensor`1.meanAxis*) and [Tensor.varAxis](xref:Tensor.Tensor`1.varAxis*) respectively.

