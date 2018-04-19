# Tensor operations

The tensor type supports all standard arithmetic operators and arithmetic functions.

## Element-wise arithmetic operators

The elementary arithmetic operators ([+](xref:Tensor.Tensor`1.op_Addition*), [-](xref:Tensor.Tensor`1.op_Subtraction*), [*](xref:Tensor.Tensor`1.op_Multiply*), [/](xref:Tensor.Tensor`1.op_Division*), [%](xref:Tensor.Tensor`1.op_Modulus*), [**](xref:Tensor.Tensor`1.Pow*), [~-](xref:Tensor.Tensor`1.op_UnaryNegation*)) are executed element-wise.
For example, consider the vectors `d` and `e`, that are initialized as follows.

```fsharp
let d = HostTensor.init [4L] (fun [|i|] -> float i)
let e = HostTensor.init [4L] (fun [|i|] -> 10.0 * float i)
// d = [   0.0000    1.0000    2.0000    3.0000]
// e = [   0.0000   10.0000   20.0000   30.0000]
```

Then we can perform an element-wise addition using the following code.

```fsharp
let f = d + e
// f = [   0.0000   11.0000   22.0000   33.0000]
```

It is also possible to apply an operator to a tensor and a scalar value of the same data type.
In this case the scalar is repeated to match the size of the given tensor, as shown in the following example.

```fsharp
let d1 = d * 100.0
// d1 = [   0.0000  100.0000  200.0000  300.0000]
```

### Automatic broadcasting
If a binary operator (for example `+`) is applied to two tensors of different shapes, the library tries to automatically broadcast both tensors to a compatible shape using the following rules.

1. If the tensors have different ranks, the tensor with the lower rank is padded from the left with axes of size one until both tensors have the same rank.
For example, if tensor `a` is of shape *4x3x1* and tensor `b` is of shape *2*, then tensor `b` is padded to the shape **1**x**1**x2.

1. For each dimension that has different size in both tensors and size one in one of the tensors, this dimension of the tensor with size one is broadcasted to the corresponding dimension of the other tensor.
Thus, in our example, the last dimension of tensor `a` is broadcasted resulting in the shape 4x3x**2** and the first and second dimensions of tensor `b` (after padding it is of shape *1x1x2*) are broadcasted, resulting in the shape **4**x**3**x2.

If the shapes still differ after applying the above rules, the operation fails and an exception is raised.

### Storage devices must match
All tensors participating in an operation must be located on the same storage device, i.e. their [Dev](xref:Tensor.Tensor`1.Dev*) property must be equal.
The result will be stored on the same device as the sources.
No automatic transfer between different devices (e.g. GPU and host memory) is performed; instead an exception is raised.

If working with tensors stored on different devices, you first have to use the [transfer](xref:Tensor.Tensor`1.transfer*) function to copy them to the same device, before applying an operator on them.

## Element-wise arithmetic and rounding functions
The standard F# arithmetic functions, such as [sin](xref:Tensor.Tensor`1.Sin*), [exp](xref:Tensor.Tensor`1.Exp*), [log](xref:Tensor.Tensor`1.Log*), and rounding functions, such as [floor](xref:Tensor.Tensor`1.Floor*) and [ceil](xref:Tensor.Tensor`1.Ceiling*), can also be applied to tensors.
They execute element-wise, just like the arithmetic operators presented above.
```fsharp
let f2 = sin f
// f2 = [   0.0000    -1.000    -0.009    0.9999]
```

## Specifying where the result should be stored
Sometimes you might want to specify in which (existing) tensor the result of an operation should be stored.
For this purpose there exist a corresponding `Fill*` variant of each operation provided by the tensor library.

For example, for the multiply operator [(*)](xref:Tensor.Tensor`1.op_Multiply*), there exist the [FillMultiply](xref:Tensor.Tensor`1.FillMultiply*) variant, which also performs a multiplication, but stores the result in the specified target tensor.
The previous contents of the target tensor are thereby overwritten.
The target tensor must have appropriate shape and data type to hold the result of the operation.
Also it must reside on the same device as the source(s) of the operation.

The following example illustrates the use of the `Fill*` functions.

```fsharp
// f3 = [  -1.0000   -1.0000   29.0000   40.0000]
f3.FillMultiply d e
// f3 = [   0.0000   10.0000   40.0000   90.0000]
```

It is also possible to perform operations in-place.
This means that one of the inputs of an operation is overwritten by the output.

```fsharp
f3.FillMultiply f3 e
// f3 = [   0.0000  100.0000  800.0000 2700.0000]
```

This is especially useful when working with very large tensors and thus care must be taken to conserve memory usage.

You can find the `Fill*` variants of each operation by checking the "*see also*" section in its reference documentation.

## Matrix multiplication (dot product)
Matrix multiplication (dot product) is implemented using the [.* operator](xref:Tensor.Tensor`1.op_DotMultiply*).
This operator can be used to calculate a vector/vector product resulting in a scalar, a matrix/vector product resulting in a vector and a matrix/matrix product resulting in a matrix.
If the inputs have more than two dimensions, a batched matrix/matrix product is computed.

The following example shows how to compute the matrix product of the *5x3* matrix `h` with the *3x3* matrix `i`, resulting in the *5x3* matrix `hi`.

```fsharp
let h = HostTensor.init [5L; 3L] (fun [|i; j|] -> 3.0 * float i + float j)
// h =
//     [[   0.0000    1.0000    2.0000]
//      [   3.0000    4.0000    5.0000]
//      [   6.0000    7.0000    8.0000]
//      [   9.0000   10.0000   11.0000]
//      [  12.0000   13.0000   14.0000]]
let i = 0.1 + HostTensor.identity 3L
// i =
//     [[   1.1000    0.1000    0.1000]
//      [   0.1000    1.1000    0.1000]
//      [   0.1000    0.1000    1.1000]]
let hi = h .* i
// hi =
```

## Linear algebra operations
The diagonal of a matrix can be extracted using the [diag](xref:Tensor.Tensor`1.diag*) function.
To create a diagonal matrix with specific elements on the diagonal use the [diagMat](xref:Tensor.Tensor`1.diagMat*) function.

The norm can be computed using the [norm](xref:Tensor.Tensor`1.norm*) and [normAxis](xref:Tensor.Tensor`1.normAxis*) functions.

To invert a square, invertable matrix use the [invert](xref:Tensor.Tensor`1.invert*) function.
However, this may be numerically instable, especially if the condition number of the matrix is low.
Thus it is usually better, but also more expensive, to compute the Moore-Penrose pseudo-inverse using the [pseudoInvert](xref:Tensor.Tensor`1.pseudoInvert*) function.
This is also applicable to non-square and non-invertable matrices.

The singular value decomposition (SVD) of a matrix is available through [SVD](xref:Tensor.Tensor`1.SVD*).
The eigen-decomposition of a symmetric matrix can be computed using the [symmetricEigenDecomposition](xref:Tensor.Tensor`1.symmetricEigenDecomposition*) function.

The tensor product (pairwise product between all elements of two tensors) between two tensors can be obtained using the the [tensorProduct](xref:Tensor.Tensor`1.tensorProduct*) function.

## Concatenation and block tensors
Tensors can be concatenated along an axis using the [concat](xref:Tensor.Tensor`1.concat*) function.

To replicate a tensor along an axis use the [replicate](xref:Tensor.Tensor`1.replicate*) function.

A tensor built out of smaller tensors block can be created using the [ofBlocks](xref:Tensor.Tensor`1.ofBlocks*) function.

## Element-wise function application (host-only)
For tensors stored in host memory, it is also possible to apply an arbitrary function element-wise using the [HostTensor.map](xref:Tensor.HostTensor.map*) function.

```fsharp
let f3 = HostTensor.map (fun x -> if x > 15.0 then 7.0 + x else -1.0) f
// f3 = [  -1.0000   -1.0000   29.0000   40.0000]
```

Likewise, the [HostTensor.map2](xref:Tensor.HostTensor.map2*) function takes two tensors and applies a binary function to their elements.
Indexed variants of both function are provided by [HostTensor.mapi](xref:Tensor.HostTensor.mapi*) and [HostTensor.mapi2](xref:Tensor.HostTensor.mapi2*).

The [HostTensor.foldAxis](xref:Tensor.HostTensor.foldAxis*) applies a function along the specified axis of a tensor and threads an state through the computation.

## Further algorithms

Further algorithms are provided in the [Tensor.Algorithm namespace](xref:Tensor.Algorithm).
