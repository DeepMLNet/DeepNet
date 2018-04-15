# Working with tensors

A *tensor* is an n-dimensional array of an arbitrary data type (for example `single` or `double`).
Tensors of data type `'T` are implemented by the [Tensor<'T>](xref:Tensor.Tensor`1) type.

A tensor can be either stored in host memory or in the memory of a GPU computing device.
Currenty only nVidia cards implementing the [CUDA API](https://developer.nvidia.com/cuda-zone) are supported.
The API for host and GPU stored tensors is mostly equal, thus a program can make use of GPU accelerated operations without porting effort.

The tensor library provides functionality similar to [Numpy's Ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.html) and [MATLAB arrays](http://www.mathworks.com/help/matlab/matrices-and-arrays.html), including vector-wise operations, reshaping, slicing, broadcasting, masked assignment, reduction operations and BLAS operations.

This open source library is written in [F#](http://fsharp.org/) and targets the [.NET Standard 2.0 platform](https://github.com/dotnet/standard/blob/master/docs/versions/netstandard2.0.md) with Linux and Microsoft Windows as supported operating systems.

## Architecture

To work with the Tensor library, install the NuGet packages as described in the [installation guide](Guide-Installation.md) and open the `Tensor` namespace within your source file.
```fsharp
open Tensor
```
The primary type you will work with is [Tensor<'T>](xref:Tensor.Tensor`1).
It provides functions to work with tensors regardless of their storage device.
The modules [HostTensor](xref:Tensor.HostTensor) and [CudaTensor](xref:Tensor.CudaTensor) contain additional functions that are only applicable to tensors stored in host or GPU memory respectively.


## Creating tensors

Let us create a $3 \times 2$ matrix, i.e. a two-dimensional tensor, of data type `int` filled with zeros in host memory.
For this purpose we use the [Tensor<'T>.zeros](xref:Tensor.Tensor`1.zeros*) function.
```fsharp
let z1 = Tensor<int>.zeros HostTensor.Dev [3L; 2L]
// z1 = [[0; 0]
//       [0; 0]
//       [0; 0]]
```
The type argument `int` tells the function which data type to use.
In many cases, it can be automatically inferred and thus omitted, but in this example there is not way for the compiler to automatically find out which data type to use.

The first argument to the `zeros` function specifies the device to use.
In this case we specified [HostTensor.Dev](xref:Tensor.HostTensor.Dev) to store the tensor in host memory.
The second argument specifies the desired shape.
All shapes and indices in this tensor library are 64-bit integers.
Thus we have to use the `L` postfix when writing integer literals, i.e. `3L` instead of `3`.

Since creating tensors in host memory is a very common operation, the library also provides the shorter notation
```fsharp
let z1 = HostTensor.zeros<int> [3L; 2L]
// z1 = [[0; 0]
//       [0; 0]
//       [0; 0]]
```
to perform the same task.
These shorthands are available for all tensor creation function and listed in the [HostTensor](xref:Tensor.HostTensor) module.

Similarly, we can use the [Tensor.ones](xref:Tensor.Tensor`1.ones) function to obtain a vector of data type `single` and size `3` filled with ones.
```fsharp
let o1 = Tensor<single>.ones HostTensor.Dev [3L]
// o1 = [0.0f; 0.0f; 0.0f]
```
The [Tensor<'T>.identity](xref:Tensor.Tensor`1.identity) function creates an identity matrix of the given size.
```fsharp
let id1 = Tensor<float>.identity HostTensor.Dev 3L
// id1 = [[1.0; 0.0; 0.0]
//        [0.0; 1.0; 0.0]
//        [0.0; 0.0; 1.0]]
```
This created a $3 \times 3$ identity matrix.

### Scalar tensors
A scalar tensor is a tensor that has a dimensionality of zero.
It contains exactly one element and can be treated like a tensor of any other dimensionality.
However, for convenience, special functions are provided to make working with scalar tensors easier.

A scalar tensor can be created with the [Tensor<'T>.scalar](xref:Tensor.Tensor`1.scalar) function (or its corresponding [HostTensor.scalar](xref:Tensor.HostTensor.scalar) shorthand).
```fsharp
let s1 = Tensor.scalar HostTensor.Dev 33.2
// s1 = 33.2
// s1.NDims = 0
// s1.Shape = []
// s1.NElems = 1L
```
Specifying an empty shape when using other creation methods, such as [Tensor<'T>.zeros](xref:Tensor.Tensor`1.zeros*), will also create a scalar tensor.

The numeric value of a scalar tensor can be obtained (and changed) using the [Tensor<'T>.Value](xref:Tensor.Tensor`1.Value) property.
```fsharp
printfn "The numeric value of s1 is %f." s1.Value
// The numeric value of s1 is 33.2.
```
If you try to use this property on a non-scalar tensor, an exception will be raised.

### Host-only creation methods
Some tensor creation methods can only produce tensors stored in host memory, which, of course, can be transferred to GPU memory subsequently.
For example the [HostTensor.init](xref:Tensor.HostTensor.init) function takes a function and uses it to compute the initial value of each element of the tensor.
```fsharp
let a = HostTensor.init [7L; 5L] (fun [i; j] -> 5.0 * float i + float j)
// a =
//    [[   0.0000    1.0000    2.0000    3.0000    4.0000]
//     [   5.0000    6.0000    7.0000    8.0000    9.0000]
//     [  10.0000   11.0000   12.0000   13.0000   14.0000]
//     [  15.0000   16.0000   17.0000   18.0000   19.0000]
//     [  20.0000   21.0000   22.0000   23.0000   24.0000]
//     [  25.0000   26.0000   27.0000   28.0000   29.0000]
//     [  30.0000   31.0000   32.0000   33.0000   34.0000]]
```
The first argument specifies the shape of the tensor.
The second argument is a function that takes the n-dimensional index (zero-based) of an entry and computes its initial value; here we use the formula $5i + j$ where $i$ is the row and $j$ is the column of the matrix.
The data type (here `float`) is automatically inferred from the return type of the initialization function.

### Creation from F# sequences, lists and arrays
The [HostTensor.ofSeq](xref:Tensor.HostTensor.ofSeq) converts an [F# sequence](https://en.wikibooks.org/wiki/F_Sharp_Programming/Sequences) of finite length into a one-dimensional tensor.
```fsharp
let seq1 = seq { for i=0 to 20 do if i % 3 = 0 then yield i } |> HostTensor.ofSeq
// seq1 = [   0    3    6    9   12   15   18]
```
The example above creates a vector of all multiplies of 3 in the range between 0 and 20.

A list can be converted into a one-dimensional tensor using the [HostTensor.ofList](xref:Tensor.HostTensor.ofList) function.
To convert an array into a tensor use the [HostTensor.ofArray](xref:Tensor.HostTensor.ofArray) function.
The [HostTensor.ofList2D](xref:Tensor.HostTensor.ofList2D) and [HostTensor.ofArray2D](xref:Tensor.HostTensor.ofArray2D) take two-dimensional lists or arrays and convert them into tensors of respective shapes.

### Conversion to F# sequences, lists and arrays
Use the [HostTensor.toSeq](xref:Tensor.HostTensor.toSeq) function to expose the elements of a tensor as a sequence.
If the tensor has more than one dimension, it is flattened before the operation is performed.

Use the [HostTensor.toList](xref:Tensor.HostTensor.toList) or [HostTensor.toList2D](xref:Tensor.HostTensor.toList2D) functions to convert a tensor into a list.
The [HostTensor.toArray](xref:Tensor.HostTensor.toArray), [HostTensor.toArray2D](xref:Tensor.HostTensor.toArray2D), [HostTensor.toArray3D](xref:Tensor.HostTensor.toArray3D) convert a tensor into an array of respective dimensionality.

All these operations copy the elements of the tensor.

Printing tensors and string representation
------------------------------------------

Tensors can be printed using the `%A` format specifier of the standard `printf` function.
```fsharp
printfn "The tensor seq1 is\n%A" seq1
// The tensor seq1 is
// [   0    3    6    9   12   15   18]
```
The output of large tensors is automatically truncated to a reasonable size.
The corresponding string representation can also be accessed thorugh the [Pretty](xref:Tensor.Tensor`1.Pretty) property.
The full (untruncated) string representation is available through the [Full](xref:Tensor.Tensor`1.Full) property.
Use the [ToString](xref:Tensor.Tensor`1.ToString) method when it is required to adjust the maximum number of elements that are printed before truncation occurs.

Accessing individual elements
-----------------------------

Individual elements of a tensor can be accessed using the `tensor.[[idx0; idx1; ...; idxN]]` notation.
Zero-based indexing is used.
```fsharp
// a =
//    [[   0.0000    1.0000    2.0000    3.0000    4.0000]
//     [   5.0000    6.0000    7.0000    8.0000    9.0000]
//     [  10.0000   11.0000   12.0000   13.0000   14.0000]
//     [  15.0000   16.0000   17.0000   18.0000   19.0000]
//     [  20.0000   21.0000   22.0000   23.0000   24.0000]
//     [  25.0000   26.0000   27.0000   28.0000   29.0000]
//     [  30.0000   31.0000   32.0000   33.0000   34.0000]]
let v = a.[[1L; 1L]]
// v = 6.0
```
The above example accesses the element at index $1,1$.
Note that the indices are specified as 64-bit integers surrounded by double brackets (`[[` and `]]`) and separated using a semicolon.

Tensors are mutable objects.
An element can be changed using the `tensor.[[idx0; idx1; ...; idxN]] <- newValue` notation.
```fsharp
a.[[2L; 2L]] <- 55.
// a =
//    [[   0.0000    1.0000    2.0000    3.0000    4.0000]
//     [   5.0000    6.0000    7.0000    8.0000    9.0000]
//     [  10.0000   11.0000   55.0000   13.0000   14.0000]
//     [  15.0000   16.0000   17.0000   18.0000   19.0000]
//     [  20.0000   21.0000   22.0000   23.0000   24.0000]
//     [  25.0000   26.0000   27.0000   28.0000   29.0000]
//     [  30.0000   31.0000   32.0000   33.0000   34.0000]]
```
The above example changes the value at index $2,2$ to 55.


Slicing
-------

Slicing creates a new view into an existing tensor.
Slicing is done using the `tensor.[rng0, rng1, ..., rngN]` notation.
Note that the ranges are specified within single brackets and separated using commas.

Let us select the first row of tensor a.

```fsharp
let a1 = a.[0, *]
```

The asterisk selects all elements of the specified dimension.
The result is a tensor of rank one (i.e. a vector) with the entries

    [   0.0000    1.0000    2.0000    3.0000    4.0000]

Since `a1` is a view of `a` it shares the same memory.
Changing an element of `a1` by assigning a new value to it

```fsharp
a1.[[1]] <- 99.
```

changes the tensor `a` as well.
This can be seen by outputting `a1` and `a`.

```fsharp
printfn "a1 is now\n%A" a1
printfn "a is now\n%A" a
```

The corresponding output is

	a1 is now
	[   0.0000   99.0000    2.0000    3.0000    4.0000]
	a is now
	[[   0.0000   99.0000    2.0000    3.0000    4.0000]
	 [   5.0000    6.0000    7.0000    8.0000    9.0000]
	 [  10.0000   11.0000   55.0000   13.0000   14.0000]
	 [  15.0000   16.0000   17.0000   18.0000   19.0000]
	 [  20.0000   21.0000   22.0000   23.0000   24.0000]
	 [  25.0000   26.0000   27.0000   28.0000   29.0000]
	 [  30.0000   31.0000   32.0000   33.0000   34.0000]]

The slicing notation can also be used for changing multiple elements of a tensor at once.
For example

```fsharp
let a2 : ArrayNDHostT<float> = ArrayNDHost.ones [5] 
a.[0, *] <- a2
```

sets all elements of the first row of `a` to 1.
The tensor `a` is now

	[[   1.0000    1.0000    1.0000    1.0000    1.0000]
	 [   5.0000    6.0000    7.0000    8.0000    9.0000]
	 [  10.0000   11.0000   55.0000   13.0000   14.0000]
	 [  15.0000   16.0000   17.0000   18.0000   19.0000]
	 [  20.0000   21.0000   22.0000   23.0000   24.0000]
	 [  25.0000   26.0000   27.0000   28.0000   29.0000]
	 [  30.0000   31.0000   32.0000   33.0000   34.0000]]

### Slicing operations

Consider the two-dimensional tensor `a` of shape $7 \times 5$ and the four-dimensional tensor `b` of shape $1 \times 2 \times 3 \times 4$.
A slice range can be one of the following.

* A range, e.g. `1..3`. Selects the specified elements in the corresponding dimension. For example `a.[1..3, 0..2]` is the $3 \times 3$ sub-tensor of `a` containing rows 1, 2, 3 and columns 0, 1, 2. **The ending index is inclusive.**
* A partial range, e.g. `1..` or `..3`. This selects all elements in the corresponding dimension to the end or from the beginning respectively. Thus `a.[1.., ..3]` is equivalent to `a.[1..6, 0..3]`.
* An asterisk `*`. Selects all elements in the corresponding dimension. For example `a.[1..3, *]` is equivalent to `a.[1..3, 0..4]`.
* An integer. The corresponding dimension collapses, e.g. `a.[*, 0]` specifies a one-dimensional tensor of shape $7$ corresponding to the first column of `a`.
* The special identifier `NewAxis`. It inserts a new axis of size one at the given position. For example `a.[*, NewAxis, *]` produces a view of shape $7 \times 1 \times 5$.
* The special identifier `Fill`. It fills any dimensions not specified (if any) with an asterisk `*`. For example `b.[0, Fill, 2]` is equivalent to `b.[0, *, *, 4]` and results into a two-dimensional view into tensor `b`.

All slice range operators can be combined arbitrarily.

The reader should note that the result of selecting a single element using the slicing operator, e.g. `a.[1,1]`, is a *tensor* of dimension zero sharing the same memory as `a`.

Shape operations
----------------

### Getting the shape
The shape of a tensor can be accessed using the function `ArrayND.shape` or using the `Shape` property.
Both methods return a list.
The rank (number of dimensions) can be accessed using the function `ArrayND.nDims` or with the `NDims` property.
The number of elements can be accessed using the function `ArrayND.nElems` or with the `NElems` property.
For example

```fsharp
printfn "a has shape %A, rank %d and %d elements." a.Shape a.NDims a.NElems
```

prints

	a has shape [7; 5], rank 2 and 35 elements.

### Reshaping
Reshaping changes the shape of the tensor while keeping the number of elements constant.

For example consider the $4 \times 4$ matrix `b`,

```fsharp
let b = ArrayNDHost.initIndexed [4; 4] (fun [y; x] -> 4 * y + x)
```

with value

    [[   0    1    2    3]
     [   4    5    6    7]
     [   8    9   10   11]
     [  12   13   14   15]]

We can use the `ArrayND.reshape` function to transform this matrix into a vector of length 16.

```fsharp
let b1 = ArrayND.reshape [16] b
```

Now `b1` has the value

    [   0    1    2    3    4    5 ...   12   13   14   15]

We can also specify -1 for the new size of at most one dimension.
In this case its size will be chosen automatically (so that the number of elements does not change).
For example

```fsharp
let b2 = ArrayND.reshape [4; 2; -1] b
```

reshapes `b` into a three dimensional tensor of shape $4 \times 2 \times 2$ with the value

    [[[   0    1]
      [   2    3]]
     [[   4    5]
      [   6    7]]
     [[   8    9]
      [  10   11]]
     [[  12   13]
      [  14   15]]]

#### View or copy?
If the tensor to reshape has row-major order (C order), then `ArrayND.reshape` creates a new view into the existing tensor.
Otherwise the tensor is copied during the reshape operation.
If you need to ensure that no copy is performed, i.e. the original and reshaped tensor share the same memory, use the `ArrayND.reshapeView` function instead.
It will raise an error if the original tensor is not in row-major order.


### Reordering axes and transposing
The `ArrayND.swapDim` function creates a new view of a tensor with the given dimensions swapped.
For example

```fsharp
let b3 = ArrayND.swapDim 0 1 b
```

transpose the matrix `b` into

    [[   0    4    8   12]
     [   1    5    9   13]
     [   2    6   10   14]
     [   3    7   11   15]]

The original and tensor with swapped axes share the same memory and modifications made to one of them will affect the other one.
A matrix can also be transposed using the `ArrayND.transpose` function or the `.T` method, i.e. `ArrayND.transpose b` and `b.T` both transpose the matrix `b`.

The `ArrayND.reorderAxes` function can reorder axes arbitrarily.
It takes a list (of length equal to the rank of the tensor) with each element specifying the new position of the corresponding axis.
The list must be a permutation of the axes indices, i.e. duplicating or leaving out axes is not permitted.
Consider the rank three tensor `b2` from above; then

```fsharp
let b4 = ArrayND.reorderAxes [2; 0; 1] b2
```

creates the view `b4` with shape $2 \times 2 \times 4$.
It is important to understand that each list entry specifies where the axis *moves to*, not where it is coming from.
Thus, in this example, axis 0 becomes axis 2, axis 1 becomes axis 0 and axis 2 becomes axis 1.

### Adding axes
The `ArrayND.padLeft` and `ArrayND.padRight` functions add a new axis of size one on the left or right respectively.
If you need to add an axis at another position, use the slicing operator with the `NewAxis` identifier.
For example

```fsharp
let b5 = b4.[*, *, NewAxis, *]
```

creates a view of shape $2 \times 2 \times 1 \times 4$.

### Broadcasting
An axis of size one can be repeated multiple times with the same value.
This is called broadcasting.
Consider the $1 \times 4$ matrix

```fsharp
let c = ArrayNDHost.initIndexed [1; 4] (fun [_; i] -> i) 
```

with value

    [[   0    1    2    3]]

We can use the `ArrayND.broadcastToShape` function,

```fsharp
let c1 = ArrayND.broadcastToShape [3; 4] c
```

to obtain the tensor

    [[   0    1    2    3]
     [   0    1    2    3]
     [   0    1    2    3]]

Broadcasting creates a view of the original tensor, thus the repeated dimensions do not use additional memory and changing the broadcasted view will also change the original as well as all indices of a broadcasted dimension.
Thus, in this example, executing

```fsharp
c1.[[1; 1]] <- 11
printfn "c1 is now\n%A" c1
printfn "c is now\n%A" c
```

prints

    c1 is now
    [[   0   11    2    3]
     [   0   11    2    3]
     [   0   11    2    3]]
    c is now
    [[   0   11    2    3]]

Broadcasting is also performed automatically when performing element-wise operations between two tensors of different, but compatible, shapes.
This will be explained in the section about element-wise tensor operations.

Tensor operations
-----------------

The tensor type supports most standard arithmetic operators and arithmetic functions.

### Element-wise binary arithmetic operators
The elementary operators (`+`, `-`, `*`, `/`, `%`, `**`) are executed element-wise.
For example, consider the vectors `a` and `b`,

```fsharp
let d = ArrayNDHost.initIndexed [4] (fun [i] -> float i)
let e = ArrayNDHost.initIndexed [4] (fun [i] -> 10. * float i)
```

with values

    d = [   0.0000    1.0000    2.0000    3.0000]
    e = [   0.0000   10.0000   20.0000   30.0000]

Then vector `f`,

```fsharp
let f = d + e
```

has the value

    [   0.0000   11.0000   22.0000   33.0000]

It is also possible to apply an operator to a tensor and a scalar of the same data type.
In this case the scalar is broadcasted (repeated) to the size of the given tensor.
For example

```fsharp
let d1 = d * 100.
```

results in

    d1 = [   0.0000  100.0000  200.0000  300.0000]

#### Automatic broadcasting
If a binary operator (for example `+`) is applied to two tensors of different shapes, the library tries to automatically broadcast both tensors to a compatible shape using the following rules.

  1. If the tensors have different ranks, the tensor with the lower rank is padded from the left with axes of size one until both tensors have the same rank.
     For example, if `a` is of shape $4 \times 3 \times 1$ and `b` is of shape $2$, then `b` is padded to the shape $\mathbf{1} \times \mathbf{1} \times 2$.
  1. For each dimension that has different size in both tensors and size one in one of the tensors, this dimension of the tensor with size one is broadcasted to the corresponding dimension of the other tensor.
  Thus, in our example, the last dimension of `a` is broadcasted resulting in the shape $4 \times 3 \times \mathbf{2}$ and the first and second dimensions of `b` are broadcasted resulting in the shape $\mathbf{4} \times \mathbf{3} \times 2$.

If the shapes still differ after applying the above rules, the operation fails.

### Element-wise arithmetic functions
The standard F# arithmetic functions, such as `sin`, `cos`, `exp`, `log`, can also be applied to tensors.
For example

```fsharp
let f2 = sin f
```

calculates the sine of `f` element-wise.


### Element-wise function application
It is also possible to apply an arbitrary function element-wise using the `ArrayND.map` function.
For example

```fsharp
ArrayND.map (fun x -> if x > 15. then 7. + x else -1.) f
```

produces the tensor

    [  -1.0000   -1.0000   29.0000   40.0000]

An in-place variant that overwrites the original tensor is the `ArrayND.mapInplace` function.
The `ArrayND.map2` function takes two tensors and applies a binary function on their elements.


### Arithmetic reduction functions
The `ArrayND.sum` function computes the sum of all elements of a tensor and returns a zero-rank tensor.
Thus, in our example

```fsharp
printfn "The sum of f is %.3f." (ArrayND.sum f |> ArrayND.value)
```

prints

    The sum of f is 66.000.

The `ArrayND.sumAxis` function computes the sum over the given axis.
Thus

```fsharp
let g = ArrayNDHost.initIndexed [4; 4] (fun [y; x] -> 4 * y + x)
ArrayND.sumAxis 0 g
```

computes the sums of all columns of the matrix

    [[   0    1    2    3]
     [   4    5    6    7]
     [   8    9   10   11]
     [  12   13   14   15]]

resulting in the vector

    [  24   28   32   36]

Likewise the `ArrayND.product` and `ArrayND.productAxis` compute the product of a tensor.

### Maximum and minimum
The `ArrayND.min` and `ArrayND.max` compute the minimum and maximum of a tensor and return a zero-rank tensor.
Analogously `ArrayND.minAxis` and `ArrayND.maxAxis` compute the minimum and maximum over the given axis.

### Matrix multiplication
Matrix multiplication (dot product) is implemented using the `.*` operator.
This operator can be used to calculate a vector/vector product resulting in a scalar, a matrix/vector product resulting in a vector and a matrix/matrix product resulting in a matrix.

For example

```fsharp
let h = ArrayNDHost.initIndexed [5; 3] (fun [i; j] -> 3. * float i + float j)
let i = 0.1 + ArrayNDHost.identity 3
let hi = h .* i
```

computes the matrix product of the $5 \times 3$ matrix `h` with the $3 \times 3$ matrix `i` resulting in the `5 \times 3` matrix `hi`.

### Tensor product
The tensor product between two tensors can be obtained using the `%*` operator or the `ArrayND.tensorProduct` function.

### Element-wise comparison operators
Element-wise comparisons are performed using the `====` (element-wise equal), `<<<<` (element-wise smaller than), `>>>>` (element-wise larger than) and `<<>>` (element-wise not equal) operators.
They return a tensor of equal shape and boolean data type.
For example

```fsharp
let j = d ==== e
```

has the value

    d = [true; false; false; false;]


### Logic reduction operations
To check whether all elements satisfy a condition, use the `ArrayND.all` function after applying the element-wise comparison operator.
To check whether at least one element satisfies a condition, use the `ArrayND.any` function after applying the element-wise comparison operator.
Thus 

```fsharp
d ==== e |> ArrayND.all
```

has the value `false`, but

```fsharp
d ==== e |> ArrayND.any
```

returns `true`.

### Element-wise logic operators
Element-wise logic operations are performed using the `~~~~` (element-wise negation), `&&&&` (element-wise and) and `||||` (element-wise or) operators.
Thus
```fsharp
~~~~j
```
has value

    [false; true; true; true;]


Disk storage in HDF5 format
---------------------------

Tensors can be stored in industry-standard [HDF5 files](https://en.wikipedia.org/wiki/Hierarchical_Data_Format).
Multiple tensors can be stored in a single HDF5 file and accessed by assigning names to them.

### Writing tensors to disk
The following code creates two tensors `k` and `l` and writes them into the HDF5 file `tensors.h5` in the current directory.

```fsharp
open Basics // TODO: remove after refactoring
let k = ArrayNDHost.initIndexed [5; 3] (fun [i; j] -> 3. * float i + float j)
let l = ArrayNDHost.initIndexed [5] (fun [i] -> 2. * float i)
let hdfFile = HDF5.OpenWrite "tensors.h5"
ArrayNDHDF.write hdfFile "k" k
ArrayNDHDF.write hdfFile "l" l
hdfFile.Dispose ()
```

The resulting file can be viewed using any HDF5 viewer, for example using the free, cross-platform [HDFView](https://www.hdfgroup.org/products/java/hdfview/) application as shown below.

![HDFView screenshot](img/hdfview.png)

### Loading tensors from disk
The following code loads the tensors `k` and `l` from the previously created HDF5 file `tensors.h5` and stores them in the variables `k2` and `l2`.

```fsharp
let hdfFile2 = HDF5.OpenRead "tensors.h5"
let k2 : ArrayNDHostT<float> = ArrayNDHDF.read hdfFile2 "k"
let l2 : ArrayNDHostT<float> = ArrayNDHDF.read hdfFile2 "l" 
hdfFile2.Dispose ()
```

The data types of `k2` and `l2` must be declared explicitly, since they must be known at compile-time.
If the declared data type does not match the data type encountered in the HDF5, an error will be raised.


Reading .npy and .npz files produced by Numpy
---------------------------------------------

For compatibility, it is possible to read `.npy` and `.npz` files produced by Numpy.
Not all features of the format are supported.
Writing `.npy` and `.npz` files is not possible; use the HDF5 format instead.

Use the `NPYFile.load` function to read an `.npy` file and return its contents as an `ArrayNDHostT`.
Use the `NPZFile.Open` function to open an `.npz` file and the `Get` method of the resulting object to obtain individual entries as `ArrayNDHostT`.

CUDA GPU support
----------------

If your workstation is equipped with a [CUDA](https://en.wikipedia.org/wiki/CUDA) [capable GPU](https://developer.nvidia.com/cuda-gpus), you can transfer tensors to GPU memory and perform operations on the GPU.
GPU tensors are instances of the generic type `ArrayNDCudaT<'T>` where `'T` is the contained data type.

**Note: While tensors can be created on or transferred to the GPU, *currently* there is no supported for accelerated operations on the GPU *when using tensor directly*. Thus executing tensor operations (except slicing and view operations) on the GPU will be very slow *at the moment* and should be avoided. This is supposed to change with future versions and does *not* affect compiled functions created by the Deep.Net library.**

### Data transfer
Tensors can be transferred to the GPU by using the `ArrayNDCuda.toDev` function.
Transfer back to host memory is done using the `ArrayNDCuda.toHost` function.

```fsharp
let m = seq {1 .. 10} |> ArrayNDHost.ofSeq
let mGpu = ArrayNDCuda.toDev m
```

`mGpu` is now a copy of `m` in GPU memory.

### Tensor creation
It is possible to create tensors directly in GPU memory.
The function `ArrayNDCuda.zeros`, `ArrayNDCuda.ones`, `ArrayNDCuda.identity` and `ArrayNDCuda.scalar` behave like their `ArrayNDHost` counterparts, except that the result is stored on the GPU.

```fsharp
let n : ArrayNDCudaT<float> = ArrayNDCuda.identity 4
```

Here we created a $4 \times 4$ identity matrix of data type `float` on the GPU.

### Operators and functions
All functions and operators described in previous section can be equally applied to GPU tensors.
For example, the code

```fsharp
let mGpuSq = mGpu.[3..5] * 3
```

takes three elements from the GPU tensor and multiplies them by 3 on the GPU.

Conclusion
----------

We presented an introduction to working with host and GPU tensors in Deep.Net.
Direct tensor manipulation will be used mostly for dataset and results handling.
The machine learning model will instead be defined as a symbolic computation graph, that supports automatic differentiation and compilation to optimized CUDA code.

