
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

