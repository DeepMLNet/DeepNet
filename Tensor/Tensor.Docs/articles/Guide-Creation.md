# Creating and transferring tensors

To work with the Tensor library, install the NuGet packages as described in the [installation guide](Guide-Installation.md) and open the `Tensor` namespace within your source file.
You can run the following examples by pasting the code into the `main` function in `Program.fs` of the skeleton project.

You can also clone the official example project, which follows this guide, from <https://github.com/DeepMLNet/Tensor.Sample>.

The primary type you will work with is [Tensor<'T>](xref:Tensor.Tensor`1).
It provides functions to work with tensors regardless of their storage device.
The modules [HostTensor](xref:Tensor.HostTensor) and [CudaTensor](xref:Tensor.CudaTensor) contain additional functions that are only applicable to tensors stored in host or GPU memory respectively.

## Creating a tensor filled with zeros

Let us create a *3x2* matrix, i.e. a two-dimensional tensor, of data type `int` filled with zeros in host memory.
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
In this case we specified [HostTensor.Dev](xref:Tensor.HostTensor.Dev()) to store the tensor in host memory.
The second argument specifies the desired shape.
All shapes and indices in this tensor library are 64-bit integers.
Thus we have to use the `L` postfix when writing integer literals, i.e. `3L` instead of `3`.

Since creating tensors in host memory is a very common operation, the library also provides the shorter notation shown below to perform the same task.
```fsharp
let z1 = HostTensor.zeros<int> [3L; 2L]
// z1 = [[0; 0]
//       [0; 0]
//       [0; 0]]
```
These shorthands are available for all tensor creation function and listed in the [HostTensor](xref:Tensor.HostTensor) module.

## Other initialization possibilities

Similarly, we can use the [Tensor<'T>.ones](xref:Tensor.Tensor`1.ones*) function to obtain a vector of data type `single` and size `3` filled with ones.
```fsharp
let o1 = Tensor<single>.ones HostTensor.Dev [3L]
// o1 = [0.0f; 0.0f; 0.0f]
```
The [Tensor<'T>.identity](xref:Tensor.Tensor`1.identity*) function creates an identity matrix of the given size.
```fsharp
let id1 = Tensor<float>.identity HostTensor.Dev 3L
// id1 = [[1.0; 0.0; 0.0]
//        [0.0; 1.0; 0.0]
//        [0.0; 0.0; 1.0]]
```
This created a *3x3* identity matrix.

## Scalar tensors
A scalar tensor is a tensor that has a dimensionality of zero.
It contains exactly one element and can be treated like a tensor of any other dimensionality.
However, for convenience, special functions are provided to make working with scalar tensors easier.

A scalar tensor can be created with the [Tensor.scalar](xref:Tensor.Tensor`1.scalar*) function (or its corresponding [HostTensor.scalar](xref:Tensor.HostTensor.scalar*) shorthand).
```fsharp
let s1 = Tensor.scalar HostTensor.Dev 33.2
// s1 = 33.2
// s1.NDims = 0
// s1.Shape = []
// s1.NElems = 1L
```
Specifying an empty shape when using other creation methods, such as [Tensor<'T>.zeros](xref:Tensor.Tensor`1.zeros*), will also create a scalar tensor.

The numeric value of a scalar tensor can be obtained (and changed) using the [Value](xref:Tensor.Tensor`1.Value*) property.
```fsharp
printfn "The numeric value of s1 is %f." s1.Value
// The numeric value of s1 is 33.2.
```
If you try to use this property on a non-scalar tensor, an exception will be raised.

## Host-only creation methods
Some tensor creation methods can only produce tensors stored in host memory, which, of course, can be transferred to GPU memory subsequently.
For example the [HostTensor.init](xref:Tensor.HostTensor.init*) function takes a function and uses it to compute the initial value of each element of the tensor.
```fsharp
let a = HostTensor.init [7L; 5L] (fun [|i; j|] -> 5.0 * float i + float j)
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
The second argument is a function that takes the n-dimensional index (zero-based) of an entry and computes its initial value; here we use the formula *5i + j* where *i* is the row and *j* is the column of the matrix.
The data type (here `float`) is automatically inferred from the return type of the initialization function.

## Creation from F# sequences, lists and arrays
The [HostTensor.ofSeq](xref:Tensor.HostTensor.ofSeq*) converts an [F# sequence](https://en.wikibooks.org/wiki/F_Sharp_Programming/Sequences) of finite length into a one-dimensional tensor.
```fsharp
let seq1 = seq { for i=0 to 20 do if i % 3 = 0 then yield i } |> HostTensor.ofSeq
// seq1 = [   0    3    6    9   12   15   18]
```
The example above creates a vector of all multiplies of 3 in the range between 0 and 20.

A list can be converted into a one-dimensional tensor using the [HostTensor.ofList](xref:Tensor.HostTensor.ofList*) function.
To convert an array into a tensor use the [HostTensor.ofArray](xref:Tensor.HostTensor.ofArray*) function.
The [HostTensor.ofList2D](xref:Tensor.HostTensor.ofList2D*) and [HostTensor.ofArray2D](xref:Tensor.HostTensor.ofArray2D*) take two-dimensional lists or arrays and convert them into tensors of respective shapes.

## Conversion to F# sequences, lists and arrays
Use the [HostTensor.toSeq](xref:Tensor.HostTensor.toSeq*) function to expose the elements of a tensor as a sequence.
If the tensor has more than one dimension, it is flattened before the operation is performed.

Use the [HostTensor.toList](xref:Tensor.HostTensor.toList*) or [HostTensor.toList2D](xref:Tensor.HostTensor.toList2D*) functions to convert a tensor into a list.
The [HostTensor.toArray](xref:Tensor.HostTensor.toArray*), [HostTensor.toArray2D](xref:Tensor.HostTensor.toArray2D*), [HostTensor.toArray3D](xref:Tensor.HostTensor.toArray3D*) convert a tensor into an array of respective dimensionality.

All these operations copy the elements of the tensor.

## Printing tensors and string representation

Tensors can be printed using the `%A` format specifier of the standard `printf` function.
```fsharp
printfn "The tensor seq1 is\n%A" seq1
// The tensor seq1 is
// [   0    3    6    9   12   15   18]
```
The output of large tensors is automatically truncated to a reasonable size.
The corresponding string representation can also be accessed thorugh the [Pretty](xref:Tensor.Tensor`1.Pretty*) property.
The full (untruncated) string representation is available through the [Full](xref:Tensor.Tensor`1.Full*) property.
Use the [ToString](xref:Tensor.Tensor`1.ToString*) method when it is required to adjust the maximum number of elements that are printed before truncation occurs.

## Transferring tensors to the GPU

If your workstation is equipped with a CUDA capable GPU, you can transfer tensors to GPU memory and perform operations on the GPU.

Tensors can be transferred to the GPU by using the [CudaTensor.transfer](xref:Tensor.CudaTensor.transfer*) function.
Transfer back to host memory is done using the [HostTensor.transfer](xref:Tensor.HostTensor.transfer*) function.

```fsharp
let m = seq {1 .. 10} |> HostTensor.ofSeq
// m.Dev = HostTensor.Dev
let mGpu = CudaTensor.transfer m
// mGpu.Dev = CudaTensor.Dev
```

The above sample code creates tensor `m` in host memory and then creates the copy `mGpu` in GPU memory.
All operations performed on `mGpu` will execute directly on the GPU.

If you receive an error message when trying to perform GPU operations, read the [troubleshooting guide](Troubleshooting.md) to get help.
