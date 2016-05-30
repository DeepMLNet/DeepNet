(*** hide ***)
#load "../../DeepNet.fsx"

(**

Working with Tensors
====================

In Deep.Net a *tensor* is an n-dimensional array of an arbitrary data type (for example `single`, `double` or `System.Numerics.Complex`).
Tensors are implemented by the `ArrayNDT<'T>` class and its derivatives `ArrayNDHostT` (for tensors stored in host memory) and `ArrayNDCudaT` (for tensors stored in CUDA GPU memory).
The Deep.Net tensor provides functionality similar to [Numpy's Ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.html) and [MATLAB arrays](http://www.mathworks.com/help/matlab/matrices-and-arrays.html), including vector-wise operations, reshaping, slicing, broadcasting, masked assignment, reduction operations and BLAS operations.
The API for host and GPU stored tensors is equal, thus a program can make use of GPU accelerated operations without much porting effort.

You can run this example by executing `FsiAnyCPU.exe docs\content\tensor.fsx` after cloning the Deep.Net repository.
You can move your mouse over any symbol in the code samples to see the full signature.


Architecture
------------

To work with the tensor library, open the `ArrayNDNS` namespace.
*)

open ArrayNDNS

(**
The tensor library consists of three primary modules: ArrayND, ArrayNDHost, ArrayNDCuda.
ArrayND contains functions to work with existing tensors (regardless of their storage location) such as `reshape`, `sum`, and `copy`.
ArrayNDHost contains functions to create new tensors in host memory, for example `zeros`, `ones` and `ofList`.
ArrayNDCuda contains functions to create new tensors in CUDA GPU memory and to facilitate the transfer of tensors between host and GPU memory, for example `zeros`, `toDev` and `fromDev`.


Creating tensors
----------------

Let us create a simple $7 \times 5$ matrix, i.e. two-dimensional tensor, in host memory.
*)

let a = ArrayNDHost.initIndexed [7; 5] (fun [y; x] -> 5.0 * float y + float x)

(**
The first argument to the `ArrayNDHost.initIndexed` function is an [F# list](https://en.wikibooks.org/wiki/F_Sharp_Programming/Lists) specifies the shape of the tensor.
The second argument is a function that takes the n-dimensional index (zero-based) of an entry and computes its initial value; here we use the formula $5y + x$ where $y$ is the row and $x$ is the column of the matrix. 
The data type (here float) is automatically inferred from the return type of the initialization function.
By default row-major storage (C storage order) is used. 

The resulting tensor has the following entries.

    [[   0.0000    1.0000    2.0000    3.0000    4.0000]
     [   5.0000    6.0000    7.0000    8.0000    9.0000]
     [  10.0000   11.0000   12.0000   13.0000   14.0000]
     [  15.0000   16.0000   17.0000   18.0000   19.0000]
     [  20.0000   21.0000   22.0000   23.0000   24.0000]
     [  25.0000   26.0000   27.0000   28.0000   29.0000]
     [  30.0000   31.0000   32.0000   33.0000   34.0000]]

We can also create a tensor filled with zeros using the `ArrayNDHost.zeros` function.
*)

let z1 : ArrayNDHostT<int> = ArrayNDHost.zeros [3]

(**
In this case we must specify the data type (int) explicitly, since there is no information in the function call to infer it automatically.

We can use the `ArrayNDHost.ones` function to obtain a tensor filled with ones.
*)

let o1 : ArrayNDHostT<single> = ArrayNDHost.ones [3]

(**
The `ArrayNDHost.identity` function creates an identity matrix of the given size.
*)
let id1 : ArrayNDHostT<float> = ArrayNDHost.identity 6
(**
This created a $6 \times 6$ identity matrix.

A tensor of rank zero can be created with the `ArrayNDHost.scalar` function.
*)
let s1 = ArrayNDHost.scalar 33.2
(**
The numeric value of a zero-rank tensor can be extracted using the `ArrayND.value` function.
*)
printfn "The numeric value of s1 is %f." (ArrayND.value s1)

(**
Printing tensors
----------------

Tensors can be printed using the `%A` format specifier of the standard `printf` function.
*)

printfn "The tensor a is\n%A" a
printfn "The tensor z1 is\n%A" z1
printfn "The tensor o1 is\n%A" o1
printfn "The tensor id1 is\n%A" id1

(**
The output of large tensors is automatically truncated to a reasonable size.


Accessing elements
------------------

Individual elements of a tensor can be accessed using the `tensor.[[idx0; idx1; ...; idxN]]` notation.
Zero-based indexing is used.
For example
*)
a.[[1; 1]]
(**
accesses the element at index $1,1$ and returns `5.0`.
Note that the indices are specified with double brackets ([[ and ]]) and separated using a semicolon.

Tensors are mutable objects.
An element can be changed using the `tensor.[[idx0; idx1; ...; idxN]] <- newValue` notation.
For example
*)
a.[[2; 2]] <- 55.
(**
changes the tensor to

    [[   0.0000    1.0000    2.0000    3.0000    4.0000]
     [   5.0000    6.0000    7.0000    8.0000    9.0000]
     [  10.0000   11.0000   55.0000   13.0000   14.0000]
     [  15.0000   16.0000   17.0000   18.0000   19.0000]
     [  20.0000   21.0000   22.0000   23.0000   24.0000]
     [  25.0000   26.0000   27.0000   28.0000   29.0000]
     [  30.0000   31.0000   32.0000   33.0000   34.0000]]


Slicing
-------

Slicing creates a new view into an existing tensor.
Slicing is done using the `tensor.[rng0, rng1, ..., rngN]` notation.
Note that the ranges are specified within single brackets and separated using commas.

Let us select the first row of tensor a.
*)
let a1 = a.[0, *]
(**
The asterisk selects all elements of the specified dimension.
The result is a tensor of rank one (i.e. a vector) with the entries

    [   0.0000    1.0000    2.0000    3.0000    4.0000]

Since `a1` is a view of `a` it shares the same memory.
Changing an element of `a1` by assigning a new value to it
*)
a1.[[1]] <- 99.
(**
changes the tensor `a` as well.
This can be seen by outputting `a1` and `a`.
*)
printfn "a1 is now\n%A" a1
printfn "a is now\n%A" a
(**
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
*)
let a2 : ArrayNDHostT<float> = ArrayNDHost.ones [5] 
a.[0, *] <- a2
(**
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

The reader should note that the result of selecting a single element using the slicing operator, e.g. $a.[1,1]$, is a *tensor* of dimension zero sharing the same memory as `a`.


Shape operations
----------------

### Getting the shape
The shape of a tensor can be accessed using the function `ArrayND.shape` or using the `Shape` property.
Both methods return a list.
The rank (number of dimensions) can be accessed using the function `ArrayND.nDims` or with the `NDims` property.
The number of elements can be accessed using the function `ArrayND.nElems` or with the `NElems` property.
For example
*)
printfn "a has shape %A, rank %d and %d elements." a.Shape a.NDims a.NElems
(**
prints

	a has shape [7; 5], rank 2 and 35 elements.

### Reshaping
Reshaping changes the shape of the tensor while keeping the number of elements constants.
Use the `ArrayND.reshape` function to create a new view of a tensor with a different shape.
*)
