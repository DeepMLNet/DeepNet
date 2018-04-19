# Elements and slicing

The tensor library provides methods to access single elements of a tensor.
Furthermore, it provides methods to access and manipulate ranges consisting of multiple elements within a tensor at once.

This page provides an overview of both concepts.
More information is available at the [Item property](xref:Tensor.Tensor`1.Item*).

## Accessing individual elements

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
The above example accesses the element at index *1,1*.
Note that the indices are specified as 64-bit integers surrounded by double brackets (`[[` and `]]`) and separated using a semicolon.

Tensors are mutable objects.
An element can be changed using the `tensor.[[idx0; idx1; ...; idxN]] <- newValue` notation.
```fsharp
a.[[2L; 2L]] <- 55.0
// a =
//    [[   0.0000    1.0000    2.0000    3.0000    4.0000]
//     [   5.0000    6.0000    7.0000    8.0000    9.0000]
//     [  10.0000   11.0000   55.0000   13.0000   14.0000]
//     [  15.0000   16.0000   17.0000   18.0000   19.0000]
//     [  20.0000   21.0000   22.0000   23.0000   24.0000]
//     [  25.0000   26.0000   27.0000   28.0000   29.0000]
//     [  30.0000   31.0000   32.0000   33.0000   34.0000]]
```
The above example changes the value at index *2,2* to 55.

## Slicing

Slicing creates a new view into an existing tensor.
A view shares the same memory with the original tensor and thus any modification affects both the original tensor and all its views.

Slicing is done using the `tensor.[rng0, rng1, ..., rngN]` notation.
Note that the ranges are specified within single brackets and separated using commas.

Let us select the first row of tensor `a`.
```fsharp
let a1 = a.[0L, *]
// a1 = [   0.0000    1.0000    2.0000    3.0000    4.0000]
```

The asterisk (`*`) selects all elements of the corresponding dimension.
Thus in this case the result is a tensor of rank one (i.e. a vector) as seen above.

Since `a1` is a view of `a` it shares the same memory.
Hence, changing an element of `a1` by assigning a new value to it changes the tensor `a` as well.

```fsharp
a1.[[1L]] <- 99.0
// a1 = [   0.0000   99.0000    2.0000    3.0000    4.0000]
// a =
//	[[   0.0000   99.0000    2.0000    3.0000    4.0000]
//	 [   5.0000    6.0000    7.0000    8.0000    9.0000]
//	 [  10.0000   11.0000   55.0000   13.0000   14.0000]
//	 [  15.0000   16.0000   17.0000   18.0000   19.0000]
//	 [  20.0000   21.0000   22.0000   23.0000   24.0000]
//	 [  25.0000   26.0000   27.0000   28.0000   29.0000]
//	 [  30.0000   31.0000   32.0000   33.0000   34.0000]]
```

Slicing can also be used to change multiple elements of a tensor at once.
For example the following code sets all elements of the first row of `a` to 1.
```fsharp
let a2 = HostTensor.ones<float> [5L]
// a2 = [1.0  1.0  1.0  1.0  1.0]
a.[0L, *] <- a2
// a =
//	[[   1.0000    1.0000    1.0000    1.0000    1.0000]
//	 [   5.0000    6.0000    7.0000    8.0000    9.0000]
//	 [  10.0000   11.0000   55.0000   13.0000   14.0000]
//	 [  15.0000   16.0000   17.0000   18.0000   19.0000]
//	 [  20.0000   21.0000   22.0000   23.0000   24.0000]
//	 [  25.0000   26.0000   27.0000   28.0000   29.0000]
//	 [  30.0000   31.0000   32.0000   33.0000   34.0000]]
```


### Slice specification

Consider the two-dimensional tensor `a` of shape *7x5* and the four-dimensional tensor `b` of shape *1x2x3x4*.
A slice range can be one of the following.

* **A range, e.g. `1L..3L`.**
Selects the specified elements in the corresponding dimension.
For example `a.[1L..3L, 0L..2L]` is the *3x3* sub-tensor of `a` containing rows *1,2,3* and columns *0,1,2*.
As it is standard in F#, the ending index is inclusive.
* **A partial range, e.g. `1L..` or `..3L`.**
This selects all elements in the corresponding dimension to the end or from the beginning respectively. 
Thus `a.[1L.., ..3L]` is equivalent to `a.[1L..6L, 0L..3L]`.
* **An asterisk `*`.**
Selects all elements in the corresponding dimension.
For example `a.[1L..3L, *]` is equivalent to `a.[1L..3L, 0L..4L]`.
* **An 64-bit integer.**
The corresponding dimension collapses, e.g. `a.[*, 0L]` specifies a one-dimensional tensor of shape *7* corresponding to the first column of `a`.
* **The special identifier `NewAxis`.**
It inserts a new axis of size one at the given position. 
For example `a.[*, NewAxis, *]` produces a view of shape *7x1x5*.
* **The special identifier `Fill`.**
It fills any dimensions not specified (if any) with an asterisk `*`.
For example `b.[0L, Fill, 2L]` is equivalent to `b.[0L, *, *, 4L]` and results into a two-dimensional view into tensor `b`.

All slice ranges can be combined arbitrarily.

The result of selecting a single element using the slicing operator, e.g. `a.[1L,1L]` is a *tensor* of dimension zero referencing a single element inside `a`.


