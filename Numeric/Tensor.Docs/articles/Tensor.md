# Tensor
This page provides an overview of most commonly used tensor functions by category.

For a complete, alphabetical reference of all tensor functions see [Tensor<'T>](xref:Tensor.Tensor`1) and the device-specific functions in [HostTensor](xref:Tensor.HostTensor) and [CudaTensor](xref:Tensor.CudaTensor).

## Creation functions
Use these functions to create a new tensor.

Function | Description
-------- | -----------
[arange](xref:Tensor.Tensor`1.arange*) | Creates a new vector filled with equaly spaced values using a specifed increment.
[counting](xref:Tensor.Tensor`1.counting*) | Creates a new vector filled with the integers from zero to the specified maximum.
[empty](xref:Tensor.Tensor`1.empty*) | Creates a new, empty tensor with the given number of dimensions.
[falses](xref:Tensor.Tensor`1.falses*) | Creates a new boolean tensor filled with falses.
[filled](xref:Tensor.Tensor`1.filled*) | Creates a new tensor filled with the specified value.
[identity](xref:Tensor.Tensor`1.identity*) | Creates a new identity matrix.
[ones](xref:Tensor.Tensor`1.ones*) | Creates a new tensor filled with ones (1).
[onesLike](xref:Tensor.Tensor`1.onesLike*) | Creates a new tensor filled with ones using the specified tensor as template.
[linspace](xref:Tensor.Tensor`1.linspace*) | Creates a new vector of given size filled with equaly spaced values.
[scalar](xref:Tensor.Tensor`1.scalar*) | Creates a new zero-dimensional (scalar) tensor with the specified value.
[scalarLike](xref:Tensor.Tensor`1.scalarLike*) | Creates a new zero-dimensional (scalar) tensor using the specified tensor as template and with  the specified value.
[trues](xref:Tensor.Tensor`1.trues*) | Creates a new boolean tensor filled with trues.
[zeros](xref:Tensor.Tensor`1.zeros*) | Creates a new tensor filled with zeros (0).
[zerosLike](xref:Tensor.Tensor`1.zerosLike*) | Creates a new tensor filled with zeros using the specified tensor as template.


## Slicing and element access functions
Use these functions to slice tensors or access individual elements of them.

Function | Description
-------- | -----------
[Item](xref:Tensor.Tensor`1.Item*) | Accesses a slice (part) of the tensor.
[M](xref:Tensor.Tensor`1.M*) | Picks elements from a tensor using one or more boolean mask tensors.
[Value](xref:Tensor.Tensor`1.value*) | Gets the value of a zero-dimensional (scalar) tensor.


## Element-wise operations
These mathematical operations are applied element-wise to each element of the tensor(s).

Function | Description
-------- | -----------
[( ~- )](xref:Tensor.Tensor`1.op_UnaryNegation*) | Element-wise negation.
[( + )](xref:Tensor.Tensor`1.op_Addition*) | Element-wise addition.
[( - )](xref:Tensor.Tensor`1.op_Subtraction*) | Element-wise substraction.
[( * )](xref:Tensor.Tensor`1.op_Multiply*) | Element-wise multiplication.
[( / )](xref:Tensor.Tensor`1.op_Division*) | Element-wise division.
[( % )](xref:Tensor.Tensor`1.op_Modulus*) | Element-wise remainder of division.
[Abs](xref:Tensor.Tensor`1.Abs*) | Element-wise absolute value.
[Acos](xref:Tensor.Tensor`1.Acos*) | Element-wise arccosine (inverse cosine).
[Asin](xref:Tensor.Tensor`1.Asin*) | Element-wise arcsine (inverse sine).
[Atan](xref:Tensor.Tensor`1.Atan*) | Element-wise arctanget (inverse tangent).
[Ceiling](xref:Tensor.Tensor`1.Ceiling*) | Element-wise ceiling (round towards positive infinity).
[Cos](xref:Tensor.Tensor`1.Cos*) | Element-wise cosine.
[Cosh](xref:Tensor.Tensor`1.Cosh*) | Element-wise hyperbolic cosine.
[Exp](xref:Tensor.Tensor`1.Exp*) | Element-wise exponential function.
[Floor](xref:Tensor.Tensor`1.Floor*) | Element-wise floor (round towards negative infinity).
[Log](xref:Tensor.Tensor`1.Log*) | Element-wise natural logarithm.
[Log10](xref:Tensor.Tensor`1.Log10*) | Element-wise common logarithm.
[Pow](xref:Tensor.Tensor`1.Pow*) | Element-wise exponentiation.
[Round](xref:Tensor.Tensor`1.Round*) | Element-wise rounding.
[Sgn](xref:Tensor.Tensor`1.Sgn*) | Element-wise sign.
[Sin](xref:Tensor.Tensor`1.Sin*) | Element-wise sine.
[Sinh](xref:Tensor.Tensor`1.Sinh*) | Element-wise hyperbolic sine.
[Sqrt](xref:Tensor.Tensor`1.Sqrt*) | Element-wise square root.
[Tan](xref:Tensor.Tensor`1.Tan*) | Element-wise tangent.
[Tanh](xref:Tensor.Tensor`1.Tanh*) | Element-wise hyperbolic tangent.
[Truncate](xref:Tensor.Tensor`1.Truncate*) | Element-wise truncation (rounding towards zero).


## Tensor operations
These functions perform various operations on one or more tensors.

Function | Description
-------- | -----------
[concat](xref:Tensor.Tensor`1.concat*) | Concatenates tensors along an axis.
[copy](xref:Tensor.Tensor`1.copy*) | Returns a copy of the tensor.
[diag](xref:Tensor.Tensor`1.diag*) | Returns a view of the diagonal of the matrix.
[diagAxis](xref:Tensor.Tensor`1.diagAxis*) | Returns a view of the diagonal along the given axes.
[diagMat](xref:Tensor.Tensor`1.diagMat*) | Creates a matrix with the specified diagonal.
[diagMatAxis](xref:Tensor.Tensor`1.diagMatAxis*) | Creates a tensor with the specified diagonal along the given axes.
[diff](xref:Tensor.Tensor`1.diff*) | Calculates the difference between adjoining elements of the vector.
[diffAxis](xref:Tensor.Tensor`1.diffAxis*) | Calculates the difference between adjoining elements along the specified axes.
[ofBlocks](xref:Tensor.Tensor`1.ofBlocks*) | Builds a tensor out of tensor blocks.
[replicate](xref:Tensor.Tensor`1.replicate*) | Repeats the tensor along an axis.
[T](xref:Tensor.Tensor`1.T*) | Transpose of a matrix.


## Linear algebra functions
Use these functions to perform basic linear algebra operations on tensors.

Function | Description
-------- | -----------
[( .* )](xref:Tensor.Tensor`1.op_DotMultiply*) | Computes the (batched) matrix product, (batched) matrix-vector product or scalar product.
[norm](xref:Tensor.Tensor`1.norm*) | Calculates the norm of the (flattened) tensor.
[normAxis](xref:Tensor.Tensor`1.normAxis*) | Calculates the norm along the specified axis.
[invert](xref:Tensor.Tensor`1.invert*) | (Batch) inverts a matrix.
[pseudoInvert](xref:Tensor.Tensor`1.pseudoInvert*) | Computes the (batched) Moore-Penrose pseudo-inverse of the specified matrix.
[SVD](xref:Tensor.Tensor`1.SVD*) | Computes the (batched) singular value decomposition (SVD) of the specified matrix.
[symmetricEigenDecomposition](xref:Tensor.Tensor`1.symmetricEigenDecomposition*) | Computes the (real) eigendecomposition of a symmetric matrix.
[tensorProduct](xref:Tensor.Tensor`1.tensorProduct*) | Computes the tensor product between two tensors.


## Shape functions
Use these functions to work with the shape and memory layout of a tensor.

Function | Description
-------- | -----------
[atLeastND](xref:Tensor.Tensor`1.atLeastND*) | Pads the tensor from the left with size-one dimensions until it has at least the specified number of dimensions.
[broadcastDim](xref:Tensor.Tensor`1.broadcastDim*) | Broadcast a dimension to a specified size.
[broadcastTo](xref:Tensor.Tensor`1.broadcastTo*) | Broadcasts the specified tensor to the specified shape.
[broadcastToSame](xref:Tensor.Tensor`1.broadcastToSame*) | Broadcasts all specified tensors to have the same shape.
[broadcastToSameInDims](xref:Tensor.Tensor`1.broadcastToSameInDims*) | Broadcasts all specified tensors to have the same size in the specified dimensions.
[cutLeft](xref:Tensor.Tensor`1.cutLeft*) | Removes the first dimension.
[cutRight](xref:Tensor.Tensor`1.cutRight*) | Removes the last dimension.
[flatten](xref:Tensor.Tensor`1.flatten*) | Flattens the tensor into a (one-dimensional) vector.
[insertAxis](xref:Tensor.Tensor`1.insertAxis*) | Insert a dimension of size one before the specifed dimension.
[isBroadcasted](xref:Tensor.Tensor`1.isBroadcasted*) | Checks if the specified tensor is broadcasted in at least one dimension.
[Layout](xref:Tensor.Tensor`1.layout*) | Memory layout of the tensor.
[NDims](xref:Tensor.Tensor`1.nDims*) | Dimensionality of the tensor.
[NElems](xref:Tensor.Tensor`1.nElems*) | Total number of elements within the tensor.
[padLeft](xref:Tensor.Tensor`1.padLeft*) | Insert a dimension of size one as the first dimension.
[padRight](xref:Tensor.Tensor`1.padRight*) | Append a dimension of size one after the last dimension.
[padToSame](xref:Tensor.Tensor`1.padToSame*) | Pads all specified tensors from the left with dimensions of size one until they have the  same dimensionality.
[permuteAxes](xref:Tensor.Tensor`1.permuteAxes*) | Permutes the axes as specified.
[reshape](xref:Tensor.Tensor`1.reshape*) | Changes the shape of a tensor.
[reverseAxis](xref:Tensor.Tensor`1.reverseAxis*) | Reverses the elements in the specified dimension.
[relayout](xref:Tensor.Tensor`1.relayout*) | Creates a tensor with the specified layout sharing its storage with the original tensor.
[Shape](xref:Tensor.Tensor`1.shape*) | Shape of the tensor.
[swapDim](xref:Tensor.Tensor`1.swapDim*) | Swaps the specified dimensions of the tensor.


## Data type functions
Use these functions to query or change the data type of the elements of a tensor.

Function | Description
-------- | -----------
[convert](xref:Tensor.Tensor`1.convert*) | Convert the elements of a tensor to the specifed type.
[DataType](xref:Tensor.Tensor`1.dataType*) | Type of data stored within the tensor.


## Device and storage functions
Use these functions to query or change the storage device of a tensor.

Function | Description
-------- | -----------
[Dev](xref:Tensor.Tensor`1.dev*) | Device the data of tensor is stored on.
[Storage](xref:Tensor.Tensor`1.Storage*) | The storage object that holds the data of this tensor.
[transfer](xref:Tensor.Tensor`1.transfer*) | Transfers a tensor to the specifed device.


## Comparison functions
Use these functions to perform comparisons of tensors. The results are mostly boolean tensors.

Function | Description
-------- | -----------
[( ==== )](xref:Tensor.Tensor`1.op_EqualsEqualsEqualsEquals*) | Element-wise equality test.
[( <<<< )](xref:Tensor.Tensor`1.op_LessLessLessLess*) | Element-wise less-than test.
[( <<== )](xref:Tensor.Tensor`1.op_LessLessEqualsEquals*) | Element-wise less-than-or-equal test.
[( >>>> )](xref:Tensor.Tensor`1.op_GreaterGreaterGreaterGreater*) | Element-wise greater-than test.
[( >>== )](xref:Tensor.Tensor`1.op_GreaterGreaterEqualsEquals*) | Element-wise greater-than-or-equal test.
[( <<>> )](xref:Tensor.Tensor`1.op_LessLessGreaterGreater*) | Element-wise not-equality test.
[almostEqual](xref:Tensor.Tensor`1.almostEqual*) | Checks if two tensors have the same (within machine precision) values in all elements.
[isClose](xref:Tensor.Tensor`1.isClose*) | Element-wise check if two tensors have same (within machine precision) values.
[isFinite](xref:Tensor.Tensor`1.isFinite*) | Element-wise finity check (not -Inf, Inf or NaN).
[maxElemwise](xref:Tensor.Tensor`1.maxElemwise*) | Element-wise maximum.
[minElemwise](xref:Tensor.Tensor`1.minElemwise*) | Element-wise minimum.
[allFinite](xref:Tensor.Tensor`1.allFinite*) | Checks that all elements of the tensor are finite.


## Logical functions
Use these functions to work with boolean tensors.

Function | Description
-------- | -----------
[( ~~~~ )](xref:Tensor.Tensor`1.op_TwiddleTwiddleTwiddleTwiddle*) | Element-wise logical negation.
[( &&&& )](xref:Tensor.Tensor`1.op_AmpAmpAmpAmp*) | Element-wise loigcal and.
[( \|\|\|\| )](xref:Tensor.Tensor`1.op_BarBarBarBar*) | Element-wise loigcal or.
[( ^^^^ )](xref:Tensor.Tensor`1.op_HatHatHatHat*) | Element-wise loigcal xor.
[all](xref:Tensor.Tensor`1.all*) | Checks if all elements of the tensor are true.
[allAxis](xref:Tensor.Tensor`1.allAxis*) | Checks if all elements along the specified axis are true.
[allElems](xref:Tensor.Tensor`1.allElems*) | Gets a sequence of all all elements within the tensor.
[allTensor](xref:Tensor.Tensor`1.allTensor*) | Checks if all elements of the tensor are true returning the result as a tensor.
[any](xref:Tensor.Tensor`1.any*) | Checks if any elements of the tensor are true.
[anyAxis](xref:Tensor.Tensor`1.anyAxis*) | Checks if any element along the specified axis is true.
[anyTensor](xref:Tensor.Tensor`1.anyTensor*) | Checks if any element of the tensor is true returning the result as a tensor.
[ifThenElse](xref:Tensor.Tensor`1.ifThenElse*) | Element-wise choice between two sources depending on a condition.


## Index functions
These functions return tensor of indices or work with them.

Function | Description
-------- | -----------
[allIdx](xref:Tensor.Tensor`1.allIdx*) | Gets a sequence of all indices to enumerate all elements within the tensor.
[allIdxOfDim](xref:Tensor.Tensor`1.allIdxOfDim*) | Gets a sequence of all indices to enumerate all elements of the specified dimension of the tensor.
[argMax](xref:Tensor.Tensor`1.argMax*) | Finds the indicies of the maximum value of the tensor.
[argMaxAxis](xref:Tensor.Tensor`1.argMaxAxis*) | Finds the index of the maximum value along the specified axis.
[argMin](xref:Tensor.Tensor`1.argMin*) | Finds the indicies of the minimum value of the tensor.
[argMinAxis](xref:Tensor.Tensor`1.argMinAxis*) | Finds the index of the minimum value along the specified axis.
[find](xref:Tensor.Tensor`1.find*) | Finds the first occurence of the specfied value and returns its indices.
[findAxis](xref:Tensor.Tensor`1.findAxis*) | Finds the first occurence of the specfied value along the specified axis and returns its index.
[gather](xref:Tensor.Tensor`1.gather*) | Selects elements from a tensor according to specified indices.
[scatter](xref:Tensor.Tensor`1.scatter*) | Disperses elements from a source tensor to a new tensor according to the specified indices.
[trueIdx](xref:Tensor.Tensor`1.trueIdx*) | Finds the indices of all element that are true.


## Reduction functions
These functions perform operations on tensors that reduce their dimensionality.

Function | Description
-------- | -----------
[countTrue](xref:Tensor.Tensor`1.countTrue*) | Counts the elements being true.
[countTrueAxis](xref:Tensor.Tensor`1.countTrueAxis*) | Counts the elements being true along the specified axis.
[max](xref:Tensor.Tensor`1.max*) | Calculates the maximum of all elements.
[maxAxis](xref:Tensor.Tensor`1.maxAxis*) | Calculates the maximum value of the elements along the specified axis.
[min](xref:Tensor.Tensor`1.min*) | Calculates the minimum of all elements.
[minAxis](xref:Tensor.Tensor`1.minAxis*) | Calculates the minimum value of the elements along the specified axis.
[mean](xref:Tensor.Tensor`1.mean*) | Calculates the mean of the tensor.
[meanAxis](xref:Tensor.Tensor`1.meanAxis*) | Calculates the mean of the elements along the specified axis.
[product](xref:Tensor.Tensor`1.product*) | Calculates the product of all elements.
[productAxis](xref:Tensor.Tensor`1.productAxis*) | Calculates the product of the elements along the specified axis.
[std](xref:Tensor.Tensor`1.std*) | Calculates the standard deviation of the tensor.
[stdAxis](xref:Tensor.Tensor`1.stdAxis*) | Calculates the standard deviation of the elements along the specified axis.
[sum](xref:Tensor.Tensor`1.sum*) | Sums all elements.
[sumAxis](xref:Tensor.Tensor`1.sumAxis*) | Sums the elements along the specified axis.
[var](xref:Tensor.Tensor`1.var*) | Calculates the variance of the tensor.
[varAxis](xref:Tensor.Tensor`1.varAxis*) | Calculates the variance of the elements along the specified axis.
[trace](xref:Tensor.Tensor`1.trace*) | Calculates the trace of the matrix.
[traceAxis](xref:Tensor.Tensor`1.traceAxis*) | Calculates the trace along the specified axes.


## Functional operations (host only)
Use these functions to perform operations that are common in functional programming languages. They require the tensor to be stored in host memory.

Function | Description
-------- | -----------
[HostTensor.foldAxis](xref:Tensor.HostTensor.foldAxis*) | Applies to specified function to all elements of the tensor, threading an accumulator through the  computation.
[HostTensor.init](xref:Tensor.HostTensor.init*) | Creates a new tensor with values returned by the specified function.
[HostTensor.map](xref:Tensor.HostTensor.map*) | Applies to specified function to all elements of the tensor.
[HostTensor.map2](xref:Tensor.HostTensor.map2*) | Applies to specified function to all elements of the two tensors.
[HostTensor.mapi](xref:Tensor.HostTensor.mapi*) | Applies to specified indexed function to all elements of the tensor.
[HostTensor.mapi2](xref:Tensor.HostTensor.mapi2*) | Applies to specified indexed function to all elements of the two tensors.


## Data exchange (host only)
Use these functions to convert tensors to and from other storage modalities. They require the tensor to be stored in host memory.

Function | Description
-------- | -----------
[HostTensor.ofArray](xref:Tensor.HostTensor.ofArray*) | Creates a one-dimensional tensor copying the specified data.
[HostTensor.ofList](xref:Tensor.HostTensor.ofList*) | Creates a one-dimensional tensor from the specified list.
[HostTensor.ofSeq](xref:Tensor.HostTensor.ofSeq*) | Creates a one-dimensional tensor from the specified sequence.
[HostTensor.read](xref:Tensor.HostTensor.read*) | Reads a tensor from the specified HDF5 object path in an HDF5 file.
[HostTensor.readUntyped](xref:Tensor.HostTensor.readUntyped*) | Reads a tensor with unspecified data type from the specified HDF5 object path in an HDF5 file.
[HostTensor.toArray](xref:Tensor.HostTensor.toArray*) | Creates an array from a one-dimensional tensor.
[HostTensor.toList](xref:Tensor.HostTensor.toList*) | Creates a list from a one-dimensional tensor.
[HostTensor.toSeq](xref:Tensor.HostTensor.toSeq*) | A sequence of all elements contained in the tensor.
[HostTensor.usingArray](xref:Tensor.HostTensor.usingArray*) | Creates a one-dimensional tensor referencing the specified data.
[HostTensor.write](xref:Tensor.HostTensor.write*) | Writes the tensor into the HDF5 file under the specfied HDF5 object path.


## Random number generation (host only)
Use these functions to generate tensors filled with random numbers.

Function | Description
-------- | -----------
[HostTensor.randomInt](xref:Tensor.HostTensor.randomInt*) | Creates a tensor filled with random integer numbers from a uniform distribution.
[HostTensor.randomNormal](xref:Tensor.HostTensor.randomNormal*) | Creates a tensor filled with random numbers from a normale distribution.
[HostTensor.randomUniform](xref:Tensor.HostTensor.randomUniform*) | Creates a tensor filled with random floating-point numbers from a uniform distribution.


