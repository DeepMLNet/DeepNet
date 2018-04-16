# Comparison and logic operations

## Element-wise comparison operators
Element-wise comparisons are performed using the [==== (element-wise equal)](xref:Tensor.Tensor`1.op_EqualsEqualsEqualsEquals*), [<<<< (element-wise smaller than)](xref:Tensor.Tensor`1.op_LessLessLessLess), [>>>> (element-wise greater than)](xref:op_GreaterGreaterGreaterGreater) and [<<>> (element-wise not equal)](xref:op_LessLessGreaterGreater) operators.

The following example compares the elements of tensors `d` and `e` for equality.
```fsharp
let d = HostTensor.ofList [0;  1; 2;  3]
let e = HostTensor.ofList [0; 10; 2; 30]
let j = d ==== e
// j = [true; false; true; false]
```

### Floating-point accuracy
When working with floating-point tensors, testing for exact equality is usually not a good approach, since it does not take into account inaccuracies that are introduced by the finite precision of floating-point data types.
Thus exact comparisions may return `false` even though two result are equal within the precision that can be expected from floating-point operations.

Therefore, a better approach for floating-point number is to use the [isClose](xref:Tensor.Tensor`1.isClose*) function, that checks for element-wise equality within machine precision.

To check whether all elements of two tensors are equal within machine precision, use the [almostEqual](xref:Tensor.Tensor`1.almostEqual*) function.


## Element-wise logic operators
Boolean tensors support the following element-wise logic operators [~~~~ (element-wise negation)](xref:Tensor.Tensor`1.op_TwiddleTwiddleTwiddleTwiddle*), [&&&& (element-wise and)](xref:Tensor.Tensor`1.op_AmpAmpAmpAmp*),  [|||| (element-wise or)](xref:Tensor.Tensor`1.op_BarBarBarBar*) and [^^^^ (element-wise xor)](xref:Tensor.Tensor`1.op_HatHatHatHat*).

The following example shows how to negate a boolean tensor and perform an and operation.

```fsharp
let nj = ~~~~j
// nj = [false true  false true ]
let jnj = j &&&& nj
// jnj = [false false false false]
```

As expected, the result of anding an element and its negation is always `false`.

## Indicies of true elements
The [trueIdx](xref:Tensor.Tensor`1.trueIdx*) finds all `true` elements within a boolean tensors and returns their indices as a new tensor.
The following example illustrates its use.

```fsharp
let a = HostTensor.ofList2D [[true; false; true; false]
                             [false; true; true; false]]
let b = Tensor.trueIdx a
// b = [[0L; 0L]
//      [0L; 2L]
//      [1L; 1L]
//      [1L; 2L]]
```

## If/then/else operation
Sometimes it is desirable to select an element from either a tensor or another, depending on the truth value of a condition.
For example, `r.[1]` should be `5` if `c.[1]` is `true`, but `7` if `c.[1]` is `false`.

The [ifThenElse](xref:Tensor.Tensor`1.ifThenElse) function provides this functionality.
It takes a boolean condition tensor, a tensor for the values to use if the condition is `true` and a tensor for the values to use if the conditions is `false`.
All three tensors must be of same shape.

The following example demonstrates the use of this function.

```fsharp
 let cond = HostTensor.ofList [true; false; false]
 let ifTrue = HostTensor.ofList [2.0; 3.0; 4.0]
 let ifFalse = HostTensor.ofList [5.0; 6.0; 7.0]
 let t = Tensor.ifThenElse cond ifTrue ifFalse
 // t = [2.0; 6.0; 7.0]
```

## Logical reduction operations
Similar to the standard reduction operations like summation, boolean tensors provide additional reduction operations specifically for then need of logic operations.

### Check if all elements are true
The [all](xref:Tensor.Tensor`1.all*) function checks whether all elements within a boolean tensor are `true`.
The result is returned as a primitive boolean.
If an empty tensor is specified, it also returns `true`.
The following example demonstrates its use.

```fsharp
let aj = Tensor.all j
// aj = false
```

The [allAxis](xref:Tensor.Tensor`1.allAxis*) function performs the same check, but in parallel for all elements along the specified axis.
In the following example it is used to check if all entries within a column of a matrix are `true`.

```fsharp
let g = HostTensor.ofList2D [[true; false; false;  true]
                             [true; false;  true; false]]
let ga = Tensor.allAxis 0 g
// ga = [true; false; false; false]
```

### Check if any element is true
To check whether at least one element in a tensor is `true`, use the [any](xref:Tensor.Tensor`1.any*) function.
If an empty tensor is specified, it returns `false`.
The [anyAxis](xref:Tensor.Tensor`1.anyAxis*) function performs the same check, but in parallel for all elements along the specified axis.

### Count true elements
The [countTrue](xref:Tensor.Tensor`1.countTrue*) and [countTrueAxis](xref:Tensor.Tensor`1.countTrueAxis*) count the number of `true` elements within a boolean tensor or along the specified axis respectively.

