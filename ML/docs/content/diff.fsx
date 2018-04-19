(*** hide ***)
#load "../../DeepNet.fsx"

(**

Automatic Differentiation
=========================

Deep.Net performs automatic reverse accumulation differentiation on symbolic expressions to calculate the derivatives of the user-specified model.

In most cases, the differentiation functions are invoked by the optimizer.
However, sometimes it is desired to obtain an expression for the derivative.

### A sample expression

We define an expression

$$$
\mathbf{f}(\mathbf{x}, \mathbf{y}) = \frac{1}{(\sin x)^2 + y} \,.

Here we do not use the model builder, because our intent is not to build a full model with parameter and optimizer support.
Instead we define the symbolic sizes directly using the `SizeSpec.symbol` function and declare variables using `Expr.var`.
*)

open ArrayNDNS
open SymTensor

let n = SizeSpec.symbol "n"
let x = Expr.var "x" [n]
let y = Expr.var "y" [n]

let f = 1.0 / ((sin x) ** 2.0 + y)

(**

Computing derivatives
---------------------

We can now compute the derivatives of $\mathbf{f}(\mathbf{x}, \mathbf{y})$.
To do so, we call the `Deriv.compute` function with the expression we want to differentiate.
The function value and the input variables can be of any dimensionality and shape.
*)

let df = Deriv.compute f

(**
The derivative object `df` now contains the derivative of `f` w.r.t. all variables that occur in that expression.

To access a specific derivative use the `Deriv.ofVar` function on `df` and pass the requested variable.
*)

let dfdx = df |> Deriv.ofVar x
let dfdy = df |> Deriv.ofVar y

(**
We now have expressions for $\partial f / \partial x$ and $\partial f / \partial y$.

Evaluating the expressions
--------------------------

We evaluate the expressions using the (slow) host interpreter.
Because we do not use the model builder, we have to invoke `Func.make` directly to create a callable function from an expression.
*)

let cmplr = DevHost.Compiler, CompileEnv.empty

let fnF = Func.make cmplr f |> arg2 x y
let fnDfdx = Func.make cmplr dfdx |> arg2 x y
let fnDfdy = Func.make cmplr dfdy |> arg2 x y

(**
`Func.make` expects two arguments: the first is the compiler (or interpreter) to use to transform the expression into a function.
We use the host interpreter (`DevHost.Compiler`) without any optional options (`CompileEnv.empty`).
The second argument is the expression to compile.

`Func.make` returns a function taking a variable environment `VarEnvT` (essentially a map from variable names to values) and returning a tensor value.
To avoid having to build the variable environment explicitly, we use the `arg2` function that modifies the resulting function to take two tensor arguments instead.

We can now generate some test values for the variables $\mathbf{x}$ and $\mathbf{y}$.
*)

let valX = seq { 0.1 .. 0.2 .. 1.0 } |> ArrayNDHost.ofSeq
let valY = seq { 1.1 .. 0.2 .. 2.0 } |> ArrayNDHost.ofSeq

(**
And compute the function values as well as derivatives.
*)

printfn "Using x = %A" valX
printfn "Using y = %A" valY
printfn "f(x, y) = %A" (fnF valX valY)
printfn ""
printfn "J_x f   = \n%A" (fnDfdx valX valY)
printfn "J_y f   = \n%A" (fnDfdy valX valY)

(**
This prints

    Using x = [   0.1000    0.3000    0.5000    0.7000    0.9000]
    Using y = [   1.1000    1.3000    1.5000    1.7000    1.9000]
    f(x, y) = [   0.9009    0.7208    0.5781    0.4728    0.3978]

    J_x f   = 
    [[  -0.1613    0.0000    0.0000    0.0000    0.0000]
     [   0.0000   -0.2934    0.0000    0.0000    0.0000]
     [   0.0000    0.0000   -0.2812    0.0000    0.0000]
     [   0.0000    0.0000    0.0000   -0.2203    0.0000]
     [   0.0000    0.0000    0.0000    0.0000   -0.1541]]
    J_y f   = 
    [[  -0.8117    0.0000    0.0000    0.0000    0.0000]
     [   0.0000   -0.5196    0.0000    0.0000    0.0000]
     [   0.0000    0.0000   -0.3342    0.0000    0.0000]
     [   0.0000    0.0000    0.0000   -0.2235    0.0000]
     [   0.0000    0.0000    0.0000    0.0000   -0.1583]]


As expected, the Jacobians are diagonal because we computed the derivatives of an element-wise function.


Meaning of the derivative matrix
--------------------------------

The derivative is always returned in the shape of a Jacobian, i.e. the derivative is always a matrix.
If $\mathbf{f}$ and $\mathbf{x}$ are vectors, this means

$$$
(J_\mathbf{x} \mathbf{f})_{ij} = \frac{\partial f_i}{\partial x_j}

and $J_\mathbf{x} \mathbf{f}$ will be an $n \times m$ matrix where $n$ is the length of $\mathbf{f}$ and $m$ is the length of $\mathbf{y}$.

If the function or an argument has the value of a matrix, the Jacobian will still be a matrix.
Consider, for instance, that $X$ is a $k \times l$ matrix and $G(X)$ is an $n \times m$ matrix-valued function.
Then the Jacobian $J_G X$ computed by Deep.Net will be a matrix of shape $k l \times n m$.
The derivative is computed as if $G$ and $X$ were flattened into vectors (using row-major order).
Thus the derivatives of the individual elements are given by

$$$
\frac{\partial G_{i,j}}{\partial X_{v,w}} = (J_\mathbf{X} \mathbf{G})_{im + j, vl + w}  

This is also true for higher-order tensors, i.e. the derivative will be computed as if any higher order tensor were flattened into a vector using row-major order.
Likewise, a scalar-valued function will produce a Jacobian matrix with one row.

### Chain rule
Using matrices to store the derivatives has the advantage that the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) is always valid.

Consider a vector-valued function $\mathbf{f} (G (\mathbf{x}))$.
Given the derivatives $J_\mathbf{x} G$ and $J_G \mathbf{f}$ we can compute the derivative $J_\mathbf{x} \mathbf{f}$ by

$$$
J_\mathbf{x} \mathbf{f} = J_G \mathbf{f} \cdot J_\mathbf{x} G 

where $\cdot$ represent the matrix dot product.


*)



