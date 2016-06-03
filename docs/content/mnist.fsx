(*** hide ***)
#load "../../DeepNet.fsx"

//SymTensor.Compiler.Cuda.Debug.Timing <- true
//SymTensor.Compiler.Cuda.Debug.MemUsage <- true

(**
Learning MNIST
==============

In this example we will show how to learn MNIST classification using a two-layer feed-forward network.

You can run this example by executing `FsiAnyCPU.exe docs\content\mnist.fsx` after cloning the Deep.Net repository.
You can move your mouse over any symbol in the code samples to see the full signature.
A [quick introduction to F# signatures](https://fsharpforfunandprofit.com/posts/function-signatures/) might be helpful.

### Namespaces
The `ArrayNDNS` namespace houses the numeric tensor functionality.
`SymTensor` houses the symbolic expression library.
`SymTensor.Compiler.Cuda` provides compilation of symbolic expressions to functions that are executed on a CUDA GPU.

*)
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda

(**

Loading MNIST
-------------

We use the Mnist module from the Datasets library to load the MNIST dataset.
*)

let mnist = Datasets.Mnist.load (__SOURCE_DIRECTORY__ + "../../../Data/MNIST")
            |> Datasets.Mnist.toCuda

(**
`Mnist.load` loads the MNIST dataset from the specified directory and `Mnist.toCuda` transfers the whole dataset to the GPU.

`mnist.TrnImgsFlat` is an array of shape $60000 \times 784$ that contains the training images and `mnist.TrnLabels` is an array of shape $60000 \times 10$ containing the corresponding labels in one-hot encoding, i.e. `mnist.TrnLbls.[[n; c]] = 1` when the n-th training sample if of class (digit) c.


Defining the model
------------------

In Deep.Net a model is defined as a symbolic function of variables (input, target) and parameters (weights, biases, etc.).
In this example we construct our model using primitive operations; in a later chapter we will show how to combine predefined building blocks into a model.
To ensure that the model is valid, Deep.Net requires the specification of the shapes of all variables and parameters during the definition of the model.
For example $a+b$ is only valid if $a$ and $b$ are tensors of the same size (or can be broadcasted to the same size).
Similarly matrix multiplication $a \cdot b$ requires $b$ to have the as many rows as $a$ has columns.
 
The shape of a tensor can either be specified as a numeric value or, as usually done, using a symbolic value, e.g. the tensor `input` representing an input batch can be declared to be of size $\mathrm{nInput} \times \mathrm{nBatch}$ using the size symbols `nInput` and `nBatch`.
The model is checked as it is defined and errors are reported immediately in the offending source code line.
This way the long and painstaking process of identifying shape mismatches is avoided.

Each model definition in Deep.Net starts by instantiating `ModelBuilder`. *)

let mb = ModelBuilder<single> "NeuralNetModel"

(**
The model builder keeps track of symbolic sizes and parameters used in the model.
It takes a generic parameter that specifies that data type of the parameters (weights, etc.) used in the model.
Since CUDA GPUs provide best performance with 32-bit floating point arithmetic the data type `single` should be used almost always.
The non-generic parameter specifies a human-readable name for the model and will become more important when defining sub-models (models that can be composed into a larger model).

### Defining size symbols

Next, we define the size symbols for batch size, input vector size, target size and hidden layer size.
Although the input vector size and target size are known in advance, we opt to define them as size symbols anyway to keep the model more general and, if we make a mistake, receive more readable error messages.
*)

let nBatch  = mb.Size "nBatch"
let nInput  = mb.Size "nInput"
let nClass  = mb.Size "nClass"
let nHidden = mb.Size "nHidden"

(**
### Defining model parameters

Our model consists of one hidden layer with a tanh activation function and an output layer with a softmax activation.
Thus our parameters consists of the weights and biases for the hidden layer and the weights for the output layer.
The ``mb.Param`` method of the model builder is used to define the parameters.
It takes a human readable name of the parameter, the shape and optionally and initialization function as parameters.
To keep things simple, we do not specify an initialization function at this point.
*)

let hiddenWeights = mb.Param ("hiddenWeights", [nHidden; nInput])
let hiddenBias    = mb.Param ("hiddenBias",    [nHidden])
let outputWeights = mb.Param ("outputWeights", [nClass; nHidden])

(**
As the curious reader may have noted, `mb.Param` returns a [reference cell](https://msdn.microsoft.com/visualfsharpdocs/conceptual/reference-cells-%5bfsharp%5d) to an expression.
The reason behind this is that for training (optimizing) of the model, it is beneficial to have all model parameters concatenated into a single continuous vector.
However, before such a vector can be constructed, all parameters and their shapes must be known.
Hence the model builder must delay its construction and returns a reference cell that will be filled when the `mb.Instantiate` method is called.

### Defining model variables

We also need to define the input and (desired) target variables of the model.
While parameters and variables are both tensors, the difference between them is that parameters are values associated and stored with the model (i.e. not depending on a particular data sample) while variables represent data samples and are passed into the model.
The `mb.Var` method of the model builder takes a human-readable name for the variable and its shape as arguments.
*)

let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]

(**
### Instantiating the model

Instantiating a model constructs a parameter vector that contains all parameters and allocates the corresponding storage space on the host or GPU.
Since storage space allocation requires numeric values for the shape of the parameters (`hiddenWeights`, `hiddenBias`, `outputWeights`) we need to provide values for the corresponding size symbols `nInput`, `nHidden` and `nClass`.

We use the `mb.SetSize` method of the model builder that takes a symbolic size and the corresponding numeric value.
*)

mb.SetSize nInput mnist.TrnImgsFlat.Shape.[1]
mb.SetSize nClass mnist.TrnLbls.Shape.[1]
mb.SetSize nHidden 100

(**
The number of inputs and classes are set from the corresponding shapes of the loaded dataset and `nHidden` is set to 100.
Consequently we have a model with 784 input neurons, 100 hidden neurons and 10 output neurons.

Since all model sizes are defined and all parameters have been declared, we can now instantiate the model using the `mb.Instantiate` method of the model builder.
It takes a single argument specifying the location of the parameter vector.
This parameter can be `DevHost` for host (CPU) storage or `DevCuda` for GPU storage.
*)

let mi = mb.Instantiate DevCuda

(**
This causes the allocation of a parameter vector of appropriate size on the CUDA GPU and the reference cells for the model parameters `hiddenWeights`, `hiddenBias` and `outputWeights` are now filled with symbolic tensor slices of the parameter vector.


### Defining model expressions

We can now define the expressions for the model.
We start with the hidden layer.
Its value is given by $\mathbf{h} = W_h \mathbf{x} + \mathbf{b_h}$ where $W_h$ are the hidden weights and $\mathbf{b_h}$ is the hidden bias.
*)

let hiddenAct = !hiddenWeights .* input.T + !hiddenBias
let hiddenVal = tanh hiddenAct

(**
The `!` operator is used to dereference the reference cells for the parameters (see above).
The `.*` operator is defined as the dot product between two matrices or a matrix and a vector.
Since the `tanh` function is overloaded in Deep.Net, it can also be applied directly symbolic expressions.
This is also true for all standard arithmetic functions defined in F#, such as `sin`, `cos`, `log`, `ceil`.

Next, we define the expressions for the predictions of the model.
The output activations are given by $\mathbf{g} = W_g \mathbf{h}$ where $W_g$ are the output weights.
The class probabilities, i.e. the probabilities $p(c=C)$ that the sample is digit $C$, are given by the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) $p(c) = \exp(g_c) / \sum_{c'=0}^9 \exp(g_c')$.
*)

let outputAct = !outputWeights .* hiddenVal
let classProb = exp outputAct / Expr.sumKeepingAxis 0 (exp outputAct)

(**
Here we see the use of the `Expr.sumKeepingAxis` function.
It takes two arguments; the first argument specifies the axis to sum over and the second is the expression that should be summed.
The result has the same shape as the input but with the summation axis length set to one.
The `Expr.sumAxis` function sums over the specified axis returning a tensor with the summation axis removed.
The `Expr.sum` function sums over all axes returning a scalar tensor.

With the prediction expression fully defined, we still need to define a loss expression to train our models.
We use the standard [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) loss in this example, $L = - \sum_{c=0}^9 t_c \log p(c)$ where $t_c=1$ if the sample is digit $c$ and $0$ otherwise (one-hot encoding).
*)

let smplLoss = - Expr.sumAxis 0 (target.T * log classProb)
let loss = Expr.mean smplLoss

(**
To have a loss expression that is independent of the batch size, we take the mean of the loss over each batch.

### Compiling functions

With the expressions fully defined, we can compile the loss expression into a function.
This is done using the `mi.Func` method of the instantiated model.
*)

let lossFn = mi.Func loss |> arg2 input target

(**
`mi.Func` produces a function that expects a variable environment as its sole argument.
A variable environment (VarEnv) is an [F# map](https://msdn.microsoft.com/en-us/visualfsharpdocs/conceptual/collections.map['key,'value]-class-[fsharp]) where keys are the symbolic variables (such as `input` or `target`) and values are the corresponding numeric tensors.
Since this makes calling the function a little awkward, we pipe the resulting function into the wrapper `arg2`.
`arg2` translates a function taking a VarEnv into a function taking two arguments that are used as values for the two symbolic variables passed as parameters to `arg2`.
Hence, in this example the resulting `lossFn` function takes two tensors as arguments, with the first argument becoming the value for the symbolic variable `input` and the second argument becoming the value for `target`.
There are wrappers `arg1`, `arg3`, `arg4`, ... for different number of arguments.
The return type of a compiled function is always a tensor.
If the result of the expression is a scalar (as in our case), the tensor will have rank zero.
*)

(**

Initializing the model parameters
---------------------------------

We initialize the parameters model by calling `mi.InitPars`.
The only argument to that function is the seed to use for random initialization of the model's parameters.
`mi.InitPars` samples from an uniform distribution with support $[-0.01, 0.01]$ to initialize all model parameters that had no initializer specified when calling `mb.Param`.

*)

mi.InitPars 123

(**

We use a fixed seed of 123 to get reproducible results, but you can change it to a time-dependent value to get varying starting points.


Testing the model
-----------------

We can now test our work so far by calculating the loss of the *untrained* model on the MNIST test set.
*)

let tstLossUntrained = lossFn mnist.TstImgsFlat mnist.TstLbls
                       |> ArrayND.value
printfn "Test loss (untrained): %.4f" tstLossUntrained

(**
We call our compiled loss function as expected and pipe the result into the `Tensor.value` function to extract the float value from the zero-rank tensor.
This should print something similar to

    Test loss (untrained): 2.3019
*)

(**

Training 
--------

An untrained model is a useless model.
To fix this we will use gradient descent to find parameter values where the loss is (locally) minimal.
*)

let opt = Optimizers.GradientDescent (loss, mi.ParameterVector, DevCuda)

(**
We use the `GradientDescent` optimizer from the `Optimizers` library.
Each optimizer in Deep.Net takes three arguments: the expression to minimize, the variable with respect to the minimization should be performed and the device (DevHost or DevCuda).
The `mi.ParameterVector` property of the model instance provides a vector that is a concatenation of all model parameters defined via calls to `mb.Param`.
In our example `mi.ParameterVector` is a vector of length $\mathrm{nHidden} \cdot \mathrm{nInput} + \mathrm{nHidden} + \mathrm{nClass} \cdot \mathrm{nHidden}$ containing flattened views of the parameters `hiddenWeights`, `hiddenBias` and `outputWeights`.
Thus we have constructed a gradient descent optimizer that minimizes the loss with respect to our model's parameters.

What remains to be done, is to compile a function that performs an optimization step when called.
The expression that performs an optimization step when evaluated is provided in the `opt.Minimize` property of the optimizer instance.
We thus define the optimization function as follows.
*)

let optFn = mi.Func opt.Minimize |> opt.Use |> arg2 input target

(**
This definition is similar to `lossFn` with the addition of piping through `opt.Use`.
It makes the resulting function accept an additional parameter corresponding to the configuration of the optimizer and injects the necessary values into the VarEnv.
`opt.Use` must be used when compiling an optimization expression.
An optimization function returns an empty tensor (zero length).

Thus `optFn` is a function taking three parameters: the input images, the target labels and a record of type `GradientDescent.Cfg` that contains the optimizer configuration.

We still need to declare the optimizer configuration.
The gradient descent optimizer has a single configurable parameter: the learning rate.
*)

let optCfg = { Optimizers.GradientDescent.Step=1e-1f }

(**
We use a learning rate of $0.1$.
This high learning rate is feasible because we will calculate the gradient on the whole dataset (50 000 images) and thus it will be very stable.
If we did [mini-batch training](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) instead, i.e. split the training set into small mini-batches and update the parameters after estimating the gradient on a mini-batch, we would have to use a smaller learning rate.
*)

(**

### Training loop

We are now ready to train and evaluate our model using a simple training loop.
*)

for itr = 0 to 1000 do
    optFn mnist.TrnImgsFlat mnist.TrnLbls optCfg |> ignore
    if itr % 50 = 0 then
        let l = lossFn mnist.TstImgsFlat mnist.TstLbls |> ArrayND.value
        printfn "Test loss after %5d iterations: %.4f" itr l

(**
We train for 1000 iterations using the whole dataset (50 000 images) in each iteration.
The loss is evaluated every 50 iterations on the test set (10 000 images) and printed.

This should produce output similar to
    
    Test loss after     0 iterations: 2.3019
    Test loss after    50 iterations: 2.0094
    Test loss after   100 iterations: 1.0628
    ....
    Test loss after  1000 iterations: 0.2713

Deep.Net also provides a generic training function with parameter and loss logging, automatic adjustment of the learning rate and automatic termination.
We will show its use in a later chapter.

*)

(**

Summary
-------
In this introductory example we showed how to define symbolic sizes and build a two-layer neural network with a softmax output layer and cross-entropy loss using elementary mathematical operators.
Training was performed on the MNIST dataset using a simple training loop.

In the following sections, we will show how to assemble models from predefined blocks (such as neural layers and loss layers) and use a Deep.Net provided, configurable training loop.

*)
