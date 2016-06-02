(*** hide ***)
#load "../../DeepNet.fsx"

(**

Model Components
================

As you start building more complex machine learning models, it becomes beneficial to build the model from small, reusable components.
For example it makes sense to define a generic [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) component and use it in multiple models.
In Deep.Net, a model component can contain other model components; for example an autoencoder component could be built using two multi-layer perceptron components.

In this document we will describe how to build a simple layer of neurons and how to instantiate it in your model.


Defining a model component
--------------------------

A model component corresponds to an F# module that contains conventionally named types and functions.
We will call our example component `Perceptron`.

We will build a component for a single layer of neurons.
Our neural layer will compute the function $f(\mathbf{x}) = \sigma ( W \mathbf{x} + \mathbf{b} )$ where $\sigma$ can be either $\mathrm{tanh}$ or the identity function $\mathrm{id}$. $W$ is the weight matrix and $\mathbf{b}$ is the bias vector.

Consequently our component has two parameters (a parameter is a quantity that changes during model training): $W$ and $\mathbf{b}$.
These two parameters give rise to two integer hyper-parameters (a hyper-parameter is fixed a model definition and does not change during training): the number of inputs `NInput` (number of columns in $W$) and the number of outputs `NOutput` (number of rows in $W$).
Furthermore we have the transfer function as a third, discrete hyper-parameter `TransferFunc` that can either be `Tanh` or `Identity`.
Let us define record types for the parameters and hyper-parameters.
*)
open ArrayNDNS; open SymTensor

module Perceptron = 

    type TransferFuncs =
        | Tanh
        | Identity

    type HyperPars = {
        NInput:         SizeSpecT
        NOutput:        SizeSpecT
        TransferFunc:   TransferFuncs
    }

    type Pars = {
        Weights:        ExprT<single> ref
        Bias:           ExprT<single> ref
        HyperPars:      HyperPars
    }

(**
We see that, by convention, the record type for the hyper-parameters is named `HyperPars`.
The fields `NInput` and `NOutput` have been defined as type `SizeSpecT`.
This is the type used by Deep.Net to represent an integral size, either symbolic or numeric.

Also by convention, the record type that stores the model parameters is named `Pars`.
The weights and bias have been defined as type `ExprT<single> ref`.

`SymTensor.ExprT<'T>` is the type of an symbolic tensor expression of data type `'T`.
For example `'T` can be `float` for a tensor containing double precision floating point numbers or, as in this case, `single` for single precision floats.
The reader might wonder, why we use the generic expression type instead of the `VarSpecT` type that represents a symbolic variable in Deep.Net.
After all, the model's parameters are variables, are they not?

While in most cases, a model parameter will be a tensor variable, it makes sense to let the user pass an arbitrary expression for the model parameter.
Consider, for example, an auto-encoder with tied input/output weights (this means that the weights of the output layer are given by the transposition of the weights of the input layer).
The user can construct such an auto-encoder using two of our perceptron components.
He just needs to set `pOut.Weights <- pIn.Weights.T`, where `pOut` represents the parameters of the output layer and `pIn` represents the parameters of the input layer, to tie the input and output weights together.
But this would not be possible if `Pars.Weights` was of type `VarSpecT` since `pIn.Weights.T` is an expression due to the use of the transposition operation.

Furthermore we observe that `Weights` and `Bias` have been declared as reference cells.
We will see the reason for that in a few lines further below.

Let us now define the functions of ours component's module.

We define a function `pars` that, by convention, returns an instance of the parameter record.
*)

    let internal initBias (seed: int) (shp: int list) : ArrayNDHostT<single> =
        ArrayNDHost.zeros shp

    let pars (mb: ModelBuilder<_>) (hp: HyperPars) = {
        Weights   = mb.Param ("Weights", [hp.NOutput; hp.NInput])
        Bias      = mb.Param ("Bias",    [hp.NOutput], initBias)
        HyperPars = hp
    }

(**
The function `pars` takes two arguments: a model builder and the hyper-parameters of the component.
It construct a parameter record and populates the weights and bias with parameter tensors obtain from the model builder by calling `mb.Param` with the appropriate shapes from the hyper-parameters. 

For the bias we also specify the custom initialization function `initBias`.
A custom initialization function takes two arguments: a random seed and a list of integers representing the shape of the instantiated parameter tensor.
It should return the initialization value of appropriate shape for the parameter.
Here, we initialize the bias with zero and thus return a zero tensor of the requested shape.
If no custom initializer is specified, the parameter is initialized using random numbers from a uniform distribution with support $\[ TODO \]$.

We also store a reference to the hyper-parameters in our parameter record to save ourselves the trouble of passing the hyper-parameter record to functions that require both the parameter record and the hyper-parameter record.

Now, we can define the function that returns the expression for the output of the perceptron component.
*)

    let pred pars input =
        let activation = !pars.Weights .* input + !pars.Bias
        match pars.HyperPars.TransferFunc with
        | Tanh     -> tanh activation
        | Identity -> activation

(**
The function computes the activation using the formula specified above and then applies the transfer function specified in the hyper-parameters.
The `!` operator is used to dereference the reference cells in the parameter record.

This concludes the definition of our model component.


Using a model component
-----------------------

*)
