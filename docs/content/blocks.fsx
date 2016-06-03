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
We will call our example component `MyFirstPerceptron`.

We will build a component for a single layer of neurons.
Our neural layer will compute the function $f(\mathbf{x}) = \mathbf{\sigma} ( W \mathbf{x} + \mathbf{b} )$ where $\mathbf{\sigma}$ can be either element-wise $\mathrm{tanh}$ or the soft-max function $\mathbf{\sigma}(\mathbf{x})_i = \exp(x_i) / \sum_{i'} \exp(x_i')$. 
$W$ is the weight matrix and $\mathbf{b}$ is the bias vector.

Consequently our component has two parameters (a parameter is a quantity that changes during model training): $W$ and $\mathbf{b}$.
These two parameters give rise to two integer hyper-parameters (a hyper-parameter is fixed a model definition and does not change during training): the number of inputs `NInput` (number of columns in $W$) and the number of outputs `NOutput` (number of rows in $W$).
Furthermore we have the transfer function as a third, discrete hyper-parameter `TransferFunc` that can either be `Tanh` or `SoftMax`.
Let us define record types for the parameters and hyper-parameters.
*)
open ArrayNDNS; open SymTensor

module MyFirstPerceptron = 

    type TransferFuncs =
        | Tanh
        | SoftMax

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
If no custom initializer is specified, the parameter is initialized using random numbers from a uniform distribution with support $[-0.01, 0.01]$.

We also store a reference to the hyper-parameters in our parameter record to save ourselves the trouble of passing the hyper-parameter record to functions that require both the parameter record and the hyper-parameter record.

Now, we can define the function that returns the expression for the output of the perceptron component.
*)

    let pred pars input =
        let activation = !pars.Weights .* input + !pars.Bias
        match pars.HyperPars.TransferFunc with
        | Tanh     -> tanh activation
        | SoftMax  -> exp activation / Expr.sumKeepingAxis 0 (exp activation)

(**
The function computes the activation using the formula specified above and then applies the transfer function specified in the hyper-parameters.
The normalization of the soft-max activation function is performed over the left-most axis.
The `!` operator is used to dereference the reference cells in the parameter record.

This concludes the definition of our model component.


Predefined model components
---------------------------
The `Models` namespace of Deep.Net contains the following model components:

  * **LinearRegression**. A linear predictor.
  * **NeuralLayer**. A layer of neurons, with weights, bias and a transfer function.
  * **LossLayer**. A layer that calculates the loss between predictions and target values using a difference  metric (for example the mean-squared-error or cross entropy).
  * **MLP**. A multi-layer neural network with a loss layer on top.


Using a model component
-----------------------
Let us rebuild the hand-crafted model described in the chapter [Learning MNIST](mnist.html) using the `MyFirstPerceptron` component and the `LossLayer` component from Deep.Net.
As before the model will consist of one hidden layer of neurons with a tanh transfer function and an output layer with a soft-max transfer function.

As in the referred chapter, we first load the MNIST dataset and declare symbolic sizes for the model.
*)

let mnist = Datasets.Mnist.load (__SOURCE_DIRECTORY__ + "../../../Data/MNIST")
            |> Datasets.Mnist.toCuda

let mb = ModelBuilder<single> "NeuralNetModel"

let nBatch  = mb.Size "nBatch"
let nInput  = mb.Size "nInput"
let nClass  = mb.Size "nClass"
let nHidden = mb.Size "nHidden"
(**
Then we instantiate the parameters of our components.
*)
let lyr1 = 
    MyFirstPerceptron.pars (mb.Module "lyr1") 
        {NInput=nInput; NOutput=nHidden; TransferFunc=MyFirstPerceptron.Tanh}
let lyr2 =
    MyFirstPerceptron.pars (mb.Module "lyr2")
        {NInput=nHidden; NOutput=nClass; TransferFunc=MyFirstPerceptron.SoftMax}
(**
We used the `mb.Module` method of the model builder to create a new, subordinated model builder for the  components.
The `mb.Module` function takes one argument that specifies an identifier for the subordinated model builder.
The name of the current model builder is combined using a dot with the specified identifier to construct the name of the subordinated model builder.
In this example `mb` has the name `NeuralNetModel` and we specified the identifier `lyr1` when calling `mb.Module`.
Hence, the subordinate model builder will have the name `NeuralNetModel.lyr1`.

The `mb.Param` method combines the name of the model builder with the specified identifier to construct the full parameter name.
Thus the weights parameter of `lyr1` will have the full name `NeuralNetModel.lyr1.Weights` and the biases will be `NeuralNetModel.lyr1.Bias`.
This automatic parameter name construction allows multiple, independent instantiations of components without name clashes.

We continue with variable definitions and model instantiation.
*)
let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]

mb.SetSize nInput mnist.TrnImgsFlat.Shape.[1]
mb.SetSize nClass mnist.TrnLbls.Shape.[1]
mb.SetSize nHidden 100

open SymTensor.Compiler.Cuda
let mi = mb.Instantiate DevCuda

(**
Next, we use the components to generate the model's expressions.
*)

let hiddenVal = MyFirstPerceptron.pred lyr1 input.T
let classProb = MyFirstPerceptron.pred lyr2 hiddenVal

(**
And the `LossLayer` from Deep.Net to generate an expression for the loss.
*)

open Models
let loss = LossLayer.loss LossLayer.CrossEntropy classProb target.T

(**
We can now precede to compile our model's expressions into functions and train it using the gradient descent optimizer for a fixed number of iterations.
*)

let opt = Optimizers.GradientDescent (loss, mi.ParameterVector, DevCuda)
let optCfg = { Optimizers.GradientDescent.Step=1e-1f }

let lossFn = mi.Func loss |> arg2 input target
let optFn = mi.Func opt.Minimize |> opt.Use |> arg2 input target

for itr = 0 to 1000 do
    optFn mnist.TrnImgsFlat mnist.TrnLbls optCfg |> ignore
    if itr % 50 = 0 then
        let l = lossFn mnist.TstImgsFlat mnist.TstLbls |> ArrayND.value
        printfn "Test loss after %5d iterations: %.4f" itr l

(**

This should produce output similar to

    Test loss after     0 iterations: 2.3013
    Test loss after    50 iterations: 1.9930
    Test loss after   100 iterations: 1.0479
    ...
    Test loss after  1000 iterations: 0.2701

*)

(**

Nesting model components
------------------------

Model components can be nested.
This means that a component can contain one more other components.
For illustration, let us define an autoencoder component using our `MyFirstPerceptron` component.

We begin by defining the hyper-parameters and parameter.
*)
module MyFirstAutoencoder =

    type HyperPars = {
        NInOut:         SizeSpecT
        NLatent:        SizeSpecT
    }

    type Pars = {
        InLayer:        MyFirstPerceptron.Pars
        OutLayer:       MyFirstPerceptron.Pars
        HyperPars:      HyperPars
    }

(**
The hyper-parameters consists of the number of inputs and output and the number of neurons that constitute the latent representation.
The parameters are made up of the parameters of the input layer and the parameters of the output layer; thus we just reuse the existing record type from the `MyFirstPerceptron` component.

Next, we define the `pars` function that instantiates a parameter record for this component.
*)

    let pars (mb: ModelBuilder<_>) (hp: HyperPars) = 
        let hpInLayer : MyFirstPerceptron.HyperPars = {
            NInput       = hp.NInOut
            NOutput      = hp.NLatent
            TransferFunc = MyFirstPerceptron.Tanh
        }
        let hpOutLayer : MyFirstPerceptron.HyperPars = {
            NInput       = hp.NLatent
            NOutput      = hp.NInOut
            TransferFunc = MyFirstPerceptron.Tanh
        }

        {
            InLayer   = MyFirstPerceptron.pars (mb.Module "InLayer") hpInLayer
            OutLayer  = MyFirstPerceptron.pars (mb.Module "OutLayer") hpOutLayer
            HyperPars = hp
        }

(**
The function computer the hyper-parameters for the input and output layer and calls the `MyFirstPerceptron.pars` function to instantiate the parameter records for the two employed perceptrons.

Now, we can define the expressions for the latent values, the reconstruction and the reconstruction error.
*)

    let latent pars input = 
        input
        |> MyFirstPerceptron.pred pars.InLayer

    let reconst pars input =
        input
        |> MyFirstPerceptron.pred pars.InLayer
        |> MyFirstPerceptron.pred pars.OutLayer

    let loss pars input = 
        input
        |> reconst pars
        |> LossLayer.loss LossLayer.MSE input

(**
This concludes the definition of the autoencoder model.
As you have seen, it is straightforward to create more complex components by combining existing components.

Finally, let us instantiate our simple autoencoder with 100 latent units and train it on MNIST.
*)

let mb2 = ModelBuilder<single> "AutoEncoderModel"

// define symbolic sizes
let nBatch2  = mb2.Size "nBatch2"
let nInput2  = mb2.Size "nInput2"
let nLatent2 = mb2.Size "nLatent2"

// define model parameters
let ae = 
    MyFirstAutoencoder.pars (mb2.Module "Autoencoder") {NInOut=nInput2; NLatent=nLatent2}

// instantiate model
mb2.SetSize nInput2 mnist.TrnImgsFlat.Shape.[1]
mb2.SetSize nLatent2 100
let mi2 = mb2.Instantiate DevCuda

// loss function
let input2 = mb2.Var "Input"  [nBatch2; nInput2]
let loss2 = MyFirstAutoencoder.loss ae input2.T
let lossFn2 = mi2.Func loss2 |> arg input2 

// optimization function
let opt2 = Optimizers.GradientDescent (loss2, mi2.ParameterVector, DevCuda)
let optCfg2 = { Optimizers.GradientDescent.Step=1e-1f }
let optFn2 = mi2.Func opt2.Minimize |> opt2.Use |> arg input2 

// initializes parameters and train
mi2.InitPars 123
for itr = 0 to 1000 do
    optFn2 mnist.TrnImgsFlat optCfg2 |> ignore
    if itr % 50 = 0 then
        let l = lossFn2 mnist.TstImgsFlat |> ArrayND.value
        printfn "Reconstruction error after %5d iterations: %.4f" itr l

(**
This should produce output similar to

    Reconstruction error after     0 iterations: 0.1139
    Reconstruction error after    50 iterations: 0.1124
    Reconstruction error after   100 iterations: 0.1105    
    ...
    Reconstruction error after  1000 iterations: 0.0641

**Note:** Training of the autoencoder seems to be slow with the current version of Deep.Net. We are investigating the reasons for this and plan to deploy optimizations that will make training faster.


Summary
-------

Model components provide a way to construct a model out of small building blocks.
Predefined models are located in `Models` namespace.
Component use and definition in Deep.Net is not constrained by a fixed interface but naming and signature conventions exist.
The model builder supports the use of components through the `mb.Module` function that creates a subordinated model builder with a distinct namespace to avoid name clashes between components.
A component can also contain further components; thus more complex components can be constructed out of simple ones.


*)
