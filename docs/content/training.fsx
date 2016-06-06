(*** hide ***)
#load "../../DeepNet.fsx"

(**

Generic Training 
================

Deep.Net contains a powerful, generic function to train your model.
Together with the [dataset handler](dataset.html) it provides the following functionality:
  
  * initialization of the model's parameters
  * mini-batch training
  * logging of losses on the training, validation and test sets
  * automatic scheduling of the learning rate 
  * termination of training when
    * a desired validation loss is reached
    * a set number of iterations have been performed
    * there is no loss improvement on the validation set within a set number of iterations
  * checkpointing allows the training state to be saved to disk and training to be restarted afterwards (useful when running on non-reliable hardware or on a compute cluster that pauses jobs or moves them around on the cluster's nodes)


### Example model
To demonstrate its use we return to our two-layer neural network model for classifying MNIST digits.
*)

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers

(**
We load the MNIST dataset using the `Mnist.load` function using a validation to training ratio of 0.1.
*)

let mnist = Mnist.load (__SOURCE_DIRECTORY__ + "../../../Data/MNIST") 0.1
            |> TrnValTst.ToCuda

(**
Next, we define and instantiate a model using the MLP (multi-layer perceptron, i.e. multi-layer neural network) component.
*)

let mb = ModelBuilder<single> "NeuralNetModel"

// define symbolic sizes
let nBatch  = mb.Size "nBatch"
let nInput  = mb.Size "nInput"
let nClass  = mb.Size "nClass"
let nHidden = mb.Size "nHidden"

// define model parameters
let mlp = 
    MLP.pars (mb.Module "MLP") 
        { Layers = [{NInput=nInput; NOutput=nHidden; TransferFunc=NeuralLayer.Tanh}
                    {NInput=nHidden; NOutput=nClass; TransferFunc=NeuralLayer.SoftMax}]
          LossMeasure = LossLayer.CrossEntropy }

// define variables
let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]

// instantiate model
mb.SetSize nInput mnist.Trn.[0].Img.Shape.[0]
mb.SetSize nClass mnist.Trn.[0].Lbl.Shape.[0]
mb.SetSize nHidden 100
let mi = mb.Instantiate DevCuda

// loss expression
let loss = MLP.loss mlp input.T target.T

(**
Note that the input and target matrices must be transposed, since the neural network model expects each sample to be a column in the matrix while the dataset provides a matrix where each row is a sample.

We instantiate the [Adam](https://arxiv.org/abs/1412.6980) optimizer to minimize the loss and use its default configuration.
*)

// optimizer
let opt = Adam (loss, mi.ParameterVector, DevCuda)
let optCfg = opt.DefaultCfg

(**
In previous example we have written a simple optimization loop by hand.
Here instead, we will employ the generic training function provided by Deep.Net.


Defining a Trainable
--------------------

The generic training function works on any object that implements the `Train.ITrainable<'Smpl, 'T>` interface where `'Smpl` is a sample record type (see [dataset handling](dataset.html)) and `'T` is the data type of the model parameters, e.g. `single`.
The easiest way to create an ITrainable from a symbolic loss expression is to use the `Train.trainableFromLossExpr` function.
This function has the signature

    val trainableFromLossExpr : modelInstance:ModelInstance<'T> -> 
                                loss:ExprT<'T> -> 
                                varEnvBuilder:('Smpl -> VarEnvT) -> 
                                optimizer:IOptimizer<'T,'OptCfg,'OptState> -> 
                                optCfg:'OptCfg -> 
                                ITrainable<'Smpl,'T> 

The arguments have the following meaning.

  * `modelInstance` is the model instance containing the parameters of the model to be trained.
  * `loss` is the loss expression to be minimized.
  * `varEnvBuilder` is a user-provided function that takes an instance of user-provided type `'Smpl` and returns a variable environment to evaluate the loss expression on this sample(s). The sample below shows how to build a variable environment from a sample.
  * `optimizer` is an instance of an optimizer. All optimizers in Deep.Net implement the `IOptimizer` interface.
  * `optCfg` is the optimizer configuration to use. The learning rate in the specified optimizer configuration will be overwritten.


Let us build a trainable for our model.
First, we need to define a function that creates a variable environment from a sample.
*)

let smplVarEnv (smpl: MnistT) =
    VarEnv.empty
    |> VarEnv.add input smpl.Img
    |> VarEnv.add target smpl.Lbl

(**
The value of the symbolic variable `input` is set to the image of the MNIST sample and the symbolic variable `target` is set to the label in one-hot encoding.

We are now ready to construct the trainable.
*)

let trainable =
    Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg
      
(**
Next, we need to specify the training configuration using the `Train.Cfg` record type.
For illustration purposes we write down the whole record instance; in practice you would copy `Train.defaultCfg` and change fields as necessary.
*)

let trainCfg : Train.Cfg = {    
    Seed               = 100   // Seed for initialization.
    BatchSize          = 10000 // Mini-batch size.    
    LossRecordInterval = 10    // Number of iterations between 
                               // evaluation of the loss.
    Termination        = Train.ItersWithoutImprovement 100
                               // Terminate after 100 iterations 
                               // without improvement on the 
                               // validation set.
    MinImprovement     = 1e-7  // Minimum loss change to count
                               // as improvement.
    TargetLoss         = None  // No target loss; thus continue
                               // until no improvement
    MinIters           = Some 100 // Train for at least 100 
                               // iterations.
    MaxIters           = None  // No hard limit on number of iterations.
    LearningRates      = [1e-3; 1e-4; 1e-5]
                               // Start with learning rate 1e-3, then
                               // change to 1e-4 when no improvement,
                               // then to 1e-5.
    CheckpointDir      = None  // No checkpoints.
    DiscardCheckpoint  = false // No checkpoints.
} 

(**
Now training can be performed by calling the `Train.train` function.
It takes three arguments: a trainable, the dataset to use and the training configuration.
The dataset was already loaded above.
*)

Train.train trainable mnist trainCfg

(**
This will produce output similar to




Summary
=======


*)





