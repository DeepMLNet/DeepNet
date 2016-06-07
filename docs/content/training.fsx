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

Training configuration
----------------------

Next, we need to specify the training configuration using the `Train.Cfg` record type.
For illustration purposes we write down the whole record instance; in practice you would copy `Train.defaultCfg` and change fields as necessary.
*)

let trainCfg : Train.Cfg = {    
    Seed               = 100   
    BatchSize          = 10000 
    LossRecordInterval = 10                                   
    Termination        = Train.ItersWithoutImprovement 100
    MinImprovement     = 1e-7  
    TargetLoss         = None  
    MinIters           = Some 100 
    MaxIters           = None  
    LearningRates      = [1e-3; 1e-4; 1e-5]                               
    CheckpointDir      = None  
    DiscardCheckpoint  = false 
} 

(**

The meaning of the fields is as follows.

  * **Seed** is the random seed for model parameter initialization.
  * **BatchSize** is the size of mini-batches used for training and evaluating the losses.
  * **LossRecordInterval** is the number of iterations to perform between evaluating the loss on the validation and test sets.
  * **Termination** is the termination criterium and can have the following values:
    * `Train.ItersWithImprovements cnt` to stop training after `cnt` iteraitons without improvement. 
    * `Train.IterGain gain` to train for $\mathrm{gain} \cdot \mathrm{bestIter}$ iterations where $\mathrm{bestIter}$ is the best iteration. Usually one would use $\mathrm{gain} \approx 2.0$.
    * `Train.Forever` disables the termination criterium.
  * **MinImprovement** is the minimum loss change to count as improvement and should be a small number.
  * **TargetLoss** can be used to specify a target validation loss that stops training when achieved. Use `Some loss` or `None`.
  * **MinIters** can be the minimum number of training iterations to perform in the form `Some iters`, or `None`.
  * **MaxIters** can be a hard limit on the training iterations in the form `Some iters`, or `None`.
  * **LearningRates** is a list of learning rates to use. Training starts with the first element and moves to the next one, when the termination criterium (specified by the field Termination) is triggered.
  * **CheckpointDir** may specify a directory in the form `Some dir`. (see checkpoint section for details)
  * **DiscardCheckpoint** prohibits loading of a checkpoint if it is `true`.

Performing the training
-----------------------

Now training can be performed by calling the `Train.train` function.
It takes three arguments: a trainable, the dataset to use and the training configuration.
The dataset was already loaded above.
*)

let result = Train.train trainable mnist trainCfg

(**
This will produce output similar to

    Initializing model parameters for training
    Training with Dataset (54000 training, 6000 validation, 10000 test Datasets.MnistTs)
    Using learning rate 0.001
        10:  trn= 0.5739  val= 0.4652  tst= 0.5173
        20:  trn= 0.3686  val= 0.3210  tst= 0.3583
       ...
       380:  trn= 0.0155  val= 0.1083  tst= 0.1114
       390:  trn= 0.0146  val= 0.1082  tst= 0.1113
       400:  trn= 0.0137  val= 0.1082  tst= 0.1112
       410:  trn= 0.0129  val= 0.1082  tst= 0.1112
       420:  trn= 0.0121  val= 0.1083  tst= 0.1113
       430:  trn= 0.0114  val= 0.1083  tst= 0.1113
       440:  trn= 0.0108  val= 0.1084  tst= 0.1114
       450:  trn= 0.0102  val= 0.1085  tst= 0.1115
       460:  trn= 0.0096  val= 0.1086  tst= 0.1116
       470:  trn= 0.0091  val= 0.1087  tst= 0.1118
       480:  trn= 0.0086  val= 0.1089  tst= 0.1120
       490:  trn= 0.0081  val= 0.1090  tst= 0.1121
       500:  trn= 0.0077  val= 0.1092  tst= 0.1123
       510:  trn= 0.0073  val= 0.1093  tst= 0.1125
    Trained for 110 iterations without improvement
    Using learning rate 0.0001
       410:  trn= 0.0135  val= 0.1082  tst= 0.1112
       420:  trn= 0.0134  val= 0.1082  tst= 0.1112
       ...
       510:  trn= 0.0123  val= 0.1083  tst= 0.1113
    Trained for 110 iterations without improvement
    Using learning rate 1e-05
       410:  trn= 0.0136  val= 0.1082  tst= 0.1112
       420:  trn= 0.0136  val= 0.1082  tst= 0.1112
       ...
       510:  trn= 0.0134  val= 0.1082  tst= 0.1112
    Trained for 110 iterations without improvement
    Training completed after 400 iterations in 00:30:07.6179551 because NoImprovement

While training is executed you can press the `q` key to stop training immediately and the `d` key to switch to the next learning rate specified in the configuration.

During training the parameters that produce the best validation loss are saved each time the losses are evaluated (as set by the `LossRecordInterval` field in the training configuration).
When the validation loss does not improve for the set number of iterations (field `Termination` in the training configuration), the best parameters are restored and the next learning rate (field `LearningRates`) from the configuration is used.
This explains why the iteration count resets by 100 steps, each time the loss stops improving.

The best validation lost is achieved around iteration 400, then the model starts to overfit.
Decreasing the learning rate does not help in this case, thus training is terminated after exhausting the list of learning rates.


Training result and log
-----------------------

The return value of `Train.train` is a record of type `TrainingResult` that contains the training results and the training log.
*)

printfn "Termination reason is %A after %A" result.TerminationReason result.Duration
printfn "The best iteration is \n%A" result.Best
printfn "The training log consists of %d entries." (List.length result.History)

(**

This prints

    Termination reason is NoImprovement after 00:29:28.1679299
    The best iteration is 
    {Iter = 400;
     TrnLoss = 0.01370835087;
     ValLoss = 0.1082176194;
     TstLoss = 0.1112449616;
     LearningRate = 0.001;}
    The training log consists of 51 entries.


It is possible to save the training result as a JSON file by calling `result.Save`.
This is useful when you use software or scripts to gather and analyze the results of multiple experiments.


Checkpointing
-------------

Checkpoint allows to training process to be interrupted and resumed later.
To enable checkpoint support, set the `CheckpointDir` of the configuration record to some suitable directory.
This directory has to be unique for each process.

When checkpoint support is enabled, the training functions traps [the CTRL+C and CTRL+BREAK signals](https://msdn.microsoft.com/en-us/library/windows/desktop/ms682541(v=vs.85).aspx).
When such a signal is received, the training state (including the model parameters) is stored in the specified directory and the process is terminated with exit code 10.
In this case, the training function does not return to the user code.

When the program is executed again and the training function is called, it checks for a valid checkpoint.
If one is found, it is loaded and training resumes where it was interrupted.

To discard an existing checkpoint (for example if training or models parameters were changed), set `DiscardCheckpoint` to true.
This will delete any existing checkpoints from disk and restart training from the beginning.


Summary
=======

With the generic training function you can train any model that has a loss expression.
The main effort is to write a small wrapper function that maps a training sample to a variable environment.
Various termination criteria, common in machine learning, are implemented.


*)





