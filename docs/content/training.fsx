(*** hide ***)
#load "../../DeepNet.fsx"

(**

Generic Training 
================

Deep.Net provides a powerful generic function to train your model.
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


### Example models
To demonstrate its use we return to our two-layer neural network model for classifying MNIST digits.


*)

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets

let mnist = Mnist.load (__SOURCE_DIRECTORY__ + "../../../Data/MNIST") 0.1
            |> TrnValTst.ToCuda

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


open Optimizers

// optimizer
let opt = Adam (loss, mi.ParameterVector, DevCuda)

//let trainable =
    //Train.trainableFromLossExpr mi loss
      //  (fun smpl)

