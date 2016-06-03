(*** hide ***)
#load "../../DeepNet.fsx"

(**

Generic Training and Dataset Handling
=====================================

Deep.Net provides a powerful generic function to train your model and generic dataset storage and handling.
Together they provide the following features:

  * TODO


### Example models
To demonstrate its use we return to our two-layer neural network model for classifying MNIST digits.


*)

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models

let mnist = Datasets.Mnist.load (__SOURCE_DIRECTORY__ + "../../../Data/MNIST")
            |> Datasets.Mnist.toCuda

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
mb.SetSize nInput mnist.TrnImgsFlat.Shape.[1]
mb.SetSize nClass mnist.TrnLbls.Shape.[1]
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

