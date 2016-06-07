open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers

let mnist = Mnist.load ("../../../../Data/MNIST") 0.1
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

// optimizer
let opt = Adam (loss, mi.ParameterVector, DevCuda)
let optCfg = opt.DefaultCfg

let smplVarEnv (smpl: MnistT) =
    VarEnv.empty
    |> VarEnv.add input smpl.Img
    |> VarEnv.add target smpl.Lbl

let trainable =
    Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

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



Debug.Timing <- true
Debug.TraceCompile <- true

let result = Train.train trainable mnist trainCfg


()
