#load "../../../../DeepNet.fsx"
#I "../../bin/Debug"
#r "GPAct.exe"

open Basics
open Models
open SymTensor
open Datasets
open Optimizers
open GPAct


let nHiddenUnits = SizeSpec.fix 30
let nHiddenGps =SizeSpec.fix 1

let cfg = {

    Model = {Layers = [GPActivationLayer 
                        {WeightTransform = {WeightTransform.defaultHyperPars with
                                             NInput                = ConfigLoader.NInput()
                                             NOutput               = nHiddenUnits
                                             Trainable             = true
                                             WeightsInit           = FanOptimal
                                             BiasInit              = Const 0.0f}
                         Activation      = {GPActivation.defaultHyperPars with
                                             NGPs                  = nHiddenGps
                                             NOutput               = nHiddenUnits
                                             NTrnSmpls             = SizeSpec.fix 10
                                             OutputMode            = GPActivation.MeanOnly
                                             CutOutsideRange       = true
                                             LengthscalesTrainable = false
                                             TrnXTrainable         = false
                                             TrnTTrainable         = false
                                             TrnSigmaTrainable     = false
                                             LengthscalesInit      = Const 1.0f
                                             TrnXInit              = Linspaced (-2.0f, 2.0f)
                                             TrnTInit              = FunOfLinspaced (-2.0f, 2.0f,(fun x -> tanh x) )
                                             TrnSigmaInit          = Const (sqrt 0.0f)
                                             Monotonicity          = None}}
                       
                       NeuralLayer
                         {NeuralLayer.defaultHyperPars with 
                              NInput        = nHiddenUnits
                              NOutput       = ConfigLoader.NOutput()
                              TransferFunc  = NeuralLayer.Identity
                              WeightsTrainable = true
                              BiasTrainable = true}
                      ]
             Loss   = LossLayer.SoftMaxCrossEntropy}

    //dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data
    Data = {Path       = "../../../../../letter-recognition.txt"
            Parameters = {CsvLoader.DefaultParameters with
                           TargetCols       = [0]
                           IntTreatment     = CsvLoader.IntAsNumerical
                           CategoryEncoding = CsvLoader.OneHot
                           Missing          = CsvLoader.SkipRow}}        
                                            
    Optimizer = Adam Adam.DefaultCfg

    Training = {Train.defaultCfg with 
                 MinIters  = Some 5000
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

    SaveParsDuringTraining = false
    PlotGPsDuringTraining  = true
}

