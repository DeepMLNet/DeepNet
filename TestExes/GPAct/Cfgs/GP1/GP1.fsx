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
let nHiddenGps =SizeSpec.fix 10

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
                                             CutOutsideRange       = true
                                             LengthscalesTrainable = true
                                             TrnXTrainable         = true
                                             TrnTTrainable         = true
                                             TrnSigmaTrainable     = false
                                             LengthscalesInit      = Const 0.4f
                                             TrnXInit              = Linspaced (-2.0f, 2.0f)
                                             TrnTInit              = Linspaced (-1.0f, 1.0f)
                                             TrnSigmaInit          = Const (sqrt 0.01f)}}
                       
                       NeuralLayer
                         {NInput        = nHiddenUnits
                          NOutput       = ConfigLoader.NOutput()
                          TransferFunc  = NeuralLayer.Identity
                          WeightsTrainable = true
                          BiasTrainable = true}
                      ]
             Loss   = LossLayer.MSE
             L1Weight = 0.0f
             L2Weight = 1e-4f}

    Data = {Path       = "../../../../Data/UCI/abalone.txt"
            Parameters = {CsvLoader.DefaultParameters with
                           TargetCols       = [8]
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

