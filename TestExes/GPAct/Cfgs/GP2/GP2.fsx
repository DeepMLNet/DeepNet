#load "../../../../DeepNet.fsx"
#I "../../bin/Debug"
#r "GPAct.exe"

open Basics
open Models
open SymTensor
open Datasets
open Optimizers
open GPAct


let nHidden1 = SizeSpec.fix 30
let nHidden2 = SizeSpec.fix 30


let cfg = {

    Model = {Layers = [
                       GPActivationLayer 
                        {WeightTransform = {WeightTransform.defaultHyperPars with
                                             NInput                = ConfigLoader.NInput()
                                             NOutput               = nHidden1
                                             Trainable             = true
                                             WeightsInit           = FanOptimal
                                             BiasInit              = Const 0.0f}
                         Activation      = {GPActivation.defaultHyperPars with
                                             NGPs                  = nHidden1
                                             NTrnSmpls             = SizeSpec.fix 10
                                             LengthscalesTrainable = true
                                             CutOutsideRange       = true
                                             TrnXTrainable         = true
                                             TrnTTrainable         = true
                                             TrnSigmaTrainable     = false
                                             LengthscalesInit      = Const 0.4f
                                             TrnXInit              = Linspaced (-2.0f, 2.0f)
                                             TrnTInit              = Linspaced (-2.0f, 2.0f)
                                             TrnSigmaInit          = Const (sqrt 0.01f)}}

                       GPActivationLayer 
                        {WeightTransform = {WeightTransform.defaultHyperPars with
                                             NInput                = nHidden1
                                             NOutput               = nHidden2
                                             Trainable             = true
                                             WeightsInit           = FanOptimal
                                             BiasInit              = Const 0.0f}
                         Activation      = {GPActivation.defaultHyperPars with
                                             NGPs                  = nHidden2
                                             NTrnSmpls             = SizeSpec.fix 10
                                             LengthscalesTrainable = true
                                             TrnXTrainable         = true
                                             CutOutsideRange       = true
                                             TrnTTrainable         = true
                                             TrnSigmaTrainable     = false
                                             LengthscalesInit      = Const 0.4f
                                             TrnXInit              = Linspaced (-2.0f, 2.0f)
                                             TrnTInit              = Linspaced (-2.0f, 2.0f)
                                             TrnSigmaInit          = Const (sqrt 0.01f)}}
                       
                       NeuralLayer
                         {NInput        = nHidden2
                          NOutput       = ConfigLoader.NOutput()
                          TransferFunc  = NeuralLayer.Identity
                          WeightsTrainable = true
                          BiasTrainable = true}
                      ]
             Loss   = LossLayer.MSE}

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

