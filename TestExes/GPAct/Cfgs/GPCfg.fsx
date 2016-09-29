#load "../../../DeepNet.fsx"
#I "../bin/Debug"
#r "GPAct.exe"

open Basics
open Models
open SymTensor
open Datasets
open Optimizers
open GPAct



let cfg = {

    Model = {Layers = [GPActivationLayer 
                        {WeightTransform = {WeightTransform.defaultHyperPars with
                                             NInput                = ConfigLoader.NInput()
                                             NOutput               = ConfigLoader.NOutput()
                                             Trainable             = true
                                             WeightsInit           = FanOptimal
                                             BiasInit              = Const 0.0f}
                         Activation      = {GPActivation.defaultHyperPars with
                                             NGPs                  = ConfigLoader.NOutput()
                                             NTrnSmpls             = SizeSpec.fix 10
                                             LengthscalesTrainable = true
                                             TrnXTrainable         = true
                                             TrnTTrainable         = true
                                             TrnSigmaTrainable     = true
                                             LengthscalesInit      = Const 0.4f
                                             TrnXInit              = Linspaced (-2.0f, 2.0f)
                                             TrnTInit              = Linspaced (-2.0f, 2.0f)
                                             TrnSigmaInit          = Const (sqrt 0.1f)}}]
             Loss   = LossLayer.MSE}

    Data = {Path       = "../../../Data/UCI/abalone.txt"
            Parameters = {CsvLoader.DefaultParameters with
                           TargetCols       = [8]
                           IntTreatment     = CsvLoader.IntAsNumerical
                           CategoryEncoding = CsvLoader.OneHot
                           Missing          = CsvLoader.SkipRow}}        
                                            
    Optimizer = Adam Adam.DefaultCfg

    Training = {Train.defaultCfg with 
                 MinIters  = Some 10000
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

    SaveParsDuringTraining = false
}

