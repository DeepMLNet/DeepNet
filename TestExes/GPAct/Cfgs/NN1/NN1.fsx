#load "../../../../DeepNet.fsx"
#I "../../bin/Debug"
#r "GPAct.exe"

open Basics
open Models
open SymTensor
open Datasets
open Optimizers
open GPAct


let nHidden = SizeSpec.fix 30


let cfg = {

    Model = {Layers = [
                       NeuralLayer
                         {NeuralLayer.defaultHyperPars with
                              NInput        = ConfigLoader.NInput()
                              NOutput       = nHidden
                              TransferFunc  = NeuralLayer.Tanh
                              WeightsTrainable = true
                              BiasTrainable = true}
                       
                       NeuralLayer
                         {NeuralLayer.defaultHyperPars with
                              NInput        = nHidden
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
                 MinIters  = Some 1000
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

    SaveParsDuringTraining = false
    PlotGPsDuringTraining  = false
}

