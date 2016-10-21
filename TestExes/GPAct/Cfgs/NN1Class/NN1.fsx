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
                         {NInput        = ConfigLoader.NInput()
                          NOutput       = nHidden
                          TransferFunc  = NeuralLayer.Tanh
                          WeightsTrainable = true
                          BiasTrainable = true}
                       
                       NeuralLayer
                         {NInput        = nHidden
                          NOutput       = ConfigLoader.NOutput()
                          TransferFunc  = NeuralLayer.SoftMax
                          WeightsTrainable = true
                          BiasTrainable = true}
                      ]
             Loss   = LossLayer.CrossEntropy
             L1Weight = 0.0f
             L2Weight = 1e-4f
             }

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
    PlotGPsDuringTraining  = false
}

