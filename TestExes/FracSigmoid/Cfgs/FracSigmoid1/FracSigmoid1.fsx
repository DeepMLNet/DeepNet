#load "../../../../DeepNet.fsx"
#I "../../bin/Debug"
#r "FracSigmoid.exe"

open Basics
open Models
open SymTensor
open Datasets
open Optimizers
open FracSigmoid
open ArrayNDNS


let nHidden1 = SizeSpec.fix 30

let info = {
    NMin    = 0.0
    NMax    = 1.1
    NPoints = 1000
    XMin    = -10.0
    XMax    =  10.0
    XPoints = 5000
    Function = FracSigmoid
    WithDeriv = false
}


let cfg = {

    Model = {Layers = [
                       TableLayer
                         {NInput        = Program.NInput()
                          NOutput       = nHidden1
                          NFrac         = SizeSpec.fix 4
                          Info          = info
                          FracTrainable = true
                          FracInit      = 0.0f
                          }
                      
                       NeuralLayer
                         {NeuralLayer.defaultHyperPars with
                           NInput        = nHidden1
                           NOutput       = Program.NOutput()
                           TransferFunc  = NeuralLayer.Identity
                           WeightsTrainable = true
                           BiasTrainable    = true
                           }
                      ]
             Loss   = LossLayer.MSE}

    Data = {Path       = "../../../../Data/UCI/abalone.txt"
            Parameters = {CsvLoader.DefaultParameters with
                           TargetCols       = [8]
                           IntTreatment     = CsvLoader.IntAsNumerical
                           CategoryEncoding = CsvLoader.OneHot
                           Missing          = CsvLoader.SkipRow}}        
                                            
    Optimizer = Adam Adam.DefaultCfg
    //Optimizer = GradientDescent GradientDescent.DefaultCfg

    Training = {Train.defaultCfg with 
                 //MinIters  = Some 10000
                 BestOn    = Training
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

    SaveParsDuringTraining = false
}


