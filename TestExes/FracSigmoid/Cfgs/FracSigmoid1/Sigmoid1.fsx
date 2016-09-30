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


let cfg = {

    Model = {Layers = [
                       NeuralLayer
                         {NInput        = Program.NInput()
                          NOutput       = nHidden1
                          TransferFunc  = NeuralLayer.Sigmoid}
                      
                       NeuralLayer
                         {NInput        = nHidden1
                          NOutput       = Program.NOutput()
                          TransferFunc  = NeuralLayer.Identity}
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
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

    SaveParsDuringTraining = false
}


