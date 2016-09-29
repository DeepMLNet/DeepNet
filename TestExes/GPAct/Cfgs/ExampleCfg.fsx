#load "../../../DeepNet.fsx"
#I "../bin/Debug"
#r "GPAct.exe"

open Basics
open Models
open Datasets
open Optimizers
open GPAct



let cfg = {

    Model = {Layers = [NeuralLayer {NInput=ConfigLoader.NInput() 
                                    NOutput=ConfigLoader.NOutput()
                                    TransferFunc=NeuralLayer.Identity}]
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

