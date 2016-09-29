#load "../../../DeepNet.fsx"
#I "../bin/Debug"
#r "TrainFromConfig.exe"

open Basics
open Models
open Datasets
open Optimizers
open TrainFromConfig



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
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

}

