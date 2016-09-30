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


let hdf = HDF5.OpenRead "../../Tables/FracSigmoid1.h5"
let name = "FracSigmoid"
let table = FracSigmoidTable.load hdf name
printfn "Using table\n%A" table.Info

let nHidden1 = SizeSpec.fix 30
let nHidden2 = SizeSpec.fix 30


let cfg = {

    Model = {Layers = [
                       TableLayer
                         {NInput        = Program.NInput()
                          NOutput       = nHidden1
                          Table         = table}
                      
                       TableLayer
                         {NInput        = nHidden1
                          NOutput       = nHidden2
                          Table         = table}

                       NeuralLayer
                         {NInput        = nHidden2
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
                                            
    //Optimizer = Adam Adam.DefaultCfg
    Optimizer = GradientDescent GradientDescent.DefaultCfg

    Training = {Train.defaultCfg with 
                 MinIters  = Some 10000
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

    SaveParsDuringTraining = false
}


