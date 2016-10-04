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

    Model = {Layers = [MeanOnlyGPLayer 
                        {MeanOnlyGPLayer.defaultHyperPars with
                            NInput                = ConfigLoader.NInput()
                            NGPs                  = nHidden
                            NTrnSmpls             = SizeSpec.fix 10
                            MeanFunction          = (fun x -> tanh x)
                            LengthscalesTrainable = true
                            TrnXTrainable         = true
                            TrnTTrainable         = true
                            TrnSigmaTrainable     = false
                            WeightsTrainable      = true
                            LengthscalesInit      = Const 0.4f
                            TrnXInit              = Linspaced (-2.0f, 2.0f)
                            TrnTInit              = Linspaced (-2.0f, 2.0f)
                            TrnSigmaInit          = Const (sqrt 0.1f)
                            WeightsInit           = FanOptimal
                            BiasInit              = Const 0.0f}
                       NeuralLayer
                         {NInput        = nHidden
                          NOutput       = ConfigLoader.NOutput()
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

    Training = {Train.defaultCfg with 
                 MinIters  = Some 5000
                 BatchSize = System.Int32.MaxValue
                 MaxIters  = None}

    SaveParsDuringTraining = false
    PlotGPsDuringTraining  = true
}

