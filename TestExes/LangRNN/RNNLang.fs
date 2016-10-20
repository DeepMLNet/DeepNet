namespace LangRNN

open Basics
open System.IO

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Optimizers
open Datasets


module RNNLang =



//    let NWords      = 8000
//    let NBatch      = 20
//    let NSteps      = 35
//    let NRecurrent  = 650
    
//    let NWords      = 100
//    let NBatch      = 1000
//    let NSteps      = 20
//    let NRecurrent  = 100
//    let NMaxSamples = 10000

    //let NWords      = 300
    let NBatch      = 1000
    let NSteps      = 35
    let NRecurrent  = 100
    //let NMaxSamples = 1000


//    let NWords      = 5
//    let NSteps      = 7
//    let NBatch      = 10
//    let NMaxSamples = 10
//    let NRecurrent  = 8

 

    let trainModel (dataset: TrnValTst<WordSeq>) vocSizeVal =
    //let trainModel (dataset: TrnValTst<WordSeqOneHot>) =
        let mb = ModelBuilder<single> ("Lang")

        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        let input   = mb.Var<int>     "Input"   [nBatch; nSteps]
        //let input   = mb.Var<single>  "Input"   [nBatch; nSteps; nWords]
        let initial = mb.Var<single>  "Initial" [nBatch; nRecurrent]
        //let target  = mb.Var<single>  "Target"  [nBatch; nSteps; nWords]
        let target  = mb.Var<int>     "Target"  [nBatch; nSteps]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = true
                //OneHotIndexInput        = false 
        }

        // final [smpl, recUnit]
        // pred  [smpl, step, word] - probability of word
        let final, pred = (initial, input) ||> RecurrentLayer.pred rnn

        // [smpl, step]
        let targetProb = pred |> Expr.gather [None; None; Some target]            
        let stepLoss = -log targetProb 
        //let stepLoss = -target * log pred

        let loss = Expr.mean stepLoss

        let mi = mb.Instantiate (DevCuda, Map [nWords,     vocSizeVal
                                               nRecurrent, NRecurrent])

        let smplVarEnv stateOpt (smpl: WordSeq) =
            let zeroInitial = ArrayNDCuda.zeros<single> [smpl.Words.Shape.[0]; NRecurrent]
            let state =
                match stateOpt with
                | Some state -> state :> IArrayNDT
                | None -> zeroInitial :> IArrayNDT
            let n = smpl.Words.Shape.[1]
            VarEnv.ofSeq [input,   smpl.Words.[*, 0 .. n-2] :> IArrayNDT
                          target,  smpl.Words.[*, 1 .. n-1] :> IArrayNDT
                          initial, state]
                          
        //let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv GradientDescent.New GradientDescent.DefaultCfg
        let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv Adam.New Adam.DefaultCfg

        let trainCfg = {
            Train.defaultCfg with
                //MaxIters     = Some 10
                BatchSize      = NBatch
                BestOn         = Training
                CheckpointDir  = Some "."
        }
        Train.train trainable dataset trainCfg |> ignore


