namespace LangRNN

open Basics
open System.IO

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Optimizers
open Datasets

module RNNTests =



    let verifyRNNGradientOneHot device =
        printfn "Verifying RNN gradient with one-hot class encoding..."
        let mb = ModelBuilder<single> ("Lang")
        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        let input   = mb.Var<single>  "Input"   [nBatch; nSteps; nWords]
        let initial = mb.Var<single>  "Initial" [nBatch; nRecurrent]
        let target  = mb.Var<single>  "Target"  [nBatch; nSteps; nWords]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = false 
        }
        let _, pred = (initial, input) ||> RecurrentLayer.pred rnn
        let loss = -target * log pred |> Expr.mean

        let NBatch, NSteps, NWords, NRecurrent = 2, 15, 4, 10
        let mi = mb.Instantiate (device, Map [nWords, NWords; nRecurrent, NRecurrent])
        mi.InitPars 100

        let rng = System.Random 123
        let vInput = rng.UniformArrayND (-1.0f, 1.0f) [NBatch; NSteps; NWords] |> device.ToDev
        let vInitial = ArrayNDHost.zeros [NBatch; NRecurrent] |> device.ToDev
        let vTarget = rng.UniformArrayND (-1.0f, 1.0f) [NBatch; NSteps; NWords] |> device.ToDev
        let varEnv = VarEnv.ofSeq [input, vInput; initial, vInitial; target, vTarget] 

        DerivCheck.checkExpr device 1e-2f 1e-3f (varEnv |> mi.Use) (loss |> mi.Use)
        printfn "Done."

    let verifyRNNGradientIndexed device =
        printfn "Verifying RNN gradient with indexed class encoding..."
        let mb = ModelBuilder<single> ("Lang")
        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        let input   = mb.Var<int>     "Input"   [nBatch; nSteps]
        let initial = mb.Var<single>  "Initial" [nBatch; nRecurrent]
        let target  = mb.Var<int>     "Target"  [nBatch; nSteps]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = true 
        }
        let _, pred = (initial, input) ||> RecurrentLayer.pred rnn
        let targetProb = pred |> Expr.gather [None; None; Some target]            
        let loss = -log targetProb |> Expr.mean

        let NBatch, NSteps, NWords, NRecurrent = 2, 6, 3, 3
        let mi = mb.Instantiate (device, Map [nWords, NWords; nRecurrent, NRecurrent])
        mi.InitPars 100

        let rng = System.Random 123
        let vInput = rng.IntArrayND (0, NWords-1) [NBatch; NSteps] |> device.ToDev
        let vInitial = ArrayNDHost.zeros<single> [NBatch; NRecurrent] |> device.ToDev
        let vTarget = rng.IntArrayND (0, NWords-1) [NBatch; NSteps] |> device.ToDev
        let varEnv = VarEnv.ofSeq [input, vInput :> IArrayNDT 
                                   initial, vInitial :> IArrayNDT 
                                   target, vTarget :> IArrayNDT] 

        DerivCheck.checkExpr device 1e-2f 1e-3f (varEnv |> mi.Use) (loss |> mi.Use)
        printfn "Done."

    let computeRNNGradientIndexed device =
        printfn "Computing RNN gradient with indexed class encoding..."
        let mb = ModelBuilder<single> ("Lang")
        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        let input   = mb.Var<int>     "Input"   [nBatch; nSteps]
        let initial = mb.Var<single>  "Initial" [nBatch; nRecurrent]
        let target  = mb.Var<int>     "Target"  [nBatch; nSteps]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = true 
        }
        let _, pred = (initial, input) ||> RecurrentLayer.pred rnn
        let targetProb = pred |> Expr.gather [None; None; Some target]            
        let loss = -log targetProb |> Expr.mean

        let NBatch, NSteps, NWords, NRecurrent = 2, 6, 3, 3
        let mi = mb.Instantiate (device, Map [nWords, NWords; nRecurrent, NRecurrent])
        mi.InitPars 100

        let rng = System.Random 123
        let vInput = rng.IntArrayND (0, NWords-1) [NBatch; NSteps] |> device.ToDev
        let vInitial = ArrayNDHost.zeros<single> [NBatch; NRecurrent] |> device.ToDev
        let vTarget = rng.IntArrayND (0, NWords-1) [NBatch; NSteps] |> device.ToDev
        let varEnv = VarEnv.ofSeq [input, vInput :> IArrayNDT 
                                   initial, vInitial :> IArrayNDT 
                                   target, vTarget :> IArrayNDT] 

        let dloss = Deriv.compute loss
        let dinitial = dloss |> Deriv.ofVar initial
        let dinitialFn = mi.Func (dinitial)
        printfn "dinitial=%A" (dinitialFn varEnv)

