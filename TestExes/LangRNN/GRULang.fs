namespace LangRNN


open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda

open Models
open Datasets
open Optimizers


module GRULang =

    type HyperPars = {
        NWords:                     SizeSpecT
        EmbeddingDim:               SizeSpecT
    }

    let defaultHyperPars = {
        NWords                      = SizeSpec.fix 0
        EmbeddingDim                = SizeSpec.fix 0
    }

    type Pars = {        
        WordToEmb:       ExprT    // [word, embDim]
        EmbToUpdate:     ExprT    // [recUnit, embDim]
        EmbToReset:      ExprT    // [recUnit, embDim]
        EmbToHidden:     ExprT    // [recUnit, embDim]
        StateToUpdate:   ExprT    // [recUnit, recUnit]
        StateToReset:    ExprT    // [recUnit, recUnit]
        StateToHidden:   ExprT    // [recUnit, recUnit]
        UpdateBias:      ExprT    // [recUnit]
        ResetBias:       ExprT    // [recUnit]
        HiddenBias:      ExprT    // [recUnit]
        StateToWord:     ExprT    // [word, recUnit]
        WordBias:        ExprT    // [word]
        HyperPars:       HyperPars
    }

    let internal initWeights seed (shp: int list) = 
        let r = 1.0f / sqrt (single shp.[1])       
        (System.Random seed).SeqSingle(-r, r) |> ArrayNDHost.ofSeqWithShape shp
        
    let internal initBias seed (shp: int list) =
        ArrayNDHost.zeros<single> shp

    let pars (mb: ModelBuilder<_>) hp = {
        WordToEmb     = mb.Param ("WordToEmb",     [hp.NWords; hp.EmbeddingDim],       initWeights)
        EmbToUpdate   = mb.Param ("EmbToUpdate",   [hp.EmbeddingDim; hp.EmbeddingDim], initWeights)
        EmbToReset    = mb.Param ("EmbToReset",    [hp.EmbeddingDim; hp.EmbeddingDim], initWeights)
        EmbToHidden   = mb.Param ("EmbToHidden",   [hp.EmbeddingDim; hp.EmbeddingDim], initWeights)
        StateToUpdate = mb.Param ("StateToUpdate", [hp.EmbeddingDim; hp.EmbeddingDim], initWeights)
        StateToReset  = mb.Param ("StateToReset",  [hp.EmbeddingDim; hp.EmbeddingDim], initWeights)
        StateToHidden = mb.Param ("StateToHidden", [hp.EmbeddingDim; hp.EmbeddingDim], initWeights)
        UpdateBias    = mb.Param ("UpdateBias",    [hp.EmbeddingDim],                  initBias)
        ResetBias     = mb.Param ("ResetBias",     [hp.EmbeddingDim],                  initBias)
        HiddenBias    = mb.Param ("HiddenBias",    [hp.EmbeddingDim],                  initBias)
        StateToWord   = mb.Param ("StateToWord",   [hp.EmbeddingDim; hp.NWords],       initWeights)
        WordBias      = mb.Param ("WordBias",      [hp.NWords],                        initBias)
        HyperPars     = hp
    }

    let sigmoid (z: ExprT) = 1.0f / (1.0f + exp (-z))
    let softmax (z: ExprT) = exp z / (Expr.sumKeepingAxis 1 (exp z) + 1e-4f)

    let pred (pars: Pars) (initial: ExprT) (words: ExprT) =
        // words            [smpl, pos]
        // input            [smpl, step]           
        // initial          [smpl, recUnit]
        // state            [smpl, step, recUnit]
        // output           [smpl, step, word]

        let nBatch = words.Shape.[0]
        let nSteps = words.Shape.[1] - 1
        let embeddingDim = pars.HyperPars.EmbeddingDim

        // build loop
        let inputSlice = Expr.var<int>    "InputSlice"  [nBatch] 
        let prevState  = Expr.var<single> "PrevState"   [nBatch; embeddingDim]

        let bcEmbUnit = Expr.arange<int> embeddingDim |> Expr.padLeft |> Expr.broadcast [nBatch; embeddingDim]
        let bcInputSlice = inputSlice |> Expr.padRight |> Expr.broadcast [nBatch; embeddingDim]
        let emb = pars.WordToEmb |> Expr.gather [Some bcInputSlice; Some bcEmbUnit]

        let update = emb .* pars.EmbToUpdate + prevState .* pars.StateToUpdate + Expr.padLeft pars.UpdateBias |> sigmoid
        let reset  = emb .* pars.EmbToReset  + prevState .* pars.StateToReset  + Expr.padLeft pars.ResetBias  |> sigmoid
        let hidden = emb .* pars.EmbToHidden + (prevState * reset) .* pars.StateToHidden                      |> tanh
        let state  = (1.0f - update) * hidden + update * prevState
        let output = state .* pars.StateToWord + Expr.padLeft pars.WordBias |> softmax

        let chState, chOutput = "State", "Output"
        let loopSpec = {
            Expr.Length = nSteps
            Expr.Vars = Map [Expr.extractVar inputSlice, Expr.SequenceArgSlice {ArgIdx=0; SliceDim=1}
                             Expr.extractVar prevState, 
                                    Expr.PreviousChannel {Channel=chState; Delay=SizeSpec.fix 1; InitialArg=1}]
            Expr.Channels = Map [chState,  {LoopValueT.Expr=state;  LoopValueT.SliceDim=1}
                                 chOutput, {LoopValueT.Expr=output; LoopValueT.SliceDim=1}]    
        }

        let input   = words.[*, 0 .. nSteps-1]
        let initial = initial |> Expr.reshape [nBatch; SizeSpec.fix 1; embeddingDim]
        let states  = Expr.loop loopSpec chState  [input; initial]
        let outputs = Expr.loop loopSpec chOutput [input; initial]
        let finalState = states.[*, nSteps-1, *]

        let target     = words.[*, 1 .. nSteps]
        let targetProb = outputs |> Expr.gather [None; None; Some target]      
        let targetProb = targetProb |> Expr.cage (Some 0.0001f, Some 100.0f)      
        let loss       = -log targetProb |> Expr.mean

        finalState, outputs, loss



module GRUTrain =
    //let EmbeddingDim = 48
    let EmbeddingDim = 128
    let NBatch       = 250

    let train (dataset: TrnValTst<WordSeq>) =
        let mb = ModelBuilder<single> ("M")

        let embeddingDim = mb.Size "embeddingDim"
        let nWords       = mb.Size "nWords"
        let nBatch       = mb.Size "nBatch"
        let nSteps       = mb.Size "nSteps"

        let words   = mb.Var<int>     "Words"   [nBatch; nSteps]
        let initial = mb.Var<single>  "Initial" [nBatch; embeddingDim]

        let model = GRULang.pars (mb.Module "GRULang") {
            NWords       = nWords
            EmbeddingDim = embeddingDim
        }      

        let final, pred, loss = (initial, words) ||> GRULang.pred model

        let mi = mb.Instantiate (DevCuda, Map [nWords,       Dataset.VocSize
                                               embeddingDim, EmbeddingDim])

        let smplVarEnv (stateOpt: ArrayNDT<single> option) (smpl: WordSeq) =
            let nBatch = smpl.Words.Shape.[0]
            let state =
                match stateOpt with
                | Some state -> 
                    if state.Shape.[0] > nBatch then state.[0 .. nBatch-1, *]
                    else state
                | None -> 
                    ArrayNDCuda.zeros<single> [nBatch; EmbeddingDim] :> ArrayNDT<_>
            VarEnv.ofSeq [words, smpl.Words :> IArrayNDT; initial, state :> IArrayNDT]
                          
        //let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv GradientDescent.New GradientDescent.DefaultCfg
        let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv Adam.New Adam.DefaultCfg

        let trainCfg = {
            Train.defaultCfg with
                //MinIters  = Some 1000
                //MaxIters  = Some 10
                //MaxIters  = Some 1000
                BatchSize = NBatch
                //LearningRates = [1e-4; 1e-5]
                CheckpointDir = Some "."
                BestOn    = Training
        }
        Train.train trainable dataset trainCfg |> ignore




        //let dLoss = Deriv.compute loss
        //let dLossDW = dLoss |> Deriv.ofVar model.StateToHidden |> Expr.sum
        //printfn "loss:\n%A" loss
        //let lossFn = mi.Func (loss) |> arg3 initial input target
        //let dLossDInitialFn = mi.Func (loss, dLossDInitial) |> arg3 initial input target
//        for i=1 to 1 do
//            printfn "Calculating loss:"
//            let lossVal = lossFn zeroInitial dataset.Trn.[0 .. NBatch-1].Words dataset.Trn.[0 .. NBatch-1].Words
//            printfn "loss=%f" (lossVal |> ArrayND.value)

        //let tr = Trace.startSessionWithRng "trc" (Some 900, None) 

//        for smpl in dataset.Trn.Batches NBatch do
//            let zeroInitial = ArrayNDCuda.zeros<single> [smpl.Words.Shape.[0]; NRecurrent]
//            printfn "Calculating and dloss:"
//            //let lossVal, dLossVal = dLossDInitialFn zeroInitial dataset.Trn.[0 .. NBatch-1].Words dataset.Trn.[0 .. NBatch-1].Words
//            let lossVal, dLossVal = dLossDInitialFn zeroInitial smpl.Words smpl.Words
//            printfn "loss=%f   dloss/dInitial=%f" (lossVal |> ArrayND.value) (dLossVal |> ArrayND.value)

        //let ts = tr.End ()
        //ts |> Trace.dumpToFile "trc.txt"
