namespace LangRNN

open DeepNet.Utils
open Tensor
open SymTensor
open SymTensor.Compiler.Cuda

open Models
open Datasets
open Optimizers


module GRULang =

    type HyperPars = {
        NWords:                     SizeSpec
        EmbeddingDim:               SizeSpec
        MultiStepLoss:              bool
    }

    let defaultHyperPars = {
        NWords                      = SizeSpec.fix 0L
        EmbeddingDim                = SizeSpec.fix 0L
        MultiStepLoss               = false
    }

    type Pars = {        
        WordToEmb:       Expr    // [word, embDim]
        EmbToUpdate:     Expr    // [recUnit, embDim]
        EmbToReset:      Expr    // [recUnit, embDim]
        EmbToHidden:     Expr    // [recUnit, embDim]
        StateToUpdate:   Expr    // [recUnit, recUnit]
        StateToReset:    Expr    // [recUnit, recUnit]
        StateToHidden:   Expr    // [recUnit, recUnit]
        UpdateBias:      Expr    // [recUnit]
        ResetBias:       Expr    // [recUnit]
        HiddenBias:      Expr    // [recUnit]
        StateToWord:     Expr    // [word, recUnit]
        WordBias:        Expr    // [word]
        HyperPars:       HyperPars
    }

    let internal initWeights seed (shp: int64 list) = 
        let r = 1.0f / sqrt (single shp.[1])       
        (System.Random seed).SeqDouble (float -r, float r) 
        |> HostTensor.ofSeqWithShape shp 
        |> Tensor<single>.convert
        
    let internal initBias seed (shp: int64 list) =
        HostTensor.zeros<single> shp

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

    let sigmoid (z: Expr) = (tanh (z/2.0f) + 1.0f) / 2.0f
    let softmax (z: Expr) =
        let c = z |> Expr.maxKeepingAxis 1
        let y = exp (z - c)
        y / Expr.sumKeepingAxis 1 y
    let negLogSoftmax (z: Expr) =
        let c = z |> Expr.maxKeepingAxis 1
        c - z + log (Expr.sumKeepingAxis 1 (exp (z - c)) + 1e-6f)
        

    let build (pars: Pars) (initialSlice: Expr) (words: Expr) (genFirstWord: Expr) =
        // words            [smpl, pos]
        // input            [smpl, step]           
        // initial          [smpl, recUnit]
        // genFirstWord     [smpl]
        // state            [smpl, step, recUnit]
        // output           [smpl, step, word]

        let nBatch = words.Shape.[0]
        let nSteps = words.Shape.[1] - 1L
        let embeddingDim = pars.HyperPars.EmbeddingDim
        let nWords = pars.HyperPars.NWords

        let initial      = initialSlice |> Expr.reshape [nBatch; SizeSpec.fix 1L; embeddingDim]
        let genFirstWord = genFirstWord |> Expr.reshape [nBatch; SizeSpec.fix 1L]

        // build loop
        let step       = Expr.var<int>    "Step"        []
        let inputSlice = Expr.var<int>    "InputSlice"  [nBatch] 
        let prevState  = Expr.var<single> "PrevState"   [nBatch; embeddingDim]
        let prevOutput = Expr.var<single> "PrevOutput"  [nBatch; nWords]

        let bcEmbUnit = Expr.arange<int> embeddingDim |> Expr.padLeft |> Expr.broadcast [nBatch; embeddingDim]
        let bcInputSlice = inputSlice |> Expr.padRight |> Expr.broadcast [nBatch; embeddingDim]
        let inpEmb = pars.WordToEmb |> Expr.gather [Some bcInputSlice; Some bcEmbUnit]
        let prvOutEmb = prevOutput .* pars.WordToEmb |> softmax
        let emb = 
            if pars.HyperPars.MultiStepLoss then Expr.ifThenElse (step <<<< 13) inpEmb prvOutEmb            
            else inpEmb

        let update = emb .* pars.EmbToUpdate + prevState .* pars.StateToUpdate           + Expr.padLeft pars.UpdateBias |> sigmoid
        let reset  = emb .* pars.EmbToReset  + prevState .* pars.StateToReset            + Expr.padLeft pars.ResetBias  |> sigmoid
        let hidden = emb .* pars.EmbToHidden + (prevState * reset) .* pars.StateToHidden + Expr.padLeft pars.HiddenBias |> tanh
        let state  = (1.0f - update) * hidden + update * prevState
        let output = state .* pars.StateToWord + Expr.padLeft pars.WordBias
        let pred   = output |> Expr.argMaxAxis 1 
        let logWordProb = output |> negLogSoftmax

        // training loop
        let chState, chLogWordProb, chPred, chOutput = "State", "LogWordProb", "Pred", "Output"
        let loopSpec = 
            if pars.HyperPars.MultiStepLoss then 
              {
                Expr.Length = nSteps
                Expr.Vars = Map [Expr.extractVar inputSlice, Expr.SequenceArgSlice {ArgIdx=0; SliceDim=1}
                                 Expr.extractVar prevOutput,
                                    Expr.PreviousChannel {Channel=chOutput; Delay=SizeSpec.fix 1L; InitialArg=2}
                                 Expr.extractVar prevState, 
                                    Expr.PreviousChannel {Channel=chState; Delay=SizeSpec.fix 1L; InitialArg=1}
                                 Expr.extractVar step, Expr.IterationIndex]
                Expr.Channels = Map [chState,       {LoopValue.Expr=state;       LoopValue.SliceDim=1}
                                     chLogWordProb, {LoopValue.Expr=logWordProb; LoopValue.SliceDim=1}
                                     chOutput,      {LoopValue.Expr=output;      LoopValue.SliceDim=1}]    
              } 
            else 
              {
                Expr.Length = nSteps
                Expr.Vars = Map [Expr.extractVar inputSlice, Expr.SequenceArgSlice {ArgIdx=0; SliceDim=1}
                                 Expr.extractVar prevState, 
                                    Expr.PreviousChannel {Channel=chState; Delay=SizeSpec.fix 1L; InitialArg=1}]
                Expr.Channels = Map [chState,       {LoopValue.Expr=state;       LoopValue.SliceDim=1}
                                     chLogWordProb, {LoopValue.Expr=logWordProb; LoopValue.SliceDim=1}]
              }
        let input         = words.[*, 0L .. nSteps-1L]
        let initialOutput = Expr.zeros<single> [nBatch; SizeSpec.fix 1L; nWords]
        let loopArgs      = if pars.HyperPars.MultiStepLoss then [input; initial; initialOutput] else [input; initial]

        let states        = Expr.loop loopSpec chState       loopArgs
        let logWordProbs  = Expr.loop loopSpec chLogWordProb loopArgs
        let finalState    = states.[*, nSteps-1L, *]
        let target        = words.[*, 1 .. nSteps]
        let loss          = logWordProbs |> Expr.gather [None; None; Some target] |> Expr.mean 

        // generating loop
        let genSteps = SizeSpec.fix 200L
        let genLoopSpec = {
            Expr.Length = genSteps
            Expr.Vars = Map [Expr.extractVar inputSlice, 
                                    Expr.PreviousChannel {Channel=chPred;  Delay=SizeSpec.fix 1L; InitialArg=0}
                             Expr.extractVar prevState, 
                                    Expr.PreviousChannel {Channel=chState; Delay=SizeSpec.fix 1L; InitialArg=1}]
            Expr.Channels = Map [chState,  {LoopValue.Expr=state;  LoopValue.SliceDim=1}
                                 chPred,   {LoopValue.Expr=pred;   LoopValue.SliceDim=1}]    
        }
        let states        = Expr.loop genLoopSpec chState  [genFirstWord; initial; initialSlice]
        let generated     = Expr.loop genLoopSpec chPred   [genFirstWord; initial; initialSlice]
        let genFinalState = states.[*, genSteps-1L, *]

        (finalState, logWordProbs, loss), (genFinalState, generated)



type GRUInst (VocSize:       int64,
              EmbeddingDim:  int64,
              MultiStepLoss: bool) =

    let mb = ModelBuilder<single> ("M")

    let embeddingDim = mb.Size "embeddingDim"
    let nWords       = mb.Size "nWords"
    let nBatch       = mb.Size "nBatch"
    let nSteps       = mb.Size "nSteps"

    let words      = mb.Var<int>     "Words"      [nBatch; nSteps]
    let initial    = mb.Var<single>  "Initial"    [nBatch; embeddingDim]
    let firstWord  = mb.Var<int>     "FirstWord"  [nBatch]

    let model = GRULang.pars (mb.Module "GRULang") {
        NWords        = nWords
        EmbeddingDim  = embeddingDim
        MultiStepLoss = MultiStepLoss
    }      

    let (final, pred, loss), (genFinal, genWords) = (initial, words, firstWord) |||> GRULang.build model

    let mi = mb.Instantiate (DevCuda, Map [nWords,       VocSize
                                           embeddingDim, EmbeddingDim])

    let generateFn = mi.Func<single, int> (genFinal, genWords) |> arg2 initial firstWord
    let processFn  = mi.Func<single>      (final)              |> arg2 initial words                       

    member this.Train (dataset: TrnValTst<WordSeq>) dropStateProb trainCfg =
        let rng = System.Random 1
        let smplVarEnv (stateOpt: Tensor<single> option) (smpl: WordSeq) =
            let nBatch = smpl.Words.Shape.[0]
            let dropState = rng.NextDouble() < dropStateProb 
            let state =
                match stateOpt with
                | Some state when state.Shape.[0] = nBatch && not dropState -> state
                | Some state when state.Shape.[0] > nBatch && not dropState -> state.[0L .. nBatch-1L, *]
                | _ -> CudaTensor.zeros<single> [nBatch; EmbeddingDim] 
            if smpl.Words.Shape.[1] < 2L then failwithf "need more than two steps per sample: %A" smpl.Words.Shape
            VarEnv.ofSeq [words, smpl.Words :> ITensor; initial, state :> ITensor]

        //let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv GradientDescent.New GradientDescent.DefaultCfg
        let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv Adam.New Adam.DefaultCfg
        Train.train trainable dataset trainCfg |> ignore

    member this.Generate seed (startWords: WordSeq) =
        // sw [smpl, step]
        let sw = startWords.Words
        let nBatch, nStart = sw.Shape.[0], sw.Shape.[1]        

        let initial = 
            if seed = 0 then 
                CudaTensor.zeros<single> [nBatch; EmbeddingDim] 
            else
                let rng = System.Random seed 
                HostTensor.randomUniform rng (-0.1f, 0.1f) [nBatch; EmbeddingDim] |> CudaTensor.transfer 
        let primed = 
            // last word of array is not actually processed
            if nStart > 1L then processFn initial sw.[*, 0L .. nStart-1L]
            else initial        

        let final, gen = generateFn primed sw.[*, nStart-1L]
        {Words=gen}




