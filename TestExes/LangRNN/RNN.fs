namespace LangRNN

open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda

open MLModels


/// A layer of recurrent neurons with an output layer on top.
module RecurrentLayer =

    /// Neural layer hyper-parameters.
    type HyperPars = {
        /// number of inputs
        NInput:                     SizeSpecT
        /// number of recurrent units
        NRecurrent:                 SizeSpecT
        /// number of output units
        NOutput:                    SizeSpecT
        /// recurrent transfer (activation) function
        RecurrentActivationFunc:    ActivationFunc
        /// output transfer (activation) function
        OutputActivationFunc:       ActivationFunc
        /// input weights trainable
        InputWeightsTrainable:      bool
        /// recurrent weights trainable
        RecurrentWeightsTrainable:  bool
        /// output weights trainable
        OutputWeightsTrainable:     bool
        /// recurrent bias trainable
        RecurrentBiasTrainable:     bool
        /// output bias trainable
        OutputBiasTrainable:        bool
    }

    /// default hyper-parameters
    let defaultHyperPars = {
        NInput                      = SizeSpec.fix 0
        NRecurrent                  = SizeSpec.fix 0
        NOutput                     = SizeSpec.fix 0
        RecurrentActivationFunc     = Tanh
        OutputActivationFunc        = SoftMax
        InputWeightsTrainable       = true
        RecurrentWeightsTrainable   = true
        OutputWeightsTrainable      = true
        RecurrentBiasTrainable      = true
        OutputBiasTrainable         = true
    }

    /// Neural layer parameters.
    type Pars = {
        /// expression for input-to-recurrent weights 
        InputWeights:      ExprT 
        /// expression for recurrent-to-recurrent weights
        RecurrentWeights:  ExprT 
        /// expression for recurrent-to-output weights
        OutputWeights:     ExprT
        /// expression for the recurrent bias
        RecurrentBias:     ExprT 
        /// expression for the output bias
        OutputBias:        ExprT 
        /// hyper-parameters
        HyperPars:         HyperPars
    }

    let internal initWeights seed (shp: int list) : ArrayNDHostT<'T> = 
        let fanIn = float shp.[1] 
        let r = 1.0 / sqrt fanIn       
        (System.Random seed).SeqDouble(-r, r)
        |> Seq.map conv<'T>
        |> ArrayNDHost.ofSeqWithShape shp
        
    let internal initBias seed (shp: int list) : ArrayNDHostT<'T> =
        Seq.initInfinite (fun _ -> conv<'T> 0)
        |> ArrayNDHost.ofSeqWithShape shp

    /// Creates the parameters.
    let pars (mb: ModelBuilder<_>) hp = {
        InputWeights     = mb.Param ("InputWeights",     [hp.NRecurrent; hp.NInput],     initWeights)
        RecurrentWeights = mb.Param ("RecurrentWeights", [hp.NRecurrent; hp.NRecurrent], initWeights)
        OutputWeights    = mb.Param ("OutputWeights",    [hp.NOutput;    hp.NRecurrent], initWeights)
        RecurrentBias    = mb.Param ("RecurrentBias",    [hp.NRecurrent],                initBias)
        OutputBias       = mb.Param ("OutputBias",       [hp.NOutput],                   initBias)
        HyperPars        = hp
    }

    /// Predicts an output sequence by running the RNN over the input sequence.
    /// Argument shapes: input[smpl, step, inUnit].
    /// Output shapes:  output[smpl, step, outUnit].
    let pred (pars: Pars) (input: ExprT) =
        // inputWeights     [recUnit, inUnit]
        // recurrentWeights [recUnit, recUnit]
        // outputWeights    [outUnit, recUnit]
        // recurrentBias    [recUnit]
        // outputBias       [recUnit]
        // input            [smpl, step, inUnit]
        // state            [smpl, step, recUnit]
        // output           [smpl, step, outUnit]

        // gate gradients
        let inputWeights     = Util.gradGate pars.HyperPars.InputWeightsTrainable     pars.InputWeights
        let recurrentWeights = Util.gradGate pars.HyperPars.RecurrentWeightsTrainable pars.RecurrentWeights
        let outputWeights    = Util.gradGate pars.HyperPars.OutputWeightsTrainable    pars.OutputWeights
        let recurrentBias    = Util.gradGate pars.HyperPars.RecurrentBiasTrainable    pars.RecurrentBias
        let outputBias       = Util.gradGate pars.HyperPars.OutputBiasTrainable       pars.OutputBias

        let nBatch = input.Shape.[0]
        let nSteps = input.Shape.[1]

        // build loop
        let inputWeightsLoop     = Expr.var<single> "InputWeightsLoop"     inputWeights.Shape
        let recurrentWeightsLoop = Expr.var<single> "RecurrentWeightsLoop" recurrentWeights.Shape
        let outputWeightsLoop    = Expr.var<single> "OutputWeightsLoop"    outputWeights.Shape
        let recurrentBiasLoop    = Expr.var<single> "RecurrentBiasLoop"    recurrentBias.Shape
        let outputBiasLoop       = Expr.var<single> "OutputBiasLoop"       outputBias.Shape

        let inputSlice  = Expr.var<single> "InputSlice"  [nBatch; pars.HyperPars.NInput]
        let prevState   = Expr.var<single> "PrevState"   [nBatch; pars.HyperPars.NRecurrent]

        let state =
            inputSlice .* inputWeightsLoop.T + prevState .* recurrentWeightsLoop.T + recurrentBiasLoop
            |> ActivationFunc.apply pars.HyperPars.RecurrentActivationFunc
        let output =
            state .* outputWeightsLoop.T + outputBiasLoop
            |> ActivationFunc.apply pars.HyperPars.OutputActivationFunc

        let chState, chOutput = "State", "Output"
        let loopSpec = {
            Expr.Length = nSteps
            Expr.Vars = Map [Expr.extractVar inputSlice, Expr.SequenceArgSlice {ArgIdx=0; SliceDim=1}
                             Expr.extractVar state, 
                                    Expr.PreviousChannel {Channel=chState; Delay=SizeSpec.fix 1; InitialArg=2}
                             Expr.extractVar inputWeightsLoop,     Expr.ConstArg 2
                             Expr.extractVar recurrentWeightsLoop, Expr.ConstArg 3
                             Expr.extractVar outputWeightsLoop,    Expr.ConstArg 4
                             Expr.extractVar recurrentBiasLoop,    Expr.ConstArg 5
                             Expr.extractVar outputBiasLoop,       Expr.ConstArg 6]
            Expr.Channels = Map [chState,  {LoopValueT.Expr=state;  LoopValueT.SliceDim=1}
                                 chOutput, {LoopValueT.Expr=output; LoopValueT.SliceDim=1}]    
        }

        let initialState = Expr.zeros<single> state.Shape
        let outputs = Expr.loop loopSpec chOutput [input; initialState; 
                                                   inputWeights; recurrentWeights; outputWeights; 
                                                   recurrentBias; outputBias]

        outputs


