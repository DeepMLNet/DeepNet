namespace Models.Neural

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Backend





/// Activation functions.
[<RequireQualifiedAccess>]
type ActFunc =
    /// tanh transfer function
    | Tanh
    /// sigmoid transfer function
    | Sigmoid        
    /// soft-max transfer function
    | SoftMax
    /// logarithm of soft-max transfer function
    | LogSoftmax
    /// rectifier unit transfer function
    | Relu
    /// no transfer function
    | Identity
with 

    /// identity activation function
    static member id (x: Expr<'T>) =
        x

    /// tanh activation function
    static member tanh (x: Expr<'T>) =
        tanh x

    /// sigmoid activation function
    static member sigmoid (x: Expr<'T>) =
        let one = Expr.scalar x.Dev (conv<'T> 1)
        let two = Expr.scalar x.Dev (conv<'T> 2)
        (tanh (x / two) + one) / two

    /// Soft-max activation function.
    /// The second dimension enumerates the possible classes.
    static member softmax (x: Expr<'T>) =
        let c = x |> Expr.maxKeepingAxis 1
        let y = exp (x - c)
        y / Expr.sumKeepingAxis 1 y

    /// Natural logarithm of soft-max activation function.
    /// The second dimension enumerates the possible classes.
    static member logSoftmax (x: Expr<'T>) =
        let c = x |> Expr<_>.maxKeepingAxis 1
        x - c - log (Expr<_>.sumKeepingAxis 1 (exp (x - c))) 

    /// Rectifier Unit function: max(x, 0)
    static member relu (x: Expr<'T>) =
        let zeros = Expr.zeros x.Dev x.Shape
        Expr.maxElemwise zeros x

    /// applies the specified activation function
    static member apply actFunc (x: Expr<'T>) =
        match actFunc with
        | Tanh       -> ActFunc.tanh x
        | Sigmoid    -> ActFunc.sigmoid x
        | SoftMax    -> ActFunc.softmax x
        | LogSoftmax -> ActFunc.logSoftmax x
        | Relu       -> ActFunc.relu x
        | Identity   -> ActFunc.id x



/// A layer that calculates the loss between predictions and targets using a difference metric.
module LossLayer =

    /// Difference metrics.
    type Measure =
        /// mean-squared error 
        | MSE 
        /// binary cross entropy
        | BinaryCrossEntropy
        /// multi-class cross entropy
        | CrossEntropy
        /// soft-max followed by multi-class cross entropy
        /// (use with identity transfer function in last layer)
        | SoftMaxCrossEntropy

    /// Returns an expression for the loss given the loss measure `lm`, the predictions
    /// `pred` and the target values `target`.
    /// If the multi-class cross entropy loss measure is used then
    /// pred.[smpl, cls] must be the predicted probability that the sample
    /// belong to class cls and target.[smpl, cls] must be 1 if the sample
    /// actually belongs to class cls and 0 otherwise.
    let loss lm (pred: Expr<'T>) (target: Expr<'T>) =
        // pred   [smpl, cls]
        // target [smpl, cls]
        let one = Expr.scalar pred.Dev (conv<'T> 1)
        let two = Expr.scalar pred.Dev (conv<'T> 2)
        match lm with
        | MSE -> 
            (pred - target) ** two
            |> Expr.mean
        | BinaryCrossEntropy ->
            -(target * log pred + (one - target) * log (one - pred))
            |> Expr.mean
        | CrossEntropy ->
            -target * log pred |> Expr.sumAxis 1 |> Expr.mean
        | SoftMaxCrossEntropy ->
            let c = pred |> Expr.maxKeepingAxis 1
            let logProb = pred - c - log (Expr.sumKeepingAxis 1 (exp (pred - c)))             
            -target * logProb |> Expr.sumAxis 1 |> Expr.mean


/// Regularization expressions.
module Regul =

    let lRegul (fac: float) (q: float) (w: Expr<'T>) =
        if fac <> 0.0 then 
            Expr.scalar w.Dev (conv<'T> fac) * Expr.sum (abs w *** Expr.scalar w.Dev (conv<'T> q))
        else
            Expr.scalar w.Dev (conv<'T> 0)



/// A layer of neurons (perceptrons).
module NeuralLayer = 

    /// Neural layer hyper-parameters.
    type HyperPars<'T> = {
        /// number of inputs
        NInput:             SizeSpec
        /// number of outputs
        NOutput:            SizeSpec
        /// transfer (activation) function
        ActFunc:            ActFunc
        /// l1 regularization weight
        L1Regul:            float 
        /// l2 regularization weight
        L2Regul:            float 
    } with 
        static member standard nInput nOutput : HyperPars<'T> = {
            NInput     = nInput
            NOutput    = nOutput
            ActFunc    = ActFunc.Tanh
            L1Regul    = 0.0
            L2Regul    = 0.0
        }


    /// Neural layer parameters.
    type Pars<'T> = {
        /// expression for the weights
        Weights:        Var<'T>
        /// expression for the biases
        Bias:           Var<'T>
        /// hyper-parameters
        HyperPars:      HyperPars<'T>
    }

    let internal initWeights (rng: System.Random) (weights: Tensor<'T>) = 
        let fanOut = weights.Shape.[0] |> float
        let fanIn = weights.Shape.[1] |> float
        let r = 4.0 * sqrt (6.0 / (fanIn + fanOut))
        let weightsHost = 
            HostTensor.randomUniform rng (conv<'T> -r, conv<'T> r) weights.Shape
        weights.TransferFrom weightsHost
        
    let internal initBias (rng: System.Random) (bias: Tensor<'T>) =
        bias.FillZeros ()

    /// Creates the parameters for the neural-layer in the supplied
    /// model builder `mb` using the hyper-parameters `hp`.
    /// The weights are initialized using random numbers from a uniform
    /// distribution with support [-r, r] where
    /// r = 4 * sqrt (6 / (hp.NInput + hp.NOutput)).
    /// The biases are initialized to zero.
    let pars (ctx: Context) rng (hp: HyperPars<'T>) = {
        Weights     = Var<'T> (ctx / "Weights", [hp.NOutput; hp.NInput]) |> Var<_>.toPar (initWeights rng)
        Bias        = Var<'T> (ctx / "Bias",    [hp.NOutput])            |> Var<_>.toPar (initBias rng)
        HyperPars   = hp
    }

    /// Returns an expression for the output (predictions) of the
    /// neural layer with parameters `pars` given the input `input`.
    /// If the soft-max transfer function is used, the normalization
    /// is performed over axis 0.
    let pred (pars: Pars<'T>) (input: Expr<'T>) =
        // weights [outUnit, inUnit]
        // bias    [outUnit]
        // input   [smpl, inUnit]
        // pred    [smpl, outUnit]
        let act = input .* (Expr pars.Weights).T + Expr pars.Bias
        ActFunc.apply pars.HyperPars.ActFunc act

    /// The regularization term for this layer.
    let regul (pars: Pars<'T>) =
        let l1reg = Regul.lRegul pars.HyperPars.L1Regul 1.0 (Expr pars.Weights)
        let l2reg = Regul.lRegul pars.HyperPars.L2Regul 2.0 (Expr pars.Weights)
        l1reg + l2reg



module User =

    let build() =
        let rng = System.Random 1
        
        // make training data
        let x = HostTensor.linspace -2.0f 2.0f 100L
        let y1 = 3.0f + 7.0f * x ** 2.0f
        let y2 = 1.0f + 2.0f * x ** 3.0f + 4.0f * x ** 4.0f
        let y = Tensor.concat 1 [y1.[*, NewAxis]; y2.[*, NewAxis]]
        printfn "x: %A" x
        printfn "y: %A" y
        let exps = HostTensor.arange 0.0f 1.0f 10.0f
        let f = x.[*, NewAxis] ** exps.[NewAxis, *]
        printfn "f: %A" f.Shape

        // context
        let ctx = Context.root HostTensor.Dev
        
        // symbolic sizes
        let nSamples = SizeSpec.symbol "nSamples"
        let nFeatures = SizeSpec.symbol "nFeatures"
        let nOutputs = SizeSpec.symbol "nOutputs"

        // model
        let inputVar = Var<float32> (ctx / "input", [nSamples; nFeatures])
        let targetVar = Var<float32> (ctx / "target", [nSamples; nOutputs])
        let input = Expr inputVar
        let target = Expr targetVar
        let hyperPars = NeuralLayer.HyperPars.standard nFeatures nOutputs
        let pars = NeuralLayer.pars ctx rng hyperPars
        let pred = NeuralLayer.pred pars input
        let loss = LossLayer.loss LossLayer.MSE pred target
        printfn "loss: %s\n" (loss.ToString())

        // substitute symbol sizes
        let sizeEnv = Map [
            SizeSpec.extractSymbol nFeatures, SizeSpec.fix f.Shape.[1]
            SizeSpec.extractSymbol nOutputs, SizeSpec.fix y.Shape.[1]
        ]
        let loss = loss |> Expr.substSymSizes sizeEnv
        printfn "substituted: %s\n" (loss.ToString())

        // parameter set
        let parSet = ParSet.fromExpr ContextPath.root loss
        let parSetInst = ParSet.inst ContextPath.root parSet
        let loss = parSetInst.Use loss 
        printfn "with ParSet: %s\n" (loss.ToString())

        // use optimizer
        let opt = Optimizers.Adam.Adam.make (loss, parSetInst)
        let minStep = opt.Step
        let minLossStep = minStep |> EvalUpdateBundle.addExpr loss
        printfn "Minimiziation step: %A\n" minStep

        // evaluate using training data
        let varEnv = 
            VarEnv.empty
            |> VarEnv.add inputVar f
            |> VarEnv.add targetVar y
            |> parSetInst.Use
        let lossVal = loss |> Expr.eval varEnv
        printfn "loss value: %A" lossVal

        // perform training step
        printfn "training..."
        for i in 1..10000 do
            let results = minLossStep |> EvalUpdateBundle.exec varEnv
            printf "step %d loss value: %f             \r" i (results.Get loss).Value
        printfn ""

        ()