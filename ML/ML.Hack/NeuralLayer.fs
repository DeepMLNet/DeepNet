namespace Models.Neural

open DeepNet.Utils
open Tensor
open Tensor.Expr



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
        static member standard: HyperPars<'T> = {
            NInput     = SizeSpec.fix 0L
            NOutput    = SizeSpec.fix 0L
            ActFunc    = ActFunc.Tanh
            L1Regul    = 0.0
            L2Regul    = 0.0
        }


    /// Neural layer parameters.
    type Pars<'T> = {
        /// expression for the weights
        Weights:        Expr<'T>
        /// expression for the biases
        Bias:           Expr<'T>
        /// hyper-parameters
        HyperPars:      HyperPars<'T>
    }

    let internal initWeights seed (shp: int64 list) : Tensor<'T> = 
        let rng = System.Random seed
        let fanOut = shp.[0] |> float
        let fanIn = shp.[1] |> float
        let r = 4.0 * sqrt (6.0 / (fanIn + fanOut)) 
        HostTensor.randomUniform rng (conv<'T> -r, conv<'T> r) shp
        
    let internal initBias seed (shp: int64 list) : Tensor<'T> =
        HostTensor.zeros shp

    /// Creates the parameters for the neural-layer in the supplied
    /// model builder `mb` using the hyper-parameters `hp`.
    /// The weights are initialized using random numbers from a uniform
    /// distribution with support [-r, r] where
    /// r = 4 * sqrt (6 / (hp.NInput + hp.NOutput)).
    /// The biases are initialized to zero.
    let pars (ctx: Context) (hp: HyperPars<'T>) = {
        //Weights   = mb.Param ("Weights", [hp.NOutput; hp.NInput], initWeights)
        //Bias      = mb.Param ("Bias",    [hp.NOutput],            initBias)
        Weights     = Expr.var (Var<'T> (ctx / "Weights", [hp.NOutput; hp.NInput]))
        Bias        = Expr.var (Var<'T> (ctx / "Bias",    [hp.NOutput]))
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
        let act = input .* pars.Weights.T + pars.Bias
        ActFunc.apply pars.HyperPars.ActFunc act

    /// The regularization term for this layer.
    let regul (pars: Pars<'T>) =
        let l1reg = Regul.lRegul pars.HyperPars.L1Regul 1.0 pars.Weights
        let l2reg = Regul.lRegul pars.HyperPars.L2Regul 2.0 pars.Weights
        l1reg + l2reg



module User =

    let build() =
        let ctx = Context.root HostTensor.Dev
        let nSamples = SizeSpec.symbol "nSamples"
        let nFeatures = SizeSpec.symbol "nFeatures"
        let inputVar = Var<float32> (ctx / "input", [nSamples; nFeatures])
        let input = Expr.var inputVar
        let hyperPars = NeuralLayer.HyperPars.standard
        let pars = NeuralLayer.pars ctx hyperPars
        let pred = NeuralLayer.pred pars input
        printfn "%A" pred

        // Now, no way to initialize pars.
        // Cannot create data when expr is constructed, because sizes are not known.
        // 
        ()