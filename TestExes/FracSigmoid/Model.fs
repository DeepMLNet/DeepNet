namespace FracSigmoid

open Basics
open ArrayNDNS
open SymTensor


/// A layer of neurons with activation functions given by table.
module TableLayer = 

    type HyperPars = {
        /// number of inputs
        NInput:         SizeSpecT
        /// number of outputs
        NOutput:        SizeSpecT
        /// transfer (activation) function
        Info:           Info
    }

    type Pars = {
        /// expression for the weights
        Weights:        ExprT 
        /// expression for the biases
        Bias:           ExprT 
        /// expression for the fractional execution parameter
        Frac:           ExprT
        /// interpolator
        Interpolator:   InterpolatorT
        /// hyper-parameters
        HyperPars:      HyperPars
    }

    let internal initWeights seed (shp: int list) : ArrayNDHostT<'T> = 
        let fanOut = shp.[0] |> float
        let fanIn = shp.[1] |> float
        let r = 4.0 * sqrt (6.0 / (fanIn + fanOut))
        let rng = System.Random seed
        
        rng.SeqDouble(-r, r)
        |> Seq.map conv<'T>
        |> ArrayNDHost.ofSeqWithShape shp
        
    let internal initBias seed (shp: int list) : ArrayNDHostT<'T> =
        Seq.initInfinite (fun _ -> conv<'T> 0)
        |> ArrayNDHost.ofSeqWithShape shp

    let internal initFrac seed (shp: int list) : ArrayNDHostT<'T> =
        Seq.initInfinite (fun _ -> conv<'T> 0)
        |> ArrayNDHost.ofSeqWithShape shp

    let pars (mb: ModelBuilder<_>) (hp: HyperPars) = 
        // create interpolator
        let info = hp.Info
        let tbl = FracSigmoidTable.generate info
        let dIps =
            match tbl.DPoints with
            | Some (dPntsdN, dPntsdX) ->
                printfn "Using provided derivatives."
                let dIpdN =
                    Interpolator.create (ArrayNDCuda.toDev dPntsdN) 
                        [info.NMin; info.XMin] [info.NMax; info.XMax]
                        [Nearest; Nearest] InterpolateLinearaly None
                let dIpdX =
                    Interpolator.create (ArrayNDCuda.toDev dPntsdX) 
                        [info.NMin; info.XMin] [info.NMax; info.XMax]
                        [Nearest; Nearest] InterpolateLinearaly None
                Some [dIpdN; dIpdX]
            | None -> None   
        let ip = 
            Interpolator.create (ArrayNDCuda.toDev tbl.Points)  
                [info.NMin; info.XMin] [info.NMax; info.XMax]
                [Nearest; Nearest] InterpolateLinearaly dIps   

        {
            Weights      = mb.Param ("Weights", [hp.NOutput; hp.NInput], initWeights)
            Bias         = mb.Param ("Bias",    [hp.NOutput],            initBias)
            Frac         = mb.Param ("Frac",    [hp.NOutput],            initFrac)
            Interpolator = ip
            HyperPars    = hp
        }

    let pred pars (input: ExprT) =
        // weights [outUnit, inUnit]
        // bias    [outUnit]
        // frac    [outUnit]
        // input   [smpl, inUnit]
        // pred    [smpl, outUnit]
        let activation = input .* pars.Weights.T + pars.Bias
        Expr.interpolate2D pars.Interpolator (pars.Frac *** 2.0f) activation

