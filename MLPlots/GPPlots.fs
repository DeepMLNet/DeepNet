namespace MLPlots
open SymTensor.Compiler.Cuda
open RProvider.graphics
open Models
open ArrayNDNS
open SymTensor
open RProvider

type GPPlots = 
    
    /// Creates num samples from in range minValue to maxValue with constant distance.
    /// Calculates mean covariance and standerdDeviation of these samples given a Gaussian process
    /// with covarianceKernel kernel, training noise Sigma trainVAlues trnX and trainTargets trnT
    static member predictGP hyperPars (sigmaNs: ArrayNDT<single>) (trnX: ArrayNDT<single>) 
            (trnT: ArrayNDT<single>) (minValue, maxValue) nPoints =

        let mb = ModelBuilder<single> "GP"

        let nTrnSmpls = mb.Size "nTrnSmpls"
        let nInput = mb.Size "nInput"
        let sigNs = mb.Var "sigs" [nTrnSmpls]
        let x = mb.Var  "x" [nTrnSmpls]
        let t = mb.Var  "t" [nTrnSmpls]
        let inp = mb.Var  "inp" [nInput]
        let pars = GaussianProcess.pars (mb.Module "GaussianProcess") hyperPars   
        let mi = mb.Instantiate (DevCuda,
                                 Map[nTrnSmpls, trnX.NElems
                                     nInput,    nPoints])

        let mean, cov = GaussianProcess.predict pars x t sigNs inp      
        let stdev = cov |> Expr.diag |> Expr.sqrtt
        
        let meanCovStdFn = mi.Func (mean, cov, stdev) |> arg4 x t sigNs inp
        
        let sX = ArrayNDHost.linSpaced minValue maxValue nPoints |> ArrayNDCuda.toDev
        let sMean, sCov, sStd = meanCovStdFn trnX trnT sigmaNs sX
        sX, sMean, sCov, sStd

    /// Plots a GAussian Process with covarianceKernel kernel, training noise Sigma trainVAlues trnX and trainTargets trnT.
    /// Step is the distance between two sample, smaller step => hihger plot smoothness and accuraccy, longer plot creation
    /// Returns a function (unit -> unit) which starts the plot when applied
    static member simplePlot (hyperPars, trnSigmas: ArrayNDT<single>, trnX: ArrayNDT<single>, trnT: ArrayNDT<single>, 
                              ?nPoints, ?minX, ?maxX, ?minY, ?maxY) =
        
        let nPoints = defaultArg nPoints 20
        let trnDist = ArrayND.max trnX - ArrayND.min trnX |> ArrayND.value
        let minValue = defaultArg minX ((trnX |> ArrayND.min |> ArrayND.value) - trnDist * 0.1f)
        let maxValue = defaultArg maxX ((trnX |> ArrayND.max |> ArrayND.value) + trnDist * 0.1f)

        let sX, sMean, sCov, sStd = GPPlots.predictGP hyperPars trnSigmas trnX trnT (minValue, maxValue) nPoints
        
        let sX = sX |> toFloatList 
        let sMean = sMean |> toFloatList 
        let sStd = sStd |> toFloatList 
        let trainX = trnX |> toFloatList
        let trainT = trnT |> toFloatList
        let upperStdev = List.map2 (fun m s-> m + s) sMean sStd |> List.rev
        let lowerStdev = List.map2 (fun m s-> m - s) sMean sStd
        let revX = List.rev sX

        let minY = defaultArg minY (List.min lowerStdev - abs (List.average lowerStdev) |> single) |> float
        let maxY = defaultArg maxY (List.max upperStdev + abs (List.average lowerStdev) |> single) |> float
        
        namedParams [   
             "x", box sX
             "y", box sMean
             "ylim", box [minY; maxY]
             "col", box "red"
             "type", box "n"
             "xlab", box "x"
             "ylab", box "y"]
        |> R.plot |> ignore
        namedParams [   
             "x", box (sX @ revX)
             "y", box (lowerStdev @ upperStdev)
             "col", box "beige"
             "border" , box "NA"]
        |> R.polygon |>ignore
        namedParams [ 
            "x", box sX
            "y", box sMean
            "col", box "black"
            "type", box "l"
            "size", box 2]
        |> R.lines |>ignore
        namedParams [ 
            "x", box trainX
            "y", box trainT
            "col", box "red"
            "type", box "p"
            "size", box 2]
        |> R.lines |>ignore

        ()

