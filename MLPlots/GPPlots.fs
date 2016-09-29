namespace MLPlots
open SymTensor.Compiler.Cuda
open RProvider.graphics
open Models
open ArrayNDNS
open SymTensor
open RProvider

module GPPlots = 
    
    /// Creates num samples from in range minValue to maxValue with constant distance.
    /// Calculates mean covariance and standerdDeviation of these samples given a Gaussian process
    /// with covarianceKernel kernel, training noise Sigma trainVAlues trnX and trainTargets trnT
    let sampleFromGP (kernel:GaussianProcess.Kernel) (sigmaNs:ArrayNDT<single>) (trnX:ArrayNDT<single>) (trnT:ArrayNDT<single>)  (minValue,maxValue) num=


        let mb = ModelBuilder<single> "GP"

        let nTrnSmpls = mb.Size "nTrnSmpls"
        let nInput = mb.Size "nInput"
        let sigNs = mb.Var "sigs" [nTrnSmpls]
        let x = mb.Var  "x" [nTrnSmpls]
        let t = mb.Var  "t" [nTrnSmpls]
        let inp = mb.Var  "inp" [nInput]
        let pars = GaussianProcess.pars (mb.Module "GaussianProcess") {Kernel = kernel}
        
        let mean, cov = GaussianProcess.regression pars x t sigNs inp

        let mi = mb.Instantiate (DevCuda,
                                 Map[nTrnSmpls, trnX.NElems
                                     nInput,    num])
        
        
        let covMat = Expr.var<single> "covMat" [nInput;nInput]
        let stdev = covMat |> Expr.diag |> Expr.sqrtt
        
        let cmplr = DevCuda.Compiler, CompileEnv.empty
        let mean_cov_fn:(ArrayNDT<single>-> ArrayNDT<single>->ArrayNDT<single>->ArrayNDT<float32> -> ArrayNDT<single>*ArrayNDT<single> )=
             mi.Func (mean, cov) |> arg4 x t sigNs inp
        
        let stdev_fn:(ArrayNDT<single>->ArrayNDT<single> )=
             Func.make cmplr stdev |> arg1 covMat
        let numf32 = single num
        printfn "%A" [minValue..((maxValue-minValue)/(numf32-1.0f))..maxValue]
        let smpls = [minValue..((maxValue-minValue)/(numf32-1.0f))..maxValue] |> ArrayNDHost.ofList  |>ArrayNDCuda.toDev
        let smean,scov = mean_cov_fn  trnX trnT sigmaNs smpls
        let sstdev = stdev_fn scov
        smpls, smean, scov, sstdev
    
    /// Plots a GAussian Process with covarianceKernel kernel, training noise Sigma trainVAlues trnX and trainTargets trnT.
    /// Step is the distance between two sample, smaller step => hihger plot smoothness and accuraccy, longer plot creation
    /// Returns a function (unit -> unit) which starts the plot when applied
    let simplePlot (kernel:GaussianProcess.Kernel) (trnSigmas:ArrayNDT<single>) (trnX:ArrayNDT<single>) (trnT:ArrayNDT<single>)  step  ()=
        let minValue = trnX |> ArrayND.min |> ArrayND.allElems |> Seq.head 
        let maxValue = trnX |> ArrayND.max |> ArrayND.allElems |> Seq.head 
        let numSmpls = (maxValue - minValue) / step + 1.0f
        let smpls,mean_smpls, _, stdev_smpls = sampleFromGP kernel trnSigmas trnX trnT  (minValue- 1.0f,maxValue+ 1.0f) (int numSmpls)
        
        let samples = smpls |> toFloatList 
        let mean = mean_smpls |> toFloatList 
        let stdev = stdev_smpls |> toFloatList 
        let trainX = trnX |> toFloatList
        let trainT = trnT |> toFloatList
        let upperStdev = List.map2 (fun m s-> m + s) mean stdev |> List.rev
        let lowerStdev = List.map2 (fun m s-> m - s) mean stdev
        let revsamples = List.rev samples

        let yLim = [(lowerStdev |> List.min) - abs(lowerStdev |> List.average);(upperStdev |> List.max) +  abs(lowerStdev |> List.average)]
        
        namedParams [   
            "x", box samples;
             "y", box mean;
             "ylim", box yLim;
             "col", box "red";
             "type", box "n"]
        |> R.plot |> ignore
        namedParams [   
            "x", box (samples @ revsamples);
             "y", box (lowerStdev@ upperStdev);
             "col", box "beige";
             "border" , box "NA"]
        |> R.polygon |>ignore
        namedParams [ 
            "x", box samples;
            "y", box mean;
            "col", box "black";
            "type", box "l";
            "size", box 2]
        |> R.lines |>ignore
        namedParams [ 
            "x", box trainX;
            "y", box trainT;
            "col", box "red";
            "type", box "p";
            "size", box 2]
        |> R.lines |>ignore
        ()
