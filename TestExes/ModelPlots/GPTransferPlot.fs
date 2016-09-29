namespace ModelPlots
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open RProvider
open RProvider.ggplot2
open Basics
open GPAct
open System
open Models

module GPTransferPlot =
//    let kse =
//        let gp = ElemExpr.idx 0   
//        let trn_smpl1 = ElemExpr.idx 1
//        let trn_smpl2 = ElemExpr.idx 2
//        let l = ElemExpr.argElem<single> 0
//        let s = ElemExpr.argElem<single> 1
//        let x = ElemExpr.argElem<single> 2
//        let y = ElemExpr.argElem<single> 3
//        let kse =
//            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) ) +
//            ElemExpr.ifThenElse trn_smpl1 trn_smpl2 (s [gp; trn_smpl1] *** 2.0f) (ElemExpr.scalar 0.0f)
//        kse
//    let ksenoSigma =
//        let gp = ElemExpr.idx 0   
//        let trn_smpl1 = ElemExpr.idx 1
//        let trn_smpl2 = ElemExpr.idx 2
//        let l = ElemExpr.argElem<single> 0
//        let x = ElemExpr.argElem<single> 1
//        let y = ElemExpr.argElem<single> 2
//        let kse =
//            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) )
//        kse
//
//    let CovMat lengthscales trnSigmas xVect yVect useSigma = 
//        let nGps = (Expr.shapeOf lengthscales).[0]
//        let sizeX = (Expr.shapeOf xVect).[1]
//        let sizeY = (Expr.shapeOf yVect).[1]
//        if useSigma then
//            Expr.elements [nGps; sizeX; sizeY] kse [lengthscales; trnSigmas; xVect; yVect]
//        else
//            Expr.elements [nGps; sizeX; sizeY] ksenoSigma [lengthscales; xVect; yVect]
//
//    let GPRegression lengthscales trnSigmas trnX trnT inX =
//        let nGps = (Expr.shapeOf lengthscales).[0]
//        let sizeX = (Expr.shapeOf trnX).[1]
//        let sizeY = (Expr.shapeOf inX).[1]
//        let K = CovMat lengthscales trnSigmas trnX trnX true
//        let Kstar = CovMat lengthscales trnSigmas trnX inX false
//        let Kstarstar = CovMat lengthscales trnSigmas inX inX false
//        let K_inv = K |> Expr.invert
//        let KstarT = Kstar |> Expr.transpose
//        let mean = KstarT .* K_inv .* trnT
//        let cov = Kstarstar - KstarT .* K_inv .* Kstar
//
//        mean,cov
    
    ///Sample num points from several gaussian processes (e.g. one GPTransfer Layer)
    /// With input values between minValue and maxValue
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

    let GPTransferTest device =
        
        let seed = 1
        let rand = Random seed
        let ntraining = 10
        let ninput = 20

        let trn_x_list =  TestFunctions.randomSortedListOfLength rand (-5.0f,5.0f) ntraining 
        let trn_x_host = trn_x_list |> ArrayNDHost.ofList
        let trn_t_list = trn_x_list |>  TestFunctions.randPolynomial rand
        let trn_t_host = trn_t_list |> ArrayNDHost.ofList

        let sigmaNs_host = (ArrayNDHost.ones<single> [ntraining]) * sqrt 0.1f

        //transfer train parametters to device (Host or GPU)
        let trn_x_val = trn_x_host  |> TestFunctions.post device
        let trn_t_val = trn_t_host  |> TestFunctions.post device
        let sigmaNs_val = sigmaNs_host  |> TestFunctions.post device

        printfn "Trn_x =\n%A" trn_x_host
        printfn "Trn_t =\n%A" trn_t_host
        let kernel = GaussianProcess.SquaredExponential (1.0f,1.0f)
        let range = (-0.5f,0.5f)
        let smpls, mean_smpls, cov_smpls, stdev_smpls = sampleFromGP kernel sigmaNs_val trn_x_val trn_t_val range ninput
        printfn "Sample points =\n%A" smpls
        printfn "Sampled means =\n%A" mean_smpls
        printfn "Sampled Covariances =\n%A" cov_smpls
        printfn "Sampled StanderdDeviations =\n%A" stdev_smpls
    
    let plot (trnSigmas:ArrayNDT<single>) (trnX:ArrayNDT<single>) (trnT:ArrayNDT<single>) (kernel:GaussianProcess.Kernel) step =
        let minValue = trnX |> ArrayND.min |> ArrayND.allElems |> Seq.head
        let maxValue = trnX |> ArrayND.max |> ArrayND.allElems |> Seq.head
        let numSmpls = (maxValue - minValue) / step + 1.0f
        let smpls,mean_smpls, _, stdev_smpls = sampleFromGP kernel trnSigmas trnX trnT  (minValue,maxValue) (int numSmpls)
        let upperStdev = mean_smpls + stdev_smpls
        let lowerStdev = mean_smpls - stdev_smpls
        
        
        ()
    ()


