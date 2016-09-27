namespace ModelPlots
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open RProvider
open RProvider.ggplot2
open Basics

module GPTransferPlot =
    let kse =
        let gp = ElemExpr.idx 0   
        let trn_smpl1 = ElemExpr.idx 1
        let trn_smpl2 = ElemExpr.idx 2
        let l = ElemExpr.argElem<single> 0
        let s = ElemExpr.argElem<single> 1
        let x = ElemExpr.argElem<single> 2
        let y = ElemExpr.argElem<single> 3
        let kse =
            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) ) +
            ElemExpr.ifThenElse trn_smpl1 trn_smpl2 (s [gp; trn_smpl1] *** 2.0f) (ElemExpr.scalar 0.0f)
        kse
    let ksenoSigma =
        let gp = ElemExpr.idx 0   
        let trn_smpl1 = ElemExpr.idx 1
        let trn_smpl2 = ElemExpr.idx 2
        let l = ElemExpr.argElem<single> 0
        let x = ElemExpr.argElem<single> 1
        let y = ElemExpr.argElem<single> 2
        let kse =
            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) )
        kse

    let CovMat lengthscales trnSigmas xVect yVect useSigma = 
        let nGps = (Expr.shapeOf lengthscales).[0]
        let sizeX = (Expr.shapeOf xVect).[1]
        let sizeY = (Expr.shapeOf yVect).[1]
        if useSigma then
            Expr.elements [nGps; sizeX; sizeY] kse [lengthscales; trnSigmas; xVect; yVect]
        else
            Expr.elements [nGps; sizeX; sizeY] ksenoSigma [lengthscales; xVect; yVect]

    let GPRegression lengthscales trnSigmas trnX trnT inX =
        let nGps = (Expr.shapeOf lengthscales).[0]
        let sizeX = (Expr.shapeOf trnX).[1]
        let sizeY = (Expr.shapeOf inX).[1]
        let K = CovMat lengthscales trnSigmas trnX trnX true
        let Kstar = CovMat lengthscales trnSigmas trnX inX false
        let Kstarstar = CovMat lengthscales trnSigmas inX inX false
        let K_inv = K |> Expr.invert
        let KstarT = Kstar |> Expr.transpose
        let mean = KstarT .* K_inv .* trnT
        let cov = Kstarstar - KstarT .* K_inv .* Kstar
        mean,cov
    
    ///Sample num points from several gaussian processes (e.g. one GPTransfer Layer)
    /// With input values between minValue and maxValue
    let sampleFromTrainedGP (lengthscales,trnSigmas,trnX,trnT) (minValue,maxValue) num=
        let nGPs = SizeSpec.symbol "nGPs"
        let nTrnSmpls = SizeSpec.symbol "nTrnSmpls"
        let nInput = SizeSpec.symbol "nInput"
        let ls = Expr.var<single> "ls" [nGPs]
        let sigs = Expr.var<single> "sigs" [nGPs;nTrnSmpls]
        let x = Expr.var<single> "x" [nGPs;nTrnSmpls]
        let t = Expr.var<single> "t" [nGPs;nTrnSmpls]
        let inp = Expr.var<single> "inp" [nGPs;nInput]

        let mean, cov = GPRegression ls sigs x t inp
        let cmplr = DevCuda.Compiler, CompileEnv.empty
        let mean_cov_fn = Func.make2 cmplr mean cov |> arg5 ls sigs x t inp

        let xDim = [minValue..((maxValue-minValue)/num)..maxValue] |> ArrayNDHost.ofList |> ArrayNDCuda.toDev
        mean_cov_fn lengthscales trnSigmas trnX trnT xDim
        
    ()


