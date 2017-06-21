namespace GaussianProcess

open RProvider
open RProvider.graphics

open Tensor
open RTools



/// Gaussian Process
type GP () =
    
    static let getNSamples (x: Tensor<float>) = 
        match Tensor.shape x with
        | [s] -> s
        | _ -> failwithf "training/test samples must be a vector but got %A" x.Shape

    static let meanVec meanFn x : Tensor<float>  = 
        HostTensor.init [getNSamples x] (fun pos -> meanFn x.[[pos.[0]]])

    static let covMat covFn xa xb : Tensor<float> = 
        HostTensor.init [getNSamples xa; getNSamples xb] 
            (fun pos -> covFn xa.[[pos.[0]]] xb.[[pos.[1]]])

    /// zero mean function
    static member meanZero (x: float) = 
        0.0

    /// squared-exponential covariance function
    static member covSe var lengthscale xa xb =
        var * exp (- ((xa-xb)**2.) / (2. * lengthscale**2.))

    /// 1st derivative of squared-exponential covariance function
    static member covDSe var lengthscale dxa xb =
        -var * exp (- ((dxa-xb)**2.) / (2. * lengthscale**2.)) * (dxa - xb) / (lengthscale**2.)

    /// 2nd derivative of squared-exponential covariance function
    static member covDDSe var lengthscale dxa dxb =
        var * exp (- ((dxa-dxb)**2.) / (2. * lengthscale**2.)) * 
            (1. / lengthscale**2. - (dxa - dxb)**2. / lengthscale**4.)

    /// squared-exponential covariance function and its derivatives
    static member covSeWithDerivs var lengthscale =
        GP.covSe var lengthscale, GP.covDSe var lengthscale, GP.covDDSe var lengthscale

    /// Returns the mean and covariance of a GP prior.
    static member prior (x, meanFn, covFn) =
        let mu = meanVec meanFn x 
        let sigma = covMat covFn x x 
        mu, sigma

    /// Returns the mean and covariance of a GP regression.
    static member regression (meanFn, covFn, tstX, trnX, trnY, trnV) =                              
        let trnMean = meanVec meanFn trnX 
        let tstMean = meanVec meanFn tstX 
        let trnTrnCov = covMat covFn trnX trnX + Tensor.diagMat trnV
        let tstTstCov = covMat covFn tstX tstX 
        let tstTrnCov = covMat covFn tstX trnX 
       
        let Kinv = Tensor.pseudoInvert trnTrnCov
        let tstMu = tstMean + tstTrnCov .* Kinv .* (trnY - trnMean)
        let tstSigma = tstTstCov - tstTrnCov .* Kinv .* tstTrnCov.T
        tstMu, tstSigma 

    /// Returns the mean and covariance and the mean and covariance of the derivative of a
    /// a GP regression with derivative targets.
    static member regressionWithDeriv ((covFn, covDFn, covDDFn), 
                                       tstX, 
                                       trnX, trnY: Tensor<float>, trnV,
                                       ?trnDX, ?trnDY, ?trnDV) =                              

        let trnDX, trnDY, trnDV =
            match trnDX, trnDY, trnDV with
            | Some trnDX, Some trnDY, Some trnDV -> trnDX, trnDY, trnDV
            | None, None, None -> 
                let empty = Tensor.empty trnY.Dev 1
                empty, empty, empty
            | _ -> failwith "trnDX, trnDY, trnDV must be specified together"

        let trnT = Tensor.ofBlocks [trnY; trnDY]

        let trnTrnCov = covMat covFn trnX trnX + Tensor.diagMat trnV
        let dTrnTrnCov = covMat covDFn trnDX trnX
        let trnDTrnCov = dTrnTrnCov.T
        let dTrnDTrnCov = covMat covDDFn trnDX trnDX + Tensor.diagMat trnDV
        let K = Tensor.ofBlocks [[trnTrnCov;  trnDTrnCov ]
                                 [dTrnTrnCov; dTrnDTrnCov]]
        let Kinv = Tensor.pseudoInvert K

        let tstTrnCov = covMat covFn tstX trnX 
        let dTstTrnCov = covMat covDFn tstX trnX
        let tstDTrnCov = covMat covDFn trnDX tstX |> Tensor.transpose
        let dTstDTrnCov = covMat covDDFn tstX trnDX
        let Kstar = Tensor.ofBlocks [[tstTrnCov; tstDTrnCov]]
        let KDstar = Tensor.ofBlocks [[dTstTrnCov; dTstDTrnCov]]

        let tstTstCov = covMat covFn tstX tstX 
        let dTstDTstCov = covMat covDDFn tstX tstX

        let tstMu = Kstar .* Kinv .* trnT
        let tstDMu = KDstar .* Kinv .* trnT
        let tstSigma = tstTstCov - Kstar .* Kinv .* Kstar.T
        let tstDSigma = dTstDTstCov - KDstar .* Kinv .* KDstar.T
        (tstMu, tstSigma), (tstDMu, tstDSigma)

    /// Plots the mean and variance of a GP.
    /// Optionally the training points for a GP regression can be specified.
    static member plot (tstX, tstMu: Tensor<float>, tstSigma, 
                        ?trnX, ?trnY, ?trnV) =
        R.lock (fun () ->
            let ary = HostTensor.toArray
            let tstStd = tstSigma |> Tensor.diag |> sqrt
            let tstYL = tstMu - tstStd
            let tstYH = tstMu + tstStd
            R.plot3(xRng=(Tensor.min tstX, Tensor.max tstX),
                    yRng=(Tensor.min tstYL, Tensor.max tstYH))
            R.fillBetween (ary tstX, ary tstYL, ary tstYH, color="lightgrey")
            R.lines2 (ary tstX, ary tstMu, color="red")
            match trnX, trnY, trnV with
            | Some trnX, Some trnY, Some trnV ->
                R.points2 (ary trnX, ary trnY, color="black", bg="black", pch=19)
            | _ -> ()
        )


    



    

    



                


