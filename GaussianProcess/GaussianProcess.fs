namespace GaussianProcess

open RProvider
open RProvider.graphics

open Tensor
open RTools


/// Parameters for the squared-exponential covariance function.
type CovSeParams = {
    /// Variance
    Variance:       float
    /// Lengthscale
    Lengthscale:    float
}

/// Mean functions.
module MeanFns =
    /// zero mean function
    let zero (x: float) = 
        0.0

/// Covariance functions.
module CovFns = 
    /// squared-exponential covariance function
    let Se {Variance=v; Lengthscale=l} xa xb =
        v * exp (- ((xa-xb)**2.) / (2. * l**2.))

    /// 1st derivative of squared-exponential covariance function
    let DSe {Variance=v; Lengthscale=l} dxa xb =
        -v * exp (- ((dxa-xb)**2.) / (2. * l**2.)) * (dxa - xb) / (l**2.)

    /// 2nd derivative of squared-exponential covariance function
    let DDSe {Variance=v; Lengthscale=l} dxa dxb =
        v * exp (- ((dxa-dxb)**2.) / (2. * l**2.)) * 
            (1. / l**2. - (dxa - dxb)**2. / l**4.)

    /// squared-exponential covariance function and its derivatives
    let SeWithDerivs pars =
        Se pars, DSe pars, DDSe pars


/// Gaussian Process
type GP () =
    
    static member getNSamples (x: Tensor<float>) = 
        match Tensor.shape x with
        | [s] -> s
        | _ -> failwithf "training/test samples must be a vector but got %A" x.Shape

    static member meanVec meanFn x : Tensor<float>  = 
        HostTensor.init [GP.getNSamples x] (fun pos -> meanFn x.[[pos.[0]]])

    static member covMat covFn xa xb : Tensor<float> = 
        HostTensor.init [GP.getNSamples xa; GP.getNSamples xb] 
            (fun pos -> covFn xa.[[pos.[0]]] xb.[[pos.[1]]])

    /// Returns the mean and covariance of a GP prior.
    static member prior (x, meanFn, covFn) =
        let mu = GP.meanVec meanFn x 
        let sigma = GP.covMat covFn x x 
        mu, sigma

    /// Returns the mean and covariance of a GP regression.
    static member regression (meanFn, covFn, tstX, trnX, trnY, trnV) =                              
        let trnMean = GP.meanVec meanFn trnX 
        let tstMean = GP.meanVec meanFn tstX 
        let trnTrnCov = GP.covMat covFn trnX trnX + Tensor.diagMat trnV
        let tstTstCov = GP.covMat covFn tstX tstX 
        let tstTrnCov = GP.covMat covFn tstX trnX 
       
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

        let trnTrnCov = GP.covMat covFn trnX trnX + Tensor.diagMat trnV
        let dTrnTrnCov = GP.covMat covDFn trnDX trnX
        let trnDTrnCov = dTrnTrnCov.T
        let dTrnDTrnCov = GP.covMat covDDFn trnDX trnDX + Tensor.diagMat trnDV
        let K = Tensor.ofBlocks [[trnTrnCov;  trnDTrnCov ]
                                 [dTrnTrnCov; dTrnDTrnCov]]
        let Kinv = Tensor.pseudoInvert K

        let tstTrnCov = GP.covMat covFn tstX trnX 
        let dTstTrnCov = GP.covMat covDFn tstX trnX
        let tstDTrnCov = GP.covMat covDFn trnDX tstX |> Tensor.transpose
        let dTstDTrnCov = GP.covMat covDDFn tstX trnDX
        let Kstar = Tensor.ofBlocks [[tstTrnCov; tstDTrnCov]]
        let KDstar = Tensor.ofBlocks [[dTstTrnCov; dTstDTrnCov]]

        let tstTstCov = GP.covMat covFn tstX tstX 
        let dTstDTstCov = GP.covMat covDDFn tstX tstX

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


    



    

    



                


