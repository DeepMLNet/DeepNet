namespace Models

open ArrayNDNS
open SymTensor
open System

module ExpectationPropagation =
    
    /// PDF of standard normal distribution
    let standardNormalPDF (x:ExprT) =
        let fact = 1.0f/sqrt(2.0f*(single System.Math.PI))
        fact * exp(-(x***2.0f)/2.0f)
    
    /// Computes approximate gaussian error 
    /// with maximum approximation error of 1.2 * 10 ** -7
    let gaussianError (x:ExprT) = 
        
        let t = 1.0f/(1.0f+0.5f*abs(x))
        let sum = -0.18628806f + 1.00002368f * t + 0.37409196f * t ***2.0f +
                   0.09678418f * t *** 3.0f - 0.18628806f * t ***4.0f + 0.27886807f * t *** 5.0f -
                   1.13520398f * t ***6.0f + 1.48851587f * t ***7.0f - 0.82215223f * t ***8.0f +
                   0.17087277f * t ***9.0f
        let tau = t * exp(-x***2.0f + sum)
        Expr.ifThenElse (x>>==0.0f) (1.0f - tau) (tau-1.0f)
    
    ///CDF of standard normal distribution
    let standardNormalCDF (x:ExprT) =
        (1.0f + gaussianError(x/sqrt(2.0f)))/2.0f
    
    /// Runs Expecattion Maximization algorithm for monotonicity
    let monotonicityEP (sigma:ExprT)  (vu:single) =
        let mu = sigma |> Expr.diag |> Expr.zerosLike
        let vSite = mu 
        let tauSite = mu 
        
        /// one update step of the EP algorithm
        let updateStep (sigma: ExprT, mu: ExprT,tauSite: ExprT,vSite: ExprT)= 
            let cov = Expr.diag sigma
            let tauMinus = (1.0f / cov) - tauSite
            let vMinus = (1.0f/cov) * mu - vSite
            let covMinus = (1.0f/tauMinus)
            let muMinus = (Expr.diagMat covMinus) .* vMinus
            let z = muMinus / (vu * sqrt(1.0f + covMinus / (vu ** 2.0f)))
            let normPdfZ = standardNormalPDF z
            let normCdfZ  = standardNormalCDF z
            let covHatf1 = covMinus *** 2.0f * normPdfZ / (normCdfZ * (vu ** 2.0f + covMinus))
            let covHatf2 = z + normPdfZ/  normCdfZ
            let covHat = covMinus - covHatf1 * covHatf2
            let muHat = muMinus - (covMinus*normPdfZ) / (normCdfZ * vu * sqrt(1.0f + covMinus / (vu ** 2.0f)))
            let tauSUpd = (1.0f /covHat) - tauMinus
            let tauSUpd = tauSUpd |> Expr.checkFinite "tauSUpd"
            let vSUpd = (1.0f / covHat) * muHat - vMinus
            let vSUpd = vSUpd |> Expr.checkFinite "vSUpd"
            let tauSite = tauSUpd 
            let vSite =  vSUpd
            let sigma = sigma + (Expr.diagMat (1.0f/tauSite))
            let mu = sigma.*vSite
            sigma,mu,tauSite,vSite

        ///Update loop of the EP algorithm runs n iterations
        let optimize  n (sigma: ExprT, mu: ExprT,tauSite: ExprT,vSite: ExprT) = 
            let newSigma, newMu,newTauSite,newVSite = updateStep (sigma,mu,tauSite,vSite)
            let prevSigma = Expr.var<single> "prevSigma" sigma.Shape
            let prevMu = Expr.var<single> "prevMu" mu.Shape
            let prevTauSite = Expr.var<single> "prevCovSite"tauSite.Shape
            let prevVSite = Expr.var<single> "prevMuSite" vSite.Shape
            let nIters = SizeSpec.fix n
            let delayTauSite = SizeSpec.fix 1
            let delayVSite = SizeSpec.fix 1
            let delaySigma = SizeSpec.fix 1
            let delayMu = SizeSpec.fix 1
            let chTauSite = "covSite"
            let chVSite = "muSite"
            let chSigma = "sigma"
            let chMu = "mu"

            let loopSpec = {
                Expr.Length = nIters
                Expr.Vars = Map [Expr.extractVar prevTauSite,  Expr.PreviousChannel {Channel=chTauSite; Delay=delayTauSite; InitialArg=0}
                                 Expr.extractVar prevVSite,  Expr.PreviousChannel {Channel=chVSite; Delay=delayVSite; InitialArg=1}
                                 Expr.extractVar prevSigma, Expr.PreviousChannel {Channel=chSigma; Delay=delaySigma; InitialArg=2}
                                 Expr.extractVar prevMu, Expr.PreviousChannel {Channel=chMu; Delay=delayMu; InitialArg=3}]
                Expr.Channels = Map [chTauSite, {LoopValueT.Expr=newTauSite; LoopValueT.SliceDim=0}
                                     chVSite, {LoopValueT.Expr=newVSite; LoopValueT.SliceDim=0}
                                     chSigma, {LoopValueT.Expr=newSigma; LoopValueT.SliceDim=0}
                                     chMu, {LoopValueT.Expr=newMu; LoopValueT.SliceDim=0}]    
            }
            let tauSite = Expr.reshape [SizeSpec.fix 1;tauSite.Shape.[0]] tauSite
            let vSite = Expr.reshape [SizeSpec.fix 1;vSite.Shape.[0]] vSite
            let sigma = Expr.reshape [SizeSpec.fix 1;sigma.Shape.[0];sigma.Shape.[1]] sigma
            let mu = Expr.reshape [SizeSpec.fix 1;mu.Shape.[0]] mu
            let newTauSite = (Expr.loop loopSpec chTauSite [tauSite;vSite;sigma;mu]).[nIters - 1,*]
            let newVSite = (Expr.loop loopSpec chVSite [tauSite;vSite;sigma;mu]).[nIters - 1,*] |> Expr.checkFinite "muSUpd" 
            newTauSite,newVSite
        let tauSite,vSite = optimize 5 (sigma,mu, tauSite,vSite)
        let covSite = (1.0f/tauSite)
        let muSite = (Expr.diagMat covSite) .* vSite
        covSite, muSite

module GaussianProcess =
    
    /// Kernek
    type Kernel =
        /// linear kernel
        | Linear
        /// squared exponential kernel
        | SquaredExponential of single*single
    
    /// GP hyperparameters
    type HyperPars = {
        Kernel:             Kernel
        MeanFunction:       (ExprT -> ExprT)
        Monotonicity:       single option
        CutOutsideRange:    bool
        }
    
    ///the dafault hyperparameters
    let defaultHyperPars ={
        Kernel = SquaredExponential (1.0f,1.0f)
        MeanFunction = (fun x -> Expr.zerosLike x)
        Monotonicity = None
        CutOutsideRange = false
        }

    /// GP parameters with linear kernel
    type ParsLinear = {
        HyperPars:  HyperPars
        }
    
    /// GP parameters with squared exponential kernel
    type ParsSE = {
        Lengthscale:    ExprT
        SignalVariance: ExprT
        HyperPars:  HyperPars
        }

    let  initLengthscale l seed (shp: int list)  : ArrayNDHostT<single> =
        ArrayNDHost.scalar l
    
    let  initSignalVariance s seed (shp: int list) : ArrayNDHostT<single> =
        ArrayNDHost.scalar s

    type Pars = LinPars of ParsLinear | SEPars of  ParsSE



    let pars (mb: ModelBuilder<_>) (hp:HyperPars) = 
        match hp.Kernel with
        | Linear -> LinPars {HyperPars = hp}
        | SquaredExponential (l,s)-> SEPars { Lengthscale = mb.Param ("Lengthscale" , [], initLengthscale l)
                                              SignalVariance = mb.Param ("SignalVariance" , [], initSignalVariance s)
                                              HyperPars = hp}
    

    /// calculates Matrix between two vectors using linear kernel
    let linearCovariance (x:ExprT) (y:ExprT) =
        let x_smpl, y_smpl  = ElemExpr.idx2
        let xvec, yvec = ElemExpr.arg2<single>
        let klin = xvec[x_smpl] * yvec[y_smpl]
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] klin [x; y]

    /// calculates Matrix between two vectors using linear kernel
    let squaredExpCovariance (l:ExprT, sigf:ExprT) (x:ExprT) (y:ExprT) =
        let x_smpl, y_smpl  = ElemExpr.idx2
        let xvec, yvec,len,sigmaf = ElemExpr.arg4<single>
        let kse = sigmaf[] * (exp -((xvec[x_smpl] - yvec[y_smpl])***2.0f)/ (2.0f * len[]***2.0f))
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] kse [x; y;l;sigf]

    /// Prediction of mean and covariance of input data xstar given train inputs x and targets y
    let predict (pars:Pars) x (y:ExprT) sigmaNs xStar =
        let covMat z z' =
            match pars with
            | LinPars _ -> linearCovariance z z'
            | SEPars parsSE  -> squaredExpCovariance (parsSE.Lengthscale,parsSE.SignalVariance) z z'
        let k           = (covMat x x)
        let kStarstar  = covMat xStar xStar
        
        let meanFct,monotonicity,cut = 
            match pars with
            | LinPars parsLin -> parsLin.HyperPars.MeanFunction, parsLin.HyperPars.Monotonicity, parsLin.HyperPars.CutOutsideRange
            | SEPars parsSE -> parsSE.HyperPars.MeanFunction, parsSE.HyperPars.Monotonicity,  parsSE.HyperPars.CutOutsideRange
        
        let meanX = meanFct x
        let meanXStar = meanFct xStar

        let mean,cov = 
            match monotonicity with
            | Some vu ->
                
                let covPdFun (x:ExprT) (k:ExprT) =
                    let i, j  = ElemExpr.idx2
                    let cMat,xvect = ElemExpr.arg2<single>
                    let xelem = xvect[i]
                    let cPdFun = cMat[i;j] |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem xelem
                    let sizeK1 = k.Shape.[0]
                    let sizeK2 = k.Shape.[0]
                    Expr.elements [sizeK1;sizeK2] cPdFun [k;x]
                
                let covPdPd  (x:ExprT) (y:ExprT) (k:ExprT)  = 
                    let i, j  = ElemExpr.idx2
                    let cMat,xvect,yvect = ElemExpr.arg3<single>
                    let xelem = xvect[i]
                    let yelem = yvect[j]
                    let cPdPd = cMat[i;j] |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem xelem |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem yelem
                    let sizeK1 = k.Shape.[0]
                    let sizeK2 = k.Shape.[0]
                    Expr.elements [sizeK1;sizeK2] cPdPd [k;x;y]
                
                ///locations of the virtual derivative points on training points
                let xm = x
                let kFf = k
                let kF'f = covMat xm x |> covPdFun xm
                let kFf' = kF'f.T
                let kF'f' = covMat xm xm |> covPdPd xm xm

                let covSite,muSite = ExpectationPropagation.monotonicityEP k vu
                
                let covSite = covSite |> Expr.checkFinite "covSite"
                let muSite = muSite |> Expr.checkFinite "muSite"

                let xJoint = Expr.concat 0 [x;xm]
                let kJoint1,kJoint2 = Expr.concat 0 [kFf;kFf'],Expr.concat 0 [kF'f;kF'f']
                let kJoint = Expr.concat 1 [kJoint1;kJoint2]
                let muJoint = Expr.concat 0 [y;muSite]
                let sigmaNMat = Expr.diagMat sigmaNs
                let zeroMatFf' = Expr.zerosLike kFf'
                let zeroMatF'f = Expr.zerosLike kF'f
                let sigmaJ1,sigmaJ2=Expr.concat 0 [sigmaNMat;zeroMatFf'],Expr.concat 0 [zeroMatF'f.T;(Expr.diagMat covSite)]
                let sigmaJoint =  Expr.concat 1 [sigmaJ1;sigmaJ2] 
                let kInv = Expr.invert (kJoint + sigmaJoint)
                let kStar = covMat xJoint xStar
                let meanXJoint = meanFct xJoint
                let mean = meanXStar + kStar.T .* kInv .* (muJoint - meanXJoint)
                let cov = kStarstar - kStar.T .* kInv .* kStar
            
                mean,cov
            | None ->
                let k = k  + Expr.diagMat sigmaNs
                let kInv        = Expr.invert k
                let kStar      = covMat x xStar
                let mean = meanXStar + kStar.T .* kInv .* (y - meanX)
                let cov = kStarstar - kStar.T .* kInv .* kStar
                mean,cov
        let mean = 
            if cut then
                let nTrnSmpls =x.NElems
                let nSmpls = xStar.NElems
                let xFirst = x.[0] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
                let yFirst = y.[0] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
                let xLast = x.[nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
                let yLast = y.[nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]

                let mean = Expr.ifThenElse (xStar <<<< xFirst) yFirst mean
                Expr.ifThenElse (xStar >>>> xLast) yLast mean
            else
                mean
        mean, cov
    /// WARNING: NOT YET IMPLEMENTED, ONLY A RIMINDER FOR LATER IMPLEMENTATION!
    /// !!! CALLING THIS FUNCTION WILL ONLY CAUSE AN ERROR !!!
    let logMarginalLiklihood (pars:Pars) x y sigmaNs xStar =
        failwith "TODO: implement logMarginalLikelihood"
