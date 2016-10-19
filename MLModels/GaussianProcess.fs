namespace Models

open ArrayNDNS
open SymTensor
open System

module ExpectationPropagation =
    
    /// PDF of standard normal distribution
    let standardNormalPDF (x:ExprT) (mu:ExprT) (cov:ExprT)=
        let fact = 1.0f / sqrt( 2.0f * (single System.Math.PI)*cov)
        fact * exp( - ((x - mu) *** 2.0f) / (2.0f * cov))
    
    /// Computes approximate gaussian error 
    /// with maximum approximation error of 1.2 * 10 ** -7
    let gaussianError (x:ExprT) = 
        
        let t = 1.0f/  (1.0f + 0.5f * abs(x))
        let sum = -1.26551233f + 1.00002368f * t + 0.37409196f * t *** 2.0f +
                   0.09678418f * t *** 3.0f - 0.18628806f * t *** 4.0f + 0.27886807f * t *** 5.0f -
                   1.13520398f * t *** 6.0f + 1.48851587f * t *** 7.0f - 0.82215223f * t *** 8.0f +
                   0.17087277f * t *** 9.0f
        let tau = t * exp(-x *** 2.0f + sum)
        Expr.ifThenElse (x>>==0.0f) (1.0f - tau) (tau - 1.0f)
    
    ///CDF of standard normal distribution
    let standardNormalCDF (x:ExprT) (mu:ExprT) (cov:ExprT) =
        (1.0f + gaussianError((x- mu) / sqrt(2.0f * cov))) / 2.0f
    
    let normalize (x:ExprT) =
        let mean = Expr.mean x
        let cov = (Expr.mean (x * x)) - (mean * mean)
        let stdev = sqrt cov
        let zeroCov = x - (Expr.reshape [SizeSpec.broadcastable] mean)
        let nonzeroCov = (x - (Expr.reshape [SizeSpec.broadcastable] mean)) / (Expr.reshape [SizeSpec.broadcastable] stdev)
        Expr.ifThenElse (cov ==== (Expr.zeroOfSameType cov)) zeroCov nonzeroCov

    /// Runs Expecattion Maximization algorithm for monotonicity
    let monotonicityEP (sigma:ExprT)  (vu:single) iters=
        let mu = sigma |> Expr.diag |> Expr.zerosLike
        let vSite = mu 
        let tauSite = mu 
        
        /// one update step of the EP algorithm
        let updateStep (sigma: ExprT, mu: ExprT,tauSite: ExprT,vSite: ExprT)= 
            let cov = Expr.diag sigma
            let tauMinus = (1.0f / cov) - tauSite |> Expr.checkFinite "tauMinus"
            let vMinus = (1.0f/cov) * mu - vSite |> Expr.checkFinite "vMinus"
            let covMinus = (1.0f/tauMinus) |> Expr.checkFinite "covMinus"
            let muMinus = covMinus * vMinus |> Expr.checkFinite "muMinus"
            let z = muMinus / (vu * sqrt(1.0f + covMinus / (vu ** 2.0f))) |> Expr.checkFinite "z"
            let normZ = standardNormalPDF z (Expr.zeroOfSameType z) (Expr.oneOfSameType z)
//            let normZ = normalize z
            let normCdfZ  = standardNormalCDF z (Expr.zeroOfSameType z) (Expr.oneOfSameType z)
            let covHatf1 = (covMinus *** 2.0f * normZ) / (normCdfZ * (vu ** 2.0f + covMinus)) |> Expr.checkFinite "covHatf1"
            let covHatf2 = z + normZ /  normCdfZ |> Expr.checkFinite "covHatf2"
            let covHat = covMinus - covHatf1 * covHatf2 |> Expr.checkFinite "covHat"
            let muHat = muMinus - (covMinus*normZ) / (normCdfZ * vu * sqrt(1.0f + covMinus / (vu ** 2.0f))) |> Expr.checkFinite "muHat"
            let tauSUpd = (1.0f /covHat) - tauMinus
            let tauSUpd = tauSUpd |> Expr.checkFinite "tauSUpd"
            let vSUpd = (1.0f / covHat) * muHat - vMinus
            let vSUpd = vSUpd |> Expr.checkFinite "vSUpd"
            let tauSite = tauSUpd 
            let vSite =  vSUpd
            let sigma = (Expr.invert sigma) + (Expr.diagMat tauSite) |> Expr.invert
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
        let tauSite,vSite = optimize iters (sigma,mu, tauSite,vSite)
        let covSite = (1.0f/tauSite)
        let muSite = covSite * vSite
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
        Monotonicity:       (single*int*single*single) option
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
        ObservationPoints:  ExprT
        HyperPars:          HyperPars
        }
    
    /// GP parameters with squared exponential kernel
    type ParsSE = {
        Lengthscale:    ExprT
        SignalVariance: ExprT
        ObservationPoints:  ExprT
        HyperPars:  HyperPars
        }

    let  initLengthscale l seed (shp: int list)  : ArrayNDHostT<single> =
        ArrayNDHost.scalar l
    
    let  initSignalVariance s seed (shp: int list) : ArrayNDHostT<single> =
        ArrayNDHost.scalar s

    let initObservationPoints minElem maxElem seed (shp: int list) : ArrayNDHostT<single> =
        ArrayNDHost.linSpaced minElem maxElem shp.[0]

    type Pars = LinPars of ParsLinear | SEPars of  ParsSE


    let pars (mb: ModelBuilder<_>) (hp:HyperPars) = 
        let n,min,max = match hp.Monotonicity with
                | Some (f,i,min,max) -> i,min,max
                | None ->  2,0.0f,1.0f
        match hp.Kernel with
        | Linear -> LinPars {ObservationPoints =  mb.Param ("ObservationPoints", [SizeSpec.fix n], initObservationPoints min max )
                             HyperPars = hp}
        | SquaredExponential (l,s)-> SEPars { Lengthscale = mb.Param ("Lengthscale" , [], initLengthscale l)
                                              SignalVariance = mb.Param ("SignalVariance" , [], initSignalVariance s)
                                              ObservationPoints =  mb.Param ("ObservationPoints", [SizeSpec.fix n], initObservationPoints min max )
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
        let kse = sigmaf[] * exp  (-( (xvec[x_smpl] - yvec[y_smpl]) *** 2.0f) / (2.0f * len[] *** 2.0f) )
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] kse [x; y;l;sigf]
    
    let covPdFunLin (x:ExprT) (y:ExprT) =
        let x_smpl, y_smpl  = ElemExpr.idx2
        let xvec, yvec = ElemExpr.arg2<single>
        let klinDeriv = yvec[y_smpl]
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] klinDeriv [x; y]

    let covPdPdLin (x:ExprT) (y:ExprT) =
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        (Expr.zerosOfSameType x [sizeX;sizeY]) + 1.0f

    let covPdFunSE (l:ExprT, sigf:ExprT) (x:ExprT) (y:ExprT) =
        let x_smpl, y_smpl  = ElemExpr.idx2
        let xvec, yvec,len,sigmaf = ElemExpr.arg4<single>
        let outerDeriv = sigmaf[] * exp( - (xvec[x_smpl] - yvec[y_smpl]) ***2.0f / (2.0f * len[]) *** 2.0f)
        let innerDeriv = (yvec[y_smpl] - xvec[x_smpl]) / (len[] *** 2.0f)
        let dksedx = outerDeriv* innerDeriv
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] dksedx [x; y;l;sigf]
    
//    let covPdFunSE (l:ExprT, sigf:ExprT) (x:ExprT) (y:ExprT) =
//        let x_smpl, y_smpl  = ElemExpr.idx2
//        let xvec, yvec,len,sigmaf = ElemExpr.arg4<single>
//        let kse = sigmaf[] * exp  (-( (xvec[x_smpl] - yvec[y_smpl]) *** 2.0f) / (2.0f * len[] *** 2.0f) )
//        let xElem = xvec[x_smpl]
//        let kseDeriv = ElemExprDeriv.compute kse  |> ElemExprDeriv.ofArgElem xElem 
//        let sizeX = Expr.nElems x
//        let sizeY = Expr.nElems y
//        Expr.elements [sizeX;sizeY] kseDeriv [x; y;l;sigf]

    let covPdPdSE (l:ExprT, sigf:ExprT) (x:ExprT) (y:ExprT) =
        let x_smpl, y_smpl  = ElemExpr.idx2
        let xvec, yvec,len,sigmaf = ElemExpr.arg4<single>
        let factor1 = sigmaf[] * exp( - (xvec[x_smpl] - yvec[y_smpl]) ***2.0f / (2.0f * len[]) *** 2.0f)
        let f1InnerDeriv =  (xvec[x_smpl] - yvec[y_smpl]) / (len[] *** 2.0f)
        let factor2 =  (yvec[y_smpl] - xvec[x_smpl]) / (len[] *** 2.0f)
        let factor2Deriv = 1.0f / (len[] *** 2.0f)
        let dksedx = factor1 * (f1InnerDeriv * factor2 + factor2Deriv)
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] dksedx [x; y;l;sigf]
//    let covPdPdSE (l:ExprT, sigf:ExprT) (x:ExprT) (y:ExprT) =
//        let x_smpl, y_smpl  = ElemExpr.idx2
//        let xvec, yvec,len,sigmaf = ElemExpr.arg4<single>
//        let kse = sigmaf[] * exp  (-( (xvec[x_smpl] - yvec[y_smpl]) *** 2.0f) / (2.0f * len[] *** 2.0f) )
//        let xElem = xvec[x_smpl]
//        let yElem = yvec[y_smpl]
//        let kseDeriv = ElemExprDeriv.compute kse  |> ElemExprDeriv.ofArgElem xElem |> ElemExprDeriv.compute |> ElemExprDeriv.ofArgElem xElem
//        let sizeX = Expr.nElems x
//        let sizeY = Expr.nElems y
//        Expr.elements [sizeX;sizeY] kseDeriv [x; y;l;sigf]
    /// Prediction of mean and covariance of input data xstar given train inputs x and targets y
    let predict (pars:Pars) x (y:ExprT) sigmaNs xStar =
        let covMat z z' =
            match pars with
            | LinPars _ -> linearCovariance z z'
            | SEPars parsSE  -> squaredExpCovariance (parsSE.Lengthscale,parsSE.SignalVariance) z z'
        let k           = (covMat x x)
        let kStarStar  = covMat xStar xStar
        
        let meanFct,monotonicity,cut,oPs = 
            match pars with
            | LinPars parsLin -> parsLin.HyperPars.MeanFunction, parsLin.HyperPars.Monotonicity, parsLin.HyperPars.CutOutsideRange, parsLin.ObservationPoints
            | SEPars parsSE -> parsSE.HyperPars.MeanFunction, parsSE.HyperPars.Monotonicity,  parsSE.HyperPars.CutOutsideRange, parsSE.ObservationPoints
        
        let meanX = meanFct x
        let meanXStar = meanFct xStar

        let mean,cov = 
            match monotonicity with
            | Some (vu,_,_,_) ->
//                
//                let covPdFun (x:ExprT) (k:ExprT) =
//                    let i, j  = ElemExpr.idx2
//                    let cMat,xvect = ElemExpr.arg2<single>
//                    let xelem = xvect[i]
//                    let cPdFun = cMat[i;j] |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem xelem
//                    let sizeK1 = k.Shape.[0]
//                    let sizeK2 = k.Shape.[1]
//                    Expr.elements [sizeK1;sizeK2] cPdFun [k;x]
                let covPdFun (x:ExprT) (y:ExprT) =
                    match pars with
                        | LinPars _ -> covPdFunLin x y
                        | SEPars parsSE  -> covPdFunSE (parsSE.Lengthscale,parsSE.SignalVariance) x y


//                let covPdPd  (x:ExprT) (y:ExprT) (k:ExprT)  = 
//                    let i, j  = ElemExpr.idx2
//                    let cMat,xvect,yvect = ElemExpr.arg3<single>
//                    let xelem = xvect[i]
//                    let yelem = yvect[j]
//                    let cPdPd = cMat[i;j] |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem xelem |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem yelem
//                    let sizeK1 = k.Shape.[0]
//                    let sizeK2 = k.Shape.[1]
//                    Expr.elements [sizeK1;sizeK2] cPdPd [k;x;y]
                let covPdPd (x:ExprT) (y:ExprT) =
                    match pars with
                        | LinPars _ -> covPdFunLin x y
                        | SEPars parsSE  -> covPdPdSE (parsSE.Lengthscale,parsSE.SignalVariance) x y
                ///locations of the virtual derivative points on training points
                let xm = oPs 
                let kFf = k
//                let kF'f = covMat xm x |> covPdFun xm
//                let kFf' = covMat x xm |> covFunPd xm
                let kF'f = covPdFun xm x
                let kFf' =  kF'f.T
                let kMm = covMat xm xm
//                let kF'f' = kMm  |> covPdPd xm xm
                let kF'f' = covPdPd xm xm
                let kF'f'= kF'f' |> Expr.checkFinite "KF'f'"
                let covSite,muSite = ExpectationPropagation.monotonicityEP kMm  vu 20
                
                let covSite = covSite |> Expr.checkFinite "covSite"
                let muSite = muSite |> Expr.checkFinite "muSite"
                let xJoint = Expr.concat 0 [x;xm]
                let kJoint1,kJoint2 = Expr.concat 1 [kFf;kFf'],Expr.concat 1 [kF'f;kF'f']
                let kJoint = Expr.concat 0 [kJoint1;kJoint2]
                let muJoint = Expr.concat 0 [y;muSite]
                let sigmaJoint =  Expr.concat 0 [sigmaNs; covSite] |> Expr.diagMat
                let kInv = Expr.invert (kJoint + sigmaJoint)
                let kStar = covMat xJoint xStar
                let meanXJoint = meanFct xJoint
                let mean =  kStar.T .* kInv .* (muJoint)
                let cov = kStarStar - kStar.T .* kInv .* kStar
            
                mean,cov
            | None ->
                let k = k  + Expr.diagMat sigmaNs
                let kInv        = Expr.invert k
                let kStar      = covMat x xStar
                let mean = meanXStar + kStar.T .* kInv .* (y - meanX)
                let cov = kStarStar - kStar.T .* kInv .* kStar
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
