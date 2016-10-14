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
        let muSite = mu 
        let covSite = mu 
        
        /// one update step of the EP algorithm
        let updateStep (sigma: ExprT, mu: ExprT,covSite: ExprT,muSite: ExprT)= 
            let cov = Expr.diag sigma
            let covMinus = 1.0f / (1.0f / cov - 1.0f / covSite)
            let muMinus = covMinus*(1.0f/cov * mu - 1.0f/covSite * muSite)
            let z = muMinus / (vu * sqrt(1.0f + covMinus / (vu ** 2.0f)))
            let normPdfZ = standardNormalPDF z
            let normCdfZ  = standardNormalCDF z
            let covHatf1 = covMinus *** 2.0f * normPdfZ / (normCdfZ * (vu ** 2.0f + covMinus))
            let covHatf2 = z + normPdfZ/  normCdfZ
            let covHat = covMinus - covHatf1 * covHatf2
            let muHat = muMinus - (covMinus*normPdfZ) / (normCdfZ * vu * sqrt(1.0f + covMinus / (vu ** 2.0f)))
            let covSUpd = 1.0f / (1.0f / covHat - 1.0f / cov)
            let muSUpd = (covSUpd * (1.0f / covHat*muHat - 1.0f / cov) * mu)
            let covSite = covSUpd 
            let muSite =  muSUpd
            let sigma = (Expr.invert sigma) + (Expr.diagMat (1.0f/covSite)) |> Expr.invert
            let mu = sigma.*(Expr.diagMat (1.0f/covSite)).*muSite
            sigma,mu,covSite,muSite

        ///Update loop of the EP algorithm runs n iterations
        let optimize  n (sigma: ExprT, mu: ExprT,covSite: ExprT,muSite: ExprT) = 
            let newSigma, newMu,newCovSite,newMuSite = updateStep (sigma,mu,covSite,muSite)
            let prevSigma = Expr.var<single> "prevSigma" sigma.Shape
            let prevMu = Expr.var<single> "prevMu" mu.Shape
            let prevCovSite = Expr.var<single> "prevCovSite"covSite.Shape
            let prevMuSite = Expr.var<single> "prevMuSite" muSite.Shape
            let nIters = SizeSpec.fix n
            let delayCovSite = SizeSpec.fix 1
            let delayMuSite = SizeSpec.fix 1
            let delaySigma = SizeSpec.fix 1
            let delayMu = SizeSpec.fix 1
            let chCovSite = "covSite"
            let chMuSite = "muSite"
            let chSigma = "sigma"
            let chMu = "mu"
            let sigVar =  VarSpec.create "GP.GaussianProcess.SignalVariance" typeof<single> []
            let ls =  VarSpec.create "GP.GaussianProcess.Lengthscale" typeof<single>[]
            let x = VarSpec.create "GP.x" typeof<single>[SizeSpec.symbol "nTrnSmpls"]
            let sigNs = VarSpec.create "GP.sigNs" typeof<single>[SizeSpec.symbol "nTrnSmpls"]

            let loopSpec = {
                Expr.Length = nIters
                Expr.Vars = Map [Expr.extractVar prevCovSite,  Expr.PreviousChannel {Channel=chCovSite; Delay=delayCovSite; InitialArg=0}
                                 Expr.extractVar prevMuSite,  Expr.PreviousChannel {Channel=chMuSite; Delay=delayMuSite; InitialArg=1}
                                 Expr.extractVar prevSigma, Expr.PreviousChannel {Channel=chSigma; Delay=delaySigma; InitialArg=2}
                                 Expr.extractVar prevMu, Expr.PreviousChannel {Channel=chMu; Delay=delayMu; InitialArg=3}
                                 sigVar, Expr.ConstArg 4
                                 ls, Expr.ConstArg 5
                                 x, Expr.ConstArg 6
                                 sigNs, Expr.ConstArg 7]
                Expr.Channels = Map [chCovSite, {LoopValueT.Expr=newCovSite; LoopValueT.SliceDim=0}
                                     chMuSite, {LoopValueT.Expr=newMuSite; LoopValueT.SliceDim=0}
                                     chSigma, {LoopValueT.Expr=newSigma; LoopValueT.SliceDim=0}
                                     chMu, {LoopValueT.Expr=newMu; LoopValueT.SliceDim=0}]    
            }
            let covSite = Expr.reshape [SizeSpec.fix 1;covSite.Shape.[0]] covSite
            let muSite = Expr.reshape [SizeSpec.fix 1;muSite.Shape.[0]] muSite
            let sigma = Expr.reshape [SizeSpec.fix 1;sigma.Shape.[0];sigma.Shape.[1]] sigma
            let mu = Expr.reshape [SizeSpec.fix 1;mu.Shape.[0]] mu
            let newCovSite = Expr.loop loopSpec chCovSite [covSite;muSite;sigma;mu;Expr.makeVar sigVar;Expr.makeVar ls;Expr.makeVar x;Expr.makeVar sigNs]
            let newMuSite = Expr.loop loopSpec chMuSite [covSite;muSite;sigma;mu;Expr.makeVar sigVar;Expr.makeVar ls;Expr.makeVar x;Expr.makeVar sigNs]
            let newSigma = Expr.loop loopSpec chSigma [covSite;muSite;sigma;mu;Expr.makeVar sigVar;Expr.makeVar ls;Expr.makeVar x;Expr.makeVar sigNs]
            let newMu = Expr.loop loopSpec chMu [covSite;muSite;sigma;mu;Expr.makeVar sigVar;Expr.makeVar ls;Expr.makeVar x;Expr.makeVar sigNs]
            newSigma.[nIters - 1,0..newSigma.Shape.[1] - 1,0..newSigma.Shape.[2] - 1],newMu.[nIters - 1,0..newMu.Shape.[1] - 1],newCovSite.[nIters - 1,0..newCovSite.Shape.[1] - 1],newMuSite.[nIters - 1,0..newMuSite.Shape.[1] - 1]
        optimize 10 (sigma,mu, covSite,muSite)


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
        let k           = (covMat x x) + Expr.diagMat sigmaNs
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


                //TODO: covSite and muSite without EP algorithm
                let _,_,covSite,muSite = ExpectationPropagation.monotonicityEP k vu
                

                
                let xJoint = Expr.concat 0 [x;xm]
                printfn "Shape kFf = %A\nShape kFf' = %A"kFf.Shape kFf'.Shape
                printfn "Shape kF'f = %A\nShape kFf' = %A"kF'f.Shape kF'f'.Shape
                let kJoint1,kJoint2 = Expr.concat 0 [kFf;kFf'],Expr.concat 0 [kF'f;kF'f']
                printfn "Shape kJoint1 = %A\nShape  kJoint2 = %A" kJoint1.Shape  kJoint2.Shape
                let kJoint = Expr.concat 1 [kJoint1;kJoint2]
                printfn "Shape kJoint = %A" kJoint.Shape
                let muJoint = Expr.concat 0 [y;muSite]
                let zeroMat = Expr.zerosLike kFf
                let zeroMatFf' = Expr.zerosLike kFf'
                let zeroMatF'f = Expr.zerosLike kF'f
                let sigmaJ1,sigmaJ2=Expr.concat 0 [zeroMat;zeroMatFf'],Expr.concat 0 [zeroMatF'f.T;(Expr.diagMat covSite)]
                printfn "Shape sigmaJ1 = %A\nShape sigmaJ2 = %A"sigmaJ1.Shape sigmaJ2.Shape
                let sigmaJoint =  Expr.concat 1 [sigmaJ1;sigmaJ2] 
                let kInv = Expr.invert (kJoint + sigmaJoint)
                let kStar = covMat xJoint xStar
                let meanXJoint = meanFct xJoint
                let mean = meanXStar + kStar.T .* kInv .* (muJoint - meanXJoint)
                let cov = kStarstar - kStar.T .* kInv .* kStar
            
                mean,cov
            | None ->
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
