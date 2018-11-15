namespace Models

open Tensor
open SymTensor

module GaussianProcess =
    
    /// Kernel Type.
    type Kernel =
        /// linear kernel
        | Linear
        /// squared exponential kernel
        | SquaredExponential of single*single



    /// Gaussian Process hyperparameter type.
    [<CustomEquality; NoComparison>]
    type HyperPars = {
        Kernel:             Kernel
        MeanFunction:       Expr -> Expr
        CutOutsideRange:    bool
    } with
        override x.Equals(yobj) =
            match yobj with
            | :? HyperPars as y -> 
                let v = Expr.var<single> "v" [SizeSpec.symbol "n"]
                x.Kernel = y.Kernel &&
                x.MeanFunction v = y.MeanFunction v &&
                x.CutOutsideRange = y.CutOutsideRange
            | _ -> false
        override x.GetHashCode() =
            hash (x.Kernel, x.CutOutsideRange)
    

    ///The dafault hyperparameters.
    let defaultHyperPars ={
        Kernel = SquaredExponential (1.0f,1.0f)
        MeanFunction = (fun x -> Expr.zerosLike x)
        CutOutsideRange = false
        }


    /// Gaussian Process parameters with linear kernel.
    type ParsLinear = {
        HyperPars:          HyperPars
        }
    

    /// Gaussian Process parameters with squared exponential kernel.
    type ParsSE = {
        Lengthscale:    Expr
        SignalVariance: Expr
        HyperPars:  HyperPars
        }


    ///Iitializes the lengthscale.
    let  initLengthscale l seed (shp: int64 list)  : Tensor<single> =
        HostTensor.scalar l
    

    /// Initializes the signal variance.
    let  initSignalVariance s seed (shp: int64 list) : Tensor<single> =
        HostTensor.scalar s


    /// Parameter Type of a Gaussian Process dependent on the used Kernel.
    type Pars = LinPars of ParsLinear 
                | SEPars of  ParsSE


    /// Parameters of the Gaussian Process.
    let pars (mb: ModelBuilder<_>) (hp:HyperPars) = 
        match hp.Kernel with
        | Linear -> LinPars {HyperPars = hp}
        | SquaredExponential (l,s)-> SEPars { Lengthscale = mb.Param ("Lengthscale" , [], initLengthscale l)
                                              SignalVariance = mb.Param ("SignalVariance" , [], initSignalVariance s)
                                              HyperPars = hp}
    

    /// Calculates covariance matrix between two vectors using linear kernel.
    let linearCovariance (x:Expr) (y:Expr) =
        let x_smpl, y_smpl  = Elem.Expr.idx2
        let xvec, yvec = Elem.Expr.arg2<single>
        let klin = xvec[x_smpl] * yvec[y_smpl]
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] klin [x; y]


    /// Calculates covariance matrix between two vectors using linear kernel.
    let squaredExpCovariance (l:Expr, sigf:Expr) (x:Expr) (y:Expr) =
        let x_smpl, y_smpl  = Elem.Expr.idx2
        let xvec, yvec,len,sigmaf = Elem.Expr.arg4<single>
        let kse = sigmaf[] * exp  (-( (xvec[x_smpl] - yvec[y_smpl]) *** 2.0f) / (2.0f * len[] *** 2.0f) )
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y

        let kse = Expr.elements [sizeX;sizeY] kse [x; y;l;sigf]
        kse
    let predict (pars:Pars) x (y:Expr) sigmaNs xStar =
        let covMat z z' =
            match pars with
            | LinPars _ -> linearCovariance z z'
            | SEPars parsSE  -> squaredExpCovariance (parsSE.Lengthscale,parsSE.SignalVariance) z z'
        let k           = (covMat x x)
        let kStarStar  = covMat xStar xStar
        
        let meanFct,cut = 
            match pars with
            | LinPars parsLin -> parsLin.HyperPars.MeanFunction, parsLin.HyperPars.CutOutsideRange
            | SEPars parsSE -> parsSE.HyperPars.MeanFunction, parsSE.HyperPars.CutOutsideRange
        
        let meanX = meanFct x
        let meanXStar = meanFct xStar

        let k = k  + Expr.diagMat sigmaNs
        let kInv        = Expr.invert k

        let kStar      = covMat x xStar

        let mean = 
            meanXStar + kStar.T .* kInv .* (y - meanX)
        let cov = kStarStar - kStar.T .* kInv .* kStar
        let mean = 
            if cut then
                let nTrnSmpls =x.NElems
                let nSmpls = xStar.NElems
                let xFirst = x.[0] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
                let xLast = x.[nTrnSmpls - 1L] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
                let yFirst = y.[0] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
                let yLast = y.[nTrnSmpls - 1L] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
//                let xFirst = x.[0] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
//                let xLast = x.[nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
//                let yFirst = y.[0] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
//                let yLast = y.[nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable]|> Expr.broadcast [nSmpls]
                let mean = Expr.ifThenElse (xStar <<<< xFirst) yFirst mean
                Expr.ifThenElse (xStar >>>> xLast) yLast mean
            else
                mean
        mean, cov


    /// WARNING: NOT YET IMPLEMENTED, ONLY A REMINDER FOR LATER IMPLEMENTATION!
    /// !!! CALLING THIS FUNCTION WILL ONLY CAUSE AN ERROR !!!
    let logMarginalLiklihood (pars:Pars) x y sigmaNs xStar =
        failwith "TODO: implement logMarginalLikelihood"
