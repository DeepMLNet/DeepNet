namespace GPAct


open SymTensor
open Models
open GPUtils

module MeanOnlyGPLayer =
    /// Hyper-parameters
    /// Hyper-parameters
    type HyperPars = {
        /// number of Inputs
        NInput:                SizeSpecT
        
        /// number od Outputs
        NOutput:                SizeSpecT
        
        /// number of GPs <= number of outputs
        NGPs:                   SizeSpecT

        /// number of training points for each GP
        NTrnSmpls:              SizeSpecT

        ///GP parameters (for all Gps in the layer)
        CutOutsideRange:        bool
        MeanFunction:       (ExprT -> ExprT)
        Monotonicity:       (single*int*single*single) option

        LengthscalesTrainable:  bool
        TrnXTrainable:          bool
        TrnTTrainable:          bool
        TrnSigmaTrainable:      bool
        WeightsTrainable:       bool

        LengthscalesInit:       InitMethod
        TrnXInit:               InitMethod
        TrnTInit:               InitMethod
        TrnSigmaInit:           InitMethod
        WeightsInit:            InitMethod
        BiasInit:               InitMethod
    }

    /// default hyper-parameters
    let defaultHyperPars = {
        NInput                = SizeSpec.fix 0
        NOutput               = SizeSpec.fix 0
        NGPs                  = SizeSpec.fix 0
        NTrnSmpls             = SizeSpec.fix 10
        CutOutsideRange       = false
        MeanFunction          = (fun x -> Expr.zerosLike x)
        Monotonicity          = None
        LengthscalesTrainable = true
        TrnXTrainable         = true
        TrnTTrainable         = true
        TrnSigmaTrainable     = true
        WeightsTrainable      = true
        LengthscalesInit      = Const 0.4f
        TrnXInit              = Linspaced (-2.0f, 2.0f)
        TrnTInit              = Linspaced (-2.0f, 2.0f)
        TrnSigmaInit          = Const (sqrt 0.1f)
        WeightsInit           = FanOptimal
        BiasInit              = Const 0.0f
    }

    /// Parameter expressions.
    type Pars = {
        /// GP lengthscales: [gp]
        Lengthscales:       ExprT 
        /// x values of GP training samples:         [gp, trn_smpl]
        TrnX:               ExprT 
        /// target values of GP training samples:    [gp, trn_smpl]
        TrnT:               ExprT 
        /// standard deviation of GP target values:  [gp, trn_smpl]
        TrnSigma:           ExprT 
        /// weights [nOutput, nInput]
        Weights:        ExprT 
        /// bias [nOutput]
        Bias:           ExprT

        /// hyper-parameters
        HyperPars:          HyperPars
    }
    

    /// creates parameters
    let pars (mb: ModelBuilder<_>) hp = 
        {
        Lengthscales   = mb.Param ("Lengthscales", [hp.NGPs],               GPUtils.initVals hp.LengthscalesInit)
        TrnX           = mb.Param ("TrnX",         [hp.NGPs; hp.NTrnSmpls], GPUtils.initVals hp.TrnXInit)
        TrnT           = mb.Param ("TrnT",         [hp.NGPs; hp.NTrnSmpls], GPUtils.initVals hp.TrnTInit)
        TrnSigma       = mb.Param ("TrnSigma",     [hp.NGPs; hp.NTrnSmpls], GPUtils.initVals hp.TrnSigmaInit)
        Weights        = mb.Param ("Weights", [hp.NOutput; hp.NInput], GPUtils.initVals hp.WeightsInit)
        Bias           = mb.Param ("Bias",    [hp.NOutput],            GPUtils.initVals hp.BiasInit)    
        HyperPars      = hp
    }


    let covMat nGps nXSmpls nYSmpls lengthscales x y=
         let gp, xSmpl, ySmpl = ElemExpr.idx3   
         let lVec, xVec,yVec  = ElemExpr.arg3<single>
         let kse = (exp -((xVec[gp;xSmpl] - yVec[gp;ySmpl])***2.0f)/ (2.0f * lVec[gp]***2.0f))
         Expr.elements [nGps;nXSmpls;nYSmpls] kse [lengthscales;x;y]
    
    let pred pars input =
        
        let nSmpls    = (Expr.shapeOf input).[0]
        let nGps      = pars.HyperPars.NGPs
        let nTrnSmpls = pars.HyperPars.NTrnSmpls
        let nOutput = pars.HyperPars.NOutput

        let lengthscales = 
            pars.Lengthscales
            |> Expr.replicateTo 0 nOutput
            |> Hold.tryRelease
            |> gate pars.HyperPars.LengthscalesTrainable
            |> Expr.checkFinite "Lengthscales"
        let trnX = 
            pars.TrnX
            |> Expr.replicateTo 0 nOutput
            |> Hold.tryRelease
            |> gate pars.HyperPars.TrnXTrainable
            |> Expr.checkFinite "TrnX"

        // trnT [gp, trn_smpl]
        let trnT = 
            pars.TrnT
            |> Expr.replicateTo 0 nOutput
            |> Hold.tryRelease
            |> gate pars.HyperPars.TrnTTrainable
            |> Expr.checkFinite "TrnT"
        let trnSigma = 
            pars.TrnSigma
            |> Expr.replicateTo 0 nOutput
            |> Hold.tryRelease
            |> gate pars.HyperPars.TrnSigmaTrainable
            |> Expr.checkFinite "TrnSigma"

        let input = Expr.checkFinite "Input" input
        let input = input .* pars.Weights.T + pars.Bias
        let K = (covMat nOutput nTrnSmpls nTrnSmpls lengthscales trnX trnX)  + Expr.diagMat trnSigma
        let KInv = Expr.invert K
        let KStarT = covMat nOutput nSmpls nTrnSmpls lengthscales input.T trnX
        let meanTrnX = pars.HyperPars.MeanFunction trnX
        let meanInput = pars.HyperPars.MeanFunction input
        let mean = meanInput + (KStarT .* KInv .* (trnT - meanTrnX)).T
        let mean = 
            if pars.HyperPars.CutOutsideRange then
                let xFirst = trnX.[*,0] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]
                let tFirst = trnT.[*,0] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]
                let xLast = trnX.[*,nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]
                let tLast = trnT.[*,nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]

                let mean = Expr.ifThenElse (input <<<< xFirst) tFirst mean
                Expr.ifThenElse (input >>>> xLast) tLast mean
            else
                mean
        mean

    let regularizationTerm pars (q:int) =
        let weights = pars.Weights
        if pars.HyperPars.WeightsTrainable then
            Regularization.lqRegularization weights q
        else 
            Expr.zeroOfSameType weights 

