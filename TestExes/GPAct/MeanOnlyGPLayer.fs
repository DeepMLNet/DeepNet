namespace GPAct


open SymTensor
open Models
open GPUtils

module MeanOnlyGPLayer =


    /// Hyper-parameters of meanOnlyGPLayer
    type HyperPars = {
        /// number of Inputs
        NInput:                SizeSpecT
        
        /// number od Outputs
        NOutput:                SizeSpecT
        
        /// number of GP units <= number of outputs
        NGPs:                   SizeSpecT

        /// number of training points for each GP
        NTrnSmpls:              SizeSpecT

        /// if true mean stays at firt / last train value
        /// if input is outside the range of training values)
        CutOutsideRange:        bool
        /// mean function of the GPs
        MeanFunction:       (ExprT -> ExprT)

        /// optimize lengthscales during training
        LengthscalesTrainable:  bool
        /// optimize trnXvalues during training
        TrnXTrainable:          bool
        /// optimize tnrTvalues during training
        TrnTTrainable:          bool
        /// optimize TrnSigmas during training
        TrnSigmaTrainable:      bool
        /// optimize weights and bias during training
        WeightsTrainable:       bool

        /// lengthscale initialization method
        LengthscalesInit:       InitMethod
        /// tnrX initialization method
        TrnXInit:               InitMethod
        /// tnrT initialization method
        TrnTInit:               InitMethod
        /// tnrSigma initialization method
        TrnSigmaInit:           InitMethod
        /// weight initialization method
        WeightsInit:        InitMethod
        /// bias initialization method
        BiasInit:           InitMethod
        
        /// l1 regularization weight
        L1Regularization:   single option
        /// l2 regularization weight
        L2Regularization:   single option
    }

    /// The default hyper-parameters.
    let defaultHyperPars = {
        NInput                = SizeSpec.fix 0
        NOutput               = SizeSpec.fix 0
        NGPs                  = SizeSpec.fix 0
        NTrnSmpls             = SizeSpec.fix 10
        CutOutsideRange       = false
        MeanFunction          = (fun x -> Expr.zerosLike x)
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
        L1Regularization    = None
        L2Regularization    = None
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
    

    /// Creates parameters.
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

    /// Calculates covariance matri using squared exponential kernel.
    let covMat nGps nXSmpls nYSmpls lengthscales x y=
         let gp, xSmpl, ySmpl = ElemExpr.idx3   
         let lVec, xVec,yVec  = ElemExpr.arg3<single>
         let kse = (exp -((xVec[gp;xSmpl] - yVec[gp;ySmpl])***2.0f)/ (2.0f * lVec[gp]***2.0f))
         Expr.elements [nGps;nXSmpls;nYSmpls] kse [lengthscales;x;y]
    
    /// Predicting mean from input.
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
        let k = (covMat nOutput nTrnSmpls nTrnSmpls lengthscales trnX trnX)  + Expr.diagMat trnSigma
        let kInv = Expr.invert k
        let kStarT = covMat nOutput nSmpls nTrnSmpls lengthscales input.T trnX
        let meanTrnX = pars.HyperPars.MeanFunction trnX
        let meanInput = pars.HyperPars.MeanFunction input
        let mean = meanInput + (kStarT .* kInv .* (trnT - meanTrnX)).T
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

    /// Calculates sum of all regularization terms of this layer.
    let regularizationTerm pars  =
        let weights = pars.Weights
        if pars.HyperPars.WeightsTrainable then
            let l1reg =
                match pars.HyperPars.L1Regularization with
                | Some f    -> f * Regularization.l1Regularization weights
                | None      -> Expr.zeroOfSameType weights
            let l2reg =
                match pars.HyperPars.L2Regularization with
                | Some f    -> f * Regularization.l1Regularization weights
                | None      -> Expr.zeroOfSameType weights
            l1reg + l2reg
        else 
            Expr.zeroOfSameType weights

