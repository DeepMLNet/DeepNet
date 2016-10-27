namespace GPAct

open SymTensor
open Models
open GPUtils

/// Propagates normal distributions through non-linearities described by GPs.
module GPActivation =
    
    type OutputMode =
        MeanOnly | MeanVariance | MeanCovariance

    /// Hyper-parameters of the model.
    type HyperPars = {
        /// number of GP units <= number of outputs and inputs
        NGPs:                   SizeSpecT

        ///number of outputs
        NOutput:                SizeSpecT

        /// number of training points for each GP
        NTrnSmpls:              SizeSpecT

        OutputMode:             OutputMode

        /// if true mean stays at firt / last train value
        /// if input is outside the range of training values
        CutOutsideRange:        bool

        /// optimize lengthscales during training
        LengthscalesTrainable:  bool
        /// optimize trnXvalues during training
        TrnXTrainable:          bool
        /// optimize tnrTvalues during training
        TrnTTrainable:          bool
        /// optimize TrnSigmas during training
        TrnSigmaTrainable:      bool
        /// monotonous activation function
        Monotonicity:           single option
        /// lengthscale initialization method
        LengthscalesInit:       InitMethod
        /// tnrX initialization method
        TrnXInit:               InitMethod
        /// tnrT initialization method
        TrnTInit:               InitMethod
        /// tnrSigma initialization method
        TrnSigmaInit:           InitMethod


    }

    /// default hyper-parameters
    let defaultHyperPars = {
        NGPs                  = SizeSpec.fix 0
        NOutput               = SizeSpec.fix 0
        NTrnSmpls             = SizeSpec.fix 10
        OutputMode            = MeanVariance
        CutOutsideRange       = false
        LengthscalesTrainable = true
        TrnXTrainable         = true
        TrnTTrainable         = true
        TrnSigmaTrainable     = true
        Monotonicity          = None
        LengthscalesInit      = Const 0.4f
        TrnXInit              = Linspaced (-2.0f, 2.0f)
        TrnTInit              = Linspaced (-2.0f, 2.0f)
        TrnSigmaInit          = Const (sqrt 0.1f)
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
        /// hyper-parameters
        HyperPars:          HyperPars
    }
    
    /// Creates parameters.
    let pars (mb: ModelBuilder<_>) hp = {
        Lengthscales   = mb.Param ("Lengthscales", [hp.NGPs],               GPUtils.initVals hp.LengthscalesInit) 
        TrnX           = mb.Param ("TrnX",         [hp.NGPs; hp.NTrnSmpls], GPUtils.initVals hp.TrnXInit)
        TrnT           = mb.Param ("TrnT",         [hp.NGPs; hp.NTrnSmpls], GPUtils.initVals hp.TrnTInit)
        TrnSigma       = mb.Param ("TrnSigma",     [hp.NGPs; hp.NTrnSmpls], GPUtils.initVals hp.TrnSigmaInit)
        HyperPars      = hp
    }

    ///The covariance Matrices of the training vectors with themselves 
    ///by GP instances with squared exponential covariance.
    let Kk nGps nTrnSmpls lengthscales trnX trnSigma = 
        // Kse element expression
        // input  x[gp, trn_smpl]
        //        l[gp]
        //        s[gp, trn_smpl]
        // output cov[gp, trn_smpl1, trn_smpl2]
        let gp, trn_smpl1, trn_smpl2 = ElemExpr.idx3   
        let l, x, s = ElemExpr.arg3<single>
        let kse =
            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) ) +
            ElemExpr.ifThenElse trn_smpl1 trn_smpl2 (s [gp; trn_smpl1] *** 2.0f) (ElemExpr.scalar 0.0f)
        
        Expr.elements [nGps; nTrnSmpls; nTrnSmpls] kse [lengthscales; trnX; trnSigma]
    
    let dKkDx nGps nTrnSmpls lengthscales trnX trnSigma = 
        // Kse element expression
        // input  x[gp, trn_smpl]
        //        l[gp]
        //        s[gp, trn_smpl]
        // output cov[gp, trn_smpl1, trn_smpl2]
        let gp, trn_smpl1, trn_smpl2 = ElemExpr.idx3   
        let l, x, s = ElemExpr.arg3<single>
        let kse =
            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) ) +
            ElemExpr.ifThenElse trn_smpl1 trn_smpl2 (s [gp; trn_smpl1] *** 2.0f) (ElemExpr.scalar 0.0f)
        let dkseDx = kse |> ElemExprDeriv.compute |> ElemExprDeriv.ofArgElem (x [gp; trn_smpl1])
        Expr.elements [nGps; nTrnSmpls; nTrnSmpls] dkseDx [lengthscales; trnX; trnSigma] 
    
    ///The covariance of training vectors and input vector 
    ///by GP instances with squared exponential covariance.
    let lk nSmpls nGps nTrnSmpls mu sigma lengthscales trnX =
        // lk element expression
        // inputs  l[gp]
        //         x[gp, trn_smpl]
        //         m[smpl, gp]        -- mu
        //         s[smpl, gp1, gp2]  -- Sigma
        // output lk[smpl, gp, trn_smpl]
        let smpl = ElemExpr.idx 0
        let gp = ElemExpr.idx 1
        let trn_smpl = ElemExpr.idx 2
        let m = ElemExpr.argElem<single> 0
        let s = ElemExpr.argElem<single> 1
        let l = ElemExpr.argElem<single> 2
        let x = ElemExpr.argElem<single> 3

        let lk1 = sqrt ( (l [gp])***2.0f / ((l [gp])***2.0f + s [smpl; gp; gp]) )
        let lk2 = exp ( -( (m [smpl; gp] - x [gp; trn_smpl])***2.0f / (2.0f * ((l [gp])***2.0f + s [smpl; gp; gp])) ) )
        let lk = lk1 * lk2
        Expr.elements [nSmpls; nGps; nTrnSmpls] lk [mu; sigma; lengthscales; trnX]

    ///Elementwise matrix needed for calculation of the variance prediction.
    let L nSmpls nGps nTrnSmpls mu sigma lengthscales trnX =
        // L element expression
        // inputs  l[gp]
        //         x[gp, trn_smpl]
        //         m[smpl, gp]        -- mu
        //         s[smpl, gp1, gp2]  -- Sigma
        // output  L[smpl, gp, trn_smpl1, trn_smpl2]
        let smpl = ElemExpr.idx 0
        let gp = ElemExpr.idx 1
        let trn_smpl1 = ElemExpr.idx 2
        let trn_smpl2 = ElemExpr.idx 3
        let m = ElemExpr.argElem<single> 0
        let s = ElemExpr.argElem<single> 1
        let l = ElemExpr.argElem<single> 2
        let x = ElemExpr.argElem<single> 3

        let L1 = sqrt ( (l [gp])***2.0f / ((l [gp])***2.0f + 2.0f * s [smpl; gp; gp]) )
        let L2a = ( m [smpl; gp] - (x [gp; trn_smpl1] + x [gp; trn_smpl2])/2.0f )***2.0f / ((l [gp])***2.0f + 2.0f * s [smpl; gp; gp])
        let L2b = (x [gp; trn_smpl1] - x [gp; trn_smpl2])***2.0f / (4.0f * (l [gp])***2.0f)
        let L2 = exp (-L2a - L2b)
        let L = L1 * L2

        Expr.elements [nSmpls; nGps; nTrnSmpls; nTrnSmpls] L [mu; sigma; lengthscales; trnX]


    ///Elementwise matrix needed for calculation of the covariance prediction.
    let Tnew nSmpls nGps nTrnSmpls mu sigma lengthscales trnX =
        // T element expression
        // inputs  l[gp]
        //         x[gp, trn_smpl]
        //         m[smpl, gp]        -- mu
        //         s[smpl, gp1, gp2]  -- Sigma
        // output  T[smpl, gp1, gp2, trn_smpl1, trn_smpl2]

        let smpl = ElemExpr.idx 0
        let gp1 = ElemExpr.idx 1
        let gp2 = ElemExpr.idx 2
        let t1 = ElemExpr.idx 3
        let t2 = ElemExpr.idx 4
        let m = ElemExpr.argElem<single> 0
        let s = ElemExpr.argElem<single> 1
        let l = ElemExpr.argElem<single> 2
        let x = ElemExpr.argElem<single> 3

        // Mathematica: k = gp1  l = gp2   i=t1   j=t2

        let eNom = (x[gp2;t2]-m[smpl;gp2])***2.f * (l[gp1]***2.f+s[smpl;gp1;gp1]) + (x[gp1;t1]-m[smpl;gp1]) * 
                   ( 2.f * (m[smpl;gp2]-x[gp2;t2]) * s[smpl;gp1;gp2] + (x[gp1;t1]-m[smpl;gp1]) * (l[gp2]***2.f + s[smpl;gp2;gp2]) ) 
        let eDnm = 2.f * ( (l[gp1]***2.f + s[smpl;gp1;gp1]) * (l[gp2]***2.f + s[smpl;gp2;gp2]) - s[smpl;gp1;gp2]***2.f )
        let e = exp(-eNom / eDnm)
        let Tnom = e * l[gp1] * l[gp2]

        let Tdnm = sqrt ( (l[gp1]***2.f + s[smpl;gp1;gp1]) * (l[gp2]***2.f + s[smpl;gp2;gp2]) - s[smpl;gp1;gp2]***2.f )

        let T = ElemExpr.ifThenElse gp1 gp2 (ElemExpr.scalar 0.0f) (Tnom / Tdnm)
        Expr.elements [nSmpls; nGps; nGps; nTrnSmpls; nTrnSmpls] T [mu; sigma; lengthscales; trnX]


    /// Replace covariance matrix diagonal by specified variance.
    let setCovDiag nSmpls nGps cov var =
        // inputs  cov[smpl, gp1, gp2]
        //         var[smpl, gp
        // output  cov[smpl, gp1, gp2]
        let smpl = ElemExpr.idx 0
        let gp1 = ElemExpr.idx 1
        let gp2 = ElemExpr.idx 2
        let c = ElemExpr.argElem<single> 0
        let v = ElemExpr.argElem<single> 1

        let cv = ElemExpr.ifThenElse gp1 gp2 (v[smpl; gp1]) (c[smpl; gp1; gp2])
        Expr.elements [nSmpls; nGps; nGps] cv [cov; var]


    ///Predicted mean and covariance from input mean and covariance.
    let pred pars (mu, sigma) =
        // mu:    input mean        [smpl, gp]
        // Sigma: input covariance  [smpl, gp1, gp2]
        let nSmpls    = (Expr.shapeOf mu).[0]
        let nTrnSmpls = pars.HyperPars.NTrnSmpls
        
        let nGps      = pars.HyperPars.NGPs
        let nOutput   = pars.HyperPars.NOutput
        // check inputs
        let mu    = mu    |> Expr.checkFinite "mu"
        let sigma = sigma |> Expr.checkFinite "sigma"
        // check parameters and gate gradients

        let lengthscales = 
            pars.Lengthscales
            |> gate pars.HyperPars.LengthscalesTrainable
            |> Expr.checkFinite "Lengthscales"
            |> Expr.replicateTo 0 nOutput 
            |> Hold.tryRelease

        let trnX = 
            pars.TrnX
            |> gate pars.HyperPars.TrnXTrainable
            |> Expr.checkFinite "TrnX"
            |> Expr.replicateTo 0 nOutput
            |> Hold.tryRelease

        // trnT [gp, trn_smpl]
        let trnT = 
            pars.TrnT
            |> gate pars.HyperPars.TrnTTrainable
            |> Expr.checkFinite "TrnT"
            |> Expr.replicateTo 0 nOutput
            |> Hold.tryRelease

        let trnSigma = 
            pars.TrnSigma
            |> gate pars.HyperPars.TrnSigmaTrainable
            |> Expr.checkFinite "TrnSigma"
            |> Expr.replicateTo 0 nOutput
            |> Hold.tryRelease

        // Kk [gp, trn_smpl1, trn_smpl2]
        let Kk = Kk nOutput nTrnSmpls lengthscales trnX trnSigma
        let Kk = Kk |> Expr.checkFinite "Kk"
        //let Kk = Kk |> Expr.dump "Kk"
        
        let KkInv = Expr.invert Kk
        let KkInv = KkInv |> Expr.checkFinite "Kk_inv"
        //let Kk_inv = Kk_inv |> Expr.dump "Kk_inv"
        
        // lk [smpl, gp, trn_smpl]
        let lk = lk nSmpls nOutput nTrnSmpls mu sigma lengthscales trnX
        let lk = lk |> Expr.checkFinite "lk"
        //let lk = lk |> Expr.dump "lk"
        
        // ([gp, trn_smpl1, trn_smpl2] .* [gp, trn_smpl])       
        // ==> beta [gp, trn_smpl]
        let beta = KkInv .* trnT
        //let beta = beta |> Expr.dump "beta"

        // ==> sum ( [smpl, gp, trn_smpl] * beta[1*, gp, trn_smpl], trn_smpl)
        // ==> pred_mean [smpl, gp]
        let predMean = lk * Expr.padLeft beta |> Expr.sumAxis 2
        let predMean = predMean |> Expr.checkFinite "pred_mean"
        
        //let predMean = pred_mean |> Expr.dump "pred_mean"
        let predMean = 
            if pars.HyperPars.CutOutsideRange then
                let xFirst = trnX.[*,0] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]
                let tFirst = trnT.[*,0] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]
                let xLast = trnX.[*,nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]
                let tLast = trnT.[*,nTrnSmpls - 1] |> Expr.reshape [SizeSpec.broadcastable;nOutput]|> Expr.broadcast [nSmpls;nOutput]

                let predMean = Expr.ifThenElse (mu <<<< xFirst) tFirst predMean
                Expr.ifThenElse (mu >>>> xLast) tLast predMean
            else
                predMean

        let regTerm =
            match pars.HyperPars.Monotonicity with
            | Some v ->
                let dKkDx = dKkDx nOutput nTrnSmpls lengthscales trnX trnSigma |> Expr.checkFinite "dKkDx"
                let dpredMeanddX = dKkDx .*  beta 
                                   |> Expr.checkFinite "dpredMeanddX"
                // exp(x > 88.0f) -> infinity for CUDA implementation of sigmoid this somehow leads to nan
//                Expr.maxElemwise (dpredMeanddX / v) ((Expr.zerosLike dpredMeanddX) - 88.0f)
                (dpredMeanddX / v)
                |> ActivationFunc.alternativeSigmoid 
                |> Expr.mean
            | None  ->Expr.zeroOfSameType mu

        let regTerm = regTerm |> Expr.checkFinite "regTerm"     
        // L[smpl, gp, trn_smpl1, trn_smpl2]
        let L = L nSmpls nOutput nTrnSmpls mu sigma lengthscales trnX

        // betaBetaT = beta .* beta.T
        // [gp, trn_smpl, 1] .* [gp, 1, trn_smpl] ==> [gp, trn_smpl, trn_smpl]
        // is equivalent to: [gp, trn_smpl, 1*] * [gp, 1*, trn_smpl]
        let betaBetaT = 
            Expr.reshape [nOutput; nTrnSmpls; SizeSpec.broadcastable] beta *
            Expr.reshape [nOutput; SizeSpec.broadcastable; nTrnSmpls] beta
        //let betaBetaT = betaBetaT |> Expr.dump "betaBetaT"

        // lkLkT = lk .* lk.T
        // [smpl, gp, trn_smpl, 1] .* [smpl, gp, 1, trn_smpl] ==> [smpl, gp, trn_smpl, trn_smpl]
        // is equivalent to: [smpl, gp, trn_smpl, 1*] * [smpl, gp, 1*, trn_smpl]
        let lkLkT =
            Expr.reshape [nSmpls; nOutput; nTrnSmpls; SizeSpec.broadcastable] lk *
            Expr.reshape [nSmpls; nOutput; SizeSpec.broadcastable; nTrnSmpls] lk
        //let lkLkT = lkLkT |> Expr.dump "lkLkT"

        // Tr( (Kk_inv - betaBetaT) .*  L )
        // ([1*, gp, trn_smpl1, trn_smpl2] - [1*, gp, trn_smpl, trn_smpl]) .* [smpl, gp, trn_smpl1, trn_smpl2]
        //   ==> Tr ([smpl, gp, trn_smpl1, trn_smpl2]) ==> [smpl, gp]
        let var1 = Expr.padLeft (KkInv - betaBetaT) .* L  |> Expr.trace
        //let var1 = var1 |> Expr.dump "var1"
        
        // Tr( lkLkT .* betaBeta.T ) 
        // [smpl, gp, trn_smpl, trn_smpl] .* [1*, gp, trn_smpl, trn_smpl] 
        //  ==> Tr ([smpl, gp, trn_smpl1, trn_smpl2]) ==> [smpl, gp]
        let var2 = lkLkT .* (Expr.padLeft betaBetaT) |> Expr.trace
        //let var2 = var2 |> Expr.dump "var2"

        let predVar = 1.0f - var1 - var2
        //let pred_var = pred_var |> Expr.dump "pred_var"

        // T[smpl, gp1, gp2, trn_smpl1, trn_smpl2]
        //let T = Told nSmpls nGps nTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX
        let T = Tnew nSmpls nOutput nTrnSmpls mu sigma lengthscales trnX
        //let T = T |> Expr.dump "T"

        // calculate betaTbeta = beta.T .* T .* beta
        // beta[gp, trn_smpl]
        // T[smpl, gp1, gp2, trn_smpl1, trn_smpl2]
        // beta[gp1, trn_smpl1].T .* T[gp1,gp2, trn_smpl1, trn_smpl2] .* beta[gp2, trn_smpl2]
        // [1*, gp1, 1*, 1, trn_smpl1] .* [smpl, gp1, gp2, trn_smpl1, trn_smpl2] .* [1*, 1*, gp2, trn_smpl2, 1]
        // ==> [smpl, gp1, gp2, 1, 1]
        let bc = SizeSpec.broadcastable
        let one = SizeSpec.one
        let betaTbeta = 
            (Expr.reshape [bc; nOutput; bc; one; nTrnSmpls] beta) .* T .* 
            (Expr.reshape [bc; bc; nOutput; nTrnSmpls; one] beta)

        // [smpl, gp1, gp2, 1, 1] ==> [smpl, gp1, gp2]
        let betaTbeta =
            betaTbeta |> Expr.reshape [nSmpls; nOutput; nOutput]   
        //let betaTbeta = betaTbeta |> Expr.dump "betaTbeta"     

        // calculate m_k * m_l
        // [smpl, gp1, 1*] * [smpl, 1*, gp2]
        // ==> [smpl, gp1, gp2]
        let mkml = 
            (Expr.reshape [nSmpls; nOutput; bc] predMean) *
            (Expr.reshape [nSmpls; bc; nOutput] predMean)
        //let mkml = mkml |> Expr.dump "mkml"

        /// calculate pred_cov_without_var =  beta.T .* T .* beta - m_k * m_l
        let predCovWithoutVar = betaTbeta - mkml
        //let pred_cov_without_var = pred_cov_without_var |> Expr.dump "pred_cov_without_var"

        let predCov =
            match pars.HyperPars.OutputMode with
            // create zero matrix the size of the covariance matrix
            | MeanOnly          -> Expr.zeros<single> [nSmpls;nOutput;nOutput]
            // create matrix with diagonal variance in lowest dimensions
            | MeanVariance      -> setCovDiag nSmpls nOutput (Expr.zeros<single> [nSmpls;nOutput;nOutput]) predVar
            // replace diagonal in pred_cov_without_var by pred_var
            | MeanCovariance    -> setCovDiag nSmpls nOutput predCovWithoutVar predVar
        // replace diagonal in pred_cov_without_var by pred_var
        let predCov = setCovDiag nSmpls nOutput predCovWithoutVar predVar

        predMean, predCov, regTerm

/// Propagates a normal distribution through a weight matrix.
module WeightTransform =
    
    /// Hyper paramaters of the weight transform layer.
    type HyperPars = {
        /// number of inputs
        NInput:             SizeSpecT 

        /// number of outputs
        NOutput:            SizeSpecT
        /// optimize weights and bias during training
        Trainable:          bool
        /// weight initialization method
        WeightsInit:        InitMethod
        /// bias initialization method
        BiasInit:           InitMethod
        /// l1 regularization weight
        L1Regularization:   single option
        /// l2 regularization weight
        L2Regularization:   single option
    }

    /// The default hyper parameters.
    let defaultHyperPars = {
        /// number of inputs
        NInput              = SizeSpec.fix 0
        /// number of outputs
        NOutput             = SizeSpec.fix 0
        /// defines if weights are trained
        Trainable           = true
        /// weight initialization method
        WeightsInit         = FanOptimal
        /// bias initialization method
        BiasInit            = Const 0.0f
        /// l1 regularization weight
        L1Regularization    = None
        /// l2 regularization weight
        L2Regularization    = None
    }

    /// Parameter expressions.
    type Pars = {
        /// weights [nOutput, nInput]
        Weights:        ExprT 
        /// bias [nOutput]
        Bias:           ExprT
        /// hyper-parameters
        HyperPars:      HyperPars
    }

    /// Creates parameters.
    let pars (mb: ModelBuilder<_>) hp = {
        Weights   = mb.Param ("Weights", [hp.NOutput; hp.NInput], GPUtils.initVals hp.WeightsInit)
        Bias      = mb.Param ("Bias",    [hp.NOutput],            GPUtils.initVals hp.BiasInit)
        HyperPars = hp
    }

    /// Mean and variance after multiplication with the weight matrix.
    let transform pars (mu, sigma) =
        // [smpl,inp] .* [inp,out] + [out]
        // => [smpl,gp]
        let newMu = mu .* pars.Weights.T + pars.Bias
        // [1*,gp,inp] .* [smpl,inp,inp] => [smpl,gp,inp]
        // [smpl,gp,inp] .* [1*,inp,gp] => [smpl,gp,gp]
        // [1*,gp,inp] .* [smpl,inp,inp] .* [1*,inp,gp]
        // => [smpl,gp,gp]
        let nGps = pars.HyperPars.NOutput
        let nInput = pars.HyperPars.NInput
        let newSigma =  (Expr.reshape [SizeSpec.broadcastable; nGps; nInput] pars.Weights) .*
                        sigma .*
                        (Expr.reshape [SizeSpec.broadcastable; nInput; nGps] pars.Weights.T)
        newMu, newSigma
    
    /// Calculates sum of all regularization terms of this layer.
    let regularizationTerm pars  =
        let weights = pars.Weights
        if pars.HyperPars.Trainable then
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


/// Layer that propagates its input normal distribution through a weight matrix and activation
/// functions described by GPs.
module GPActivationLayer = 
    
    /// Hyper parameters of GP Activation layer.
    type HyperPars = {
        WeightTransform: WeightTransform.HyperPars
        Activation:      GPActivation.HyperPars
    }

    /// Default hyper parameters.
    let defaultHyperPars = {
        WeightTransform = WeightTransform.defaultHyperPars
        Activation      = GPActivation.defaultHyperPars
    }

    /// Parameter expressions.
    type Pars = {
        /// weight transform parameters
        WeightTransform: WeightTransform.Pars
        /// GP activation function parameters
        Activation:      GPActivation.Pars
        /// hyper-parameters
        HyperPars:       HyperPars
    }

    /// Creates parameters.
    let pars (mb: ModelBuilder<_>) (hp: HyperPars) = 
        if hp.Activation.NOutput <> hp.WeightTransform.NOutput then
            failwith "number of Outputs must equal number of output units in weight transform"
        {
            WeightTransform = WeightTransform.pars (mb.Module "WeightTransform") hp.WeightTransform
            Activation = GPActivation.pars (mb.Module "Activation") hp.Activation
            HyperPars = hp
        }


    /// Propagates the input normal distribution through a weight matrix and activation
    /// functions described by GPs.
    let pred (pars: Pars) (meanIn, covIn) = 
        let meanTf, covTf  = WeightTransform.transform pars.WeightTransform (meanIn, covIn) 
        let regWT =WeightTransform.regularizationTerm pars.WeightTransform
        let meanAct,covAct,regAct = GPActivation.pred pars.Activation (meanTf, covTf)
        meanAct, covAct,regWT+regAct


