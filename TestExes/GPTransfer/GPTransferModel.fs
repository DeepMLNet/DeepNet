namespace GPTransfer

open ArrayNDNS
open SymTensor


module GPActivationLayer =

    type HyperPars = {
        /// number of units, i.e. number of GPs
        NGPs:     SizeSpecT

        /// number of training samples for each GP
        NTrnSmpls:  SizeSpecT
    }

    type Pars = {
        /// GP lengthscales: [gp]
        Lengthscales:       ExprT<single> ref
        /// x values of GP training samples:         [gp, trn_smpl]
        TrnX:               ExprT<single> ref
        /// target values of GP training samples:    [gp, trn_smpl]
        TrnT:               ExprT<single> ref
        /// standard deviation of GP target values:  [gp, trn_smpl]
        TrnSigma:           ExprT<single> ref

        /// hyper-parameters
        HyperPars:          HyperPars
    }

    let internal initLengthscales seed (shp: int list) : ArrayNDHostT<'T> = 
        ArrayNDHost.zeros shp

    let internal initTrnX seed (shp: int list) : ArrayNDHostT<'T> = 
        ArrayNDHost.zeros shp

    let internal initTrnT seed (shp: int list) : ArrayNDHostT<'T> = 
        ArrayNDHost.zeros shp

    let internal initTrnSigma seed (shp: int list) : ArrayNDHostT<'T> = 
        ArrayNDHost.zeros shp


    let pars (mb: ModelBuilder<_>) hp = {
        Lengthscales   = mb.Param ("Lengthscales", [hp.NGPs],               initLengthscales)
        TrnX           = mb.Param ("TrnX",         [hp.NGPs; hp.NTrnSmpls], initTrnX)
        TrnT           = mb.Param ("TrnT",         [hp.NGPs; hp.NTrnSmpls], initTrnT)
        TrnSigma       = mb.Param ("TrnSigma",     [hp.NGPs; hp.NTrnSmpls], initTrnSigma)
        HyperPars      = hp
    }

    let Kk nGps nTrnSmpls lengthscales trnX trnSigma = 
        // Kse element expression
        // input  x[gp, trn_smpl]
        //        l[gp]
        //        sigma[gp, trn_smpl]
        // output cov[gp, trn_smpl1, trn_smpl2]
        let gp = ElemExpr.idx 0   
        let trn_smpl1 = ElemExpr.idx 1
        let trn_smpl2 = ElemExpr.idx 2
        let l = ElemExpr.argElem 0
        let x = ElemExpr.argElem 1
        let sigma = ElemExpr.argElem 2
        let kse =
            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])**2.0f) / (2.0f * (l [gp])**2.0f) ) +
            ElemExpr.kroneckerIf trn_smpl1 trn_smpl2 (sigma [gp; trn_smpl1])
        
        Expr.elements [nGps; nTrnSmpls; nTrnSmpls] kse [lengthscales; trnX; trnSigma]

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
        let m = ElemExpr.argElem 0
        let s = ElemExpr.argElem 1
        let l = ElemExpr.argElem 2
        let x = ElemExpr.argElem 3

        let lk1 = sqrt ( (l [gp])**2.0f / ((l [gp])**2.0f + s [smpl; gp; gp]) )
        let lk2 = exp ( -( (m [smpl; gp] - x [gp; trn_smpl])**2.0f / (2.0f * ((l [gp])**2.0f + s [smpl; gp; gp])) ) )
        let lk = lk1 * lk2

        Expr.elements [nSmpls; nGps; nTrnSmpls] lk [mu; sigma; lengthscales; trnX]



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
        let m = ElemExpr.argElem 0
        let s = ElemExpr.argElem 1
        let l = ElemExpr.argElem 2
        let x = ElemExpr.argElem 3

        let L1 = sqrt ( (l [gp])**2.0f / ((l [gp])**2.0f + 2.0f * s [smpl; gp; gp]) )
        let L2a = ( m [smpl; gp] - (x [gp; trn_smpl1] + x [gp; trn_smpl2])/2.0f )**2.0f / ((l [gp])*2.0f + 2.0f * s [smpl; gp; gp])
        let L2b = (x [gp; trn_smpl1] - x [gp; trn_smpl2])**2.0f / (4.0f * (l [gp])**2.0f)
        let L2 = exp (-L2a - L2b)
        let L = L1 * L2

        Expr.elements [nSmpls; nGps; nTrnSmpls; nTrnSmpls] L [mu; sigma; lengthscales; trnX]


    let pred pars mu sigma =
        // mu:    input mean        [smpl, gp]
        // Sigma: input covariance  [smpl, gp1, gp2]

        let nSmpls = (Expr.shapeOf mu).[0]
        let nGps = pars.HyperPars.NGPs
        let nTrnSmpls = pars.HyperPars.NTrnSmpls

        // Kk [gp, trn_smpl1, trn_smpl2]
        let Kk = Kk pars.HyperPars.NGPs pars.HyperPars.NTrnSmpls !pars.Lengthscales !pars.TrnX !pars.TrnSigma
        let Kk_inv = Expr.invert Kk
        // lk [smpl, gp, trn_smpl]
        let lk = lk nSmpls pars.HyperPars.NGPs pars.HyperPars.NTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX
        // trnT [gp, trn_smpl]
        let trnT = pars.TrnT

        // ([gp, trn_smpl1, trn_smpl2] .* [gp, trn_smpl])       
        // ==> beta [gp, trn_smpl]
        let beta = Kk_inv .* !trnT

        // ==> sum ( [smpl, gp, trn_smpl] * beta[1*, gp, trn_smpl], trn_smpl)
        // ==> pred_mean [smpl, gp]
        let pred_mean = lk * Expr.padLeft beta |> Expr.sumAxis 2

        // L[smpl, gp, trn_smpl1, trn_smpl2]
        let L = L nSmpls pars.HyperPars.NGPs pars.HyperPars.NTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX
     
        // betaBetaT = beta .* beta.T
        // [gp, trn_smpl, 1] .* [gp, 1, trn_smpl] ==> [gp, trn_smpl, trn_smpl]
        // is equivalent to: [gp, trn_smpl, 1*] * [gp, 1*, trn_smpl]
        let betaBetaT = 
            Expr.reshape [nGps; nTrnSmpls; SizeSpec.broadcastable] beta *
            Expr.reshape [nGps; SizeSpec.broadcastable; nTrnSmpls] beta

        // lkLkT = lk .* lk.T
        // [smpl, gp, trn_smpl, 1] .* [smpl, gp, 1, trn_smpl] ==> [smpl, gp, trn_smpl, trn_smpl]
        // is equivalent to: [smpl, gp, trn_smpl, 1*] * [smpl, gp, 1*, trn_smpl]
        let lkLkT =
            Expr.reshape [nSmpls; nGps; nTrnSmpls; SizeSpec.broadcastable] lk *
            Expr.reshape [nSmpls; nGps; SizeSpec.broadcastable; nTrnSmpls] lk

        // Tr( (Kk_inv - betaBetaT) .*  L )
        // ([1*, gp, trn_smpl1, trn_smpl2] - [1*, gp, trn_smpl, trn_smpl]) .* [smpl, gp, trn_smpl1, trn_smpl2]
        //   ==> Tr ([smpl, gp, trn_smpl1, trn_smpl2]) ==> [smpl, gp]
        let var1 = Expr.padLeft (Kk_inv - betaBetaT) .* L  |> Expr.trace

        // Tr( lkLkT .* betaBeta.T )
        // [smpl, gp, trn_smpl, trn_smpl] .* [1*, gp, trn_smpl, trn_smpl] 
        //  ==> Tr ([smpl, gp, trn_smpl1, trn_smpl2]) ==> [smpl, gp]
        let var2 = lkLkT .* (Expr.padLeft betaBetaT) |> Expr.trace

        let pred_var = 1.0f - var1 - var2


        pred_mean, pred_var




