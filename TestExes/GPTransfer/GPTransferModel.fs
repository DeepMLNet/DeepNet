namespace GPTransfer

open ArrayNDNS
open SymTensor
open System

module MultiGPLayer =

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

    let pinv a =
        a|> Expr.transpose |> Expr.dot a |> Expr.invert |> Expr.dot (Expr.transpose a)

    ///The covariance Matrices of the training vectors with themselves 
    ///by GP instances with squared exponential covariance.
    let Kk nGps nTrnSmpls lengthscales trnX trnSigma = 
        // Kse element expression
        // input  x[gp, trn_smpl]
        //        l[gp]
        //        s[gp, trn_smpl]
        // output cov[gp, trn_smpl1, trn_smpl2]
        let gp = ElemExpr.idx 0   
        let trn_smpl1 = ElemExpr.idx 1
        let trn_smpl2 = ElemExpr.idx 2
        let l = ElemExpr.argElem 0
        let x = ElemExpr.argElem 1
        let s = ElemExpr.argElem 2
        let kse =
            exp (- ((x [gp; trn_smpl1] - x [gp; trn_smpl2])**2.0f) / (2.0f * (l [gp])**2.0f) ) +
            ElemExpr.ifThenElse trn_smpl1 trn_smpl2 (s [gp; trn_smpl1] ** 2.0f) (ElemExpr.zero())
        
        Expr.elements [nGps; nTrnSmpls; nTrnSmpls] kse [lengthscales; trnX; trnSigma]

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
        let m = ElemExpr.argElem 0
        let s = ElemExpr.argElem 1
        let l = ElemExpr.argElem 2
        let x = ElemExpr.argElem 3

        let lk1 = sqrt ( (l [gp])**2.0f / ((l [gp])**2.0f + s [smpl; gp; gp]) )
        let lk2 = exp ( -( (m [smpl; gp] - x [gp; trn_smpl])**2.0f / (2.0f * ((l [gp])**2.0f + s [smpl; gp; gp])) ) )
        let lk = lk1 * lk2

        Expr.elements [nSmpls; nGps; nTrnSmpls] lk [mu; sigma; lengthscales; trnX]


    ///Elementwise Matrix needed for calculation of the varance prediction.
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

    ///Elementwise Matrix needed for calculation of the covarance prediction.
    let T nSmpls nGps nTrnSmpls mu sigma lengthscales trnX =
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
        let m = ElemExpr.argElem 0
        let s = ElemExpr.argElem 1
        let l = ElemExpr.argElem 2
        let x = ElemExpr.argElem 3

        // Mathematica: k = gp1  l = gp2   i=t1   j=t2

        let eNom = (x[gp2;t2]-m[smpl;gp2])**2.f * (l[gp1]**2.f+s[smpl;gp1;gp1]) + (x[gp1;t1]-m[smpl;gp1]) * 
                   ( 2.f * (m[smpl;gp2]-x[gp2;t2]) * s[smpl;gp1;gp2] + (x[gp1;t1]-m[smpl;gp1]) * (l[gp2]**2.f + s[smpl;gp2;gp2]) ) 
        let eDnm = 2.f * ( (l[gp1]**2.f + s[smpl;gp1;gp1]) * (l[gp2]**2.f + s[smpl;gp2;gp2]) - s[smpl;gp1;gp2]**2.f )
        let e = exp(-eNom / eDnm)
        let Tnom = e * l[gp1] * l[gp2]

        let sq1 = s[smpl;gp1;gp1] * s[smpl;gp2;gp2] - s[smpl;gp1;gp2]**2.f
        let sq2Nom = s[smpl;gp1;gp2]**2.f - (l[gp1]**2.f + s[smpl;gp1;gp1]) * (l[gp2]**2.f + s[smpl;gp2;gp2])
        let sq2Dnm = s[smpl;gp1;gp2]**2.f - s[smpl;gp1;gp1] * s[smpl;gp2;gp2]
        let Tdnm = sqrt (sq1 * sq2Nom / sq2Dnm)

        let T = Tnom / Tdnm
        Expr.elements [nSmpls; nGps; nGps; nTrnSmpls; nTrnSmpls] T [mu; sigma; lengthscales; trnX]


    let setCovDiag nSmpls nGps cov var =
        // replace covariance matrix diagonal by variance
        // inputs  cov[smpl, gp1, gp2]
        //         var[smpl, gp
        // output  cov[smpl, gp1, gp2]
        let smpl = ElemExpr.idx 0
        let gp1 = ElemExpr.idx 1
        let gp2 = ElemExpr.idx 2
        let c = ElemExpr.argElem 0
        let v = ElemExpr.argElem 1

        let cv = ElemExpr.ifThenElse gp1 gp2 (v[smpl; gp1]) (c[smpl; gp1; gp2])
        Expr.elements [nSmpls; nGps; nGps] cv [cov; var]

    ///Predicted mean and covariance from input mean and covariance.
    let pred pars mu sigma =
        // mu:    input mean        [smpl, gp]
        // Sigma: input covariance  [smpl, gp1, gp2]

        let nSmpls = (Expr.shapeOf mu).[0]
        let nGps = pars.HyperPars.NGPs
        let nTrnSmpls = pars.HyperPars.NTrnSmpls

        // Kk [gp, trn_smpl1, trn_smpl2]
        let Kk = Kk nGps nTrnSmpls !pars.Lengthscales !pars.TrnX !pars.TrnSigma
        let Kk = Kk |> Expr.dump "Kk"
        let Kk_inv = Expr.invert Kk
        let Kk_inv = Kk_inv |> Expr.dump "Kk_inv"
        let Kk_pinv = pinv Kk
        let Kk_pinv = Kk_pinv |> Expr.dump "Kk_pinv"
//        let Kk_inv = Kk_pinv
        // lk [smpl, gp, trn_smpl]
        let lk = lk nSmpls nGps nTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX
        let lk = lk |> Expr.dump "lk"
        // trnT [gp, trn_smpl]
        let trnT = pars.TrnT

        // ([gp, trn_smpl1, trn_smpl2] .* [gp, trn_smpl])       
        // ==> beta [gp, trn_smpl]
        let beta = Kk_inv .* !trnT
        let beta = beta |> Expr.dump "beta"

        // ==> sum ( [smpl, gp, trn_smpl] * beta[1*, gp, trn_smpl], trn_smpl)
        // ==> pred_mean [smpl, gp]
        let pred_mean = lk * Expr.padLeft beta |> Expr.sumAxis 2
        let pred_mean = pred_mean |> Expr.dump "pred_mean"

        // L[smpl, gp, trn_smpl1, trn_smpl2]
        let L = L nSmpls nGps nTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX
     
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

        // T[smpl, gp1, gp2, trn_smpl1, trn_smpl2]
        let T = T nSmpls nGps nTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX

        // calculate pred_cov = beta.T .* T .* beta
        // beta[gp, trn_smpl]
        // T[smpl, gp1, gp2, trn_smpl1, trn_smpl2]
        // beta[gp1, trn_smpl1].T .* T[gp1,gp2, trn_smpl1, trn_smpl2] .* beta[gp2, trn_smpl2]
        // [1*, gp1, 1*, 1, trn_smpl1] .* [smpl, gp1, gp2, trn_smpl1, trn_smpl2] .* [1*, 1*, gp2, trn_smpl2, 1]
        // ==> [smpl, gp1, gp2, 1, 1]
        let bc = SizeSpec.broadcastable
        let one = SizeSpec.one
        let pred_cov_without_var = 
            (Expr.reshape [bc; nGps; bc; one; nTrnSmpls] beta) .* T .* 
            (Expr.reshape [bc; bc; nGps; nTrnSmpls; one] beta)
        
        let meankmeanl = 
            (Expr.reshape [nSmpls; nGps; bc] pred_mean) .*
            (Expr.reshape [nSmpls; bc; nGps] pred_mean)
//        ///[smpl, gp1, gp2, 1, 1] - [smpl, gp1] .* [smpl, gp2]
//        let pred_cov_without_var = 
//            pred_cov_without_var -
//            (Expr.reshape [nSmpls; nGps; nGps;one;one] meankmeanl)
        let pred_cov_without_var =
            pred_cov_without_var |> Expr.reshape [nSmpls; nGps; nGps]

        // replace diagonal in pred_cov_without_var by pred_var
        let pred_cov = setCovDiag nSmpls nGps pred_cov_without_var pred_var

        pred_mean, pred_cov




