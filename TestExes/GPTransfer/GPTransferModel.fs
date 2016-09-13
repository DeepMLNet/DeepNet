namespace GPTransfer

open ArrayNDNS
open SymTensor
open System
open Basics
open Models

module MultiGPLayer =

    type HyperPars = {
        /// number of units, i.e. number of GPs = Number of outputs
        NGPs:       SizeSpecT

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


    let internal initLengthscales seed (shp: int list) : ArrayNDHostT<single> = 
         let rng = System.Random seed
         //Right now: all GPs equal
         ArrayNDHost.ones shp

    let internal initTrnX seed (shp: int list) : ArrayNDHostT<single> = 
        let n_gps = shp.[0]
        let n_trn = shp.[1]
        let rng = System.Random seed
        //Right now: all GPs equal
        rng.SortedUniformArrayND (-10.0f,10.0f) shp

    let internal initTrnT seed (shp: int list) : ArrayNDHostT<single> = 
        ArrayNDHost.zeros shp

    let internal initTrnSigma seed (shp: int list) : ArrayNDHostT<single> = 
        (ArrayNDHost.ones<single> shp) * sqrt 0.1f
    

    let pars (mb: ModelBuilder<_>) hp = {
        Lengthscales   = mb.Param ("Lengthscales", [hp.NGPs],               initLengthscales)
        TrnX           = mb.Param ("TrnX",         [hp.NGPs; hp.NTrnSmpls], initTrnX)
        TrnT           = mb.Param ("TrnT",         [hp.NGPs; hp.NTrnSmpls], initTrnT)
        TrnSigma       = mb.Param ("TrnSigma",     [hp.NGPs; hp.NTrnSmpls], initTrnSigma)
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
        let L2a = ( m [smpl; gp] - (x [gp; trn_smpl1] + x [gp; trn_smpl2])/2.0f )**2.0f / ((l [gp])**2.0f + 2.0f * s [smpl; gp; gp])
        let L2b = (x [gp; trn_smpl1] - x [gp; trn_smpl2])**2.0f / (4.0f * (l [gp])**2.0f)
        let L2 = exp (-L2a - L2b)
        let L = L1 * L2

        Expr.elements [nSmpls; nGps; nTrnSmpls; nTrnSmpls] L [mu; sigma; lengthscales; trnX]

    ///Elementwise Matrix needed for calculation of the covarance prediction.
    let Told nSmpls nGps nTrnSmpls mu sigma lengthscales trnX =
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

        let T = ElemExpr.ifThenElse gp1 gp2 (ElemExpr.zero ()) (Tnom / Tdnm)
        Expr.elements [nSmpls; nGps; nGps; nTrnSmpls; nTrnSmpls] T [mu; sigma; lengthscales; trnX]

    ///Elementwise Matrix needed for calculation of the covarance prediction.
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

        let Tdnm = sqrt ( (l[gp1]**2.f + s[smpl;gp1;gp1]) * (l[gp2]**2.f + s[smpl;gp2;gp2]) - s[smpl;gp1;gp2]**2.f )

        let T = ElemExpr.ifThenElse gp1 gp2 (ElemExpr.zero()) (Tnom / Tdnm)
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
    let pred pars (mu, sigma) =
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
        let L = L |> Expr.dump "L"
     
        // betaBetaT = beta .* beta.T
        // [gp, trn_smpl, 1] .* [gp, 1, trn_smpl] ==> [gp, trn_smpl, trn_smpl]
        // is equivalent to: [gp, trn_smpl, 1*] * [gp, 1*, trn_smpl]
        let betaBetaT = 
            Expr.reshape [nGps; nTrnSmpls; SizeSpec.broadcastable] beta *
            Expr.reshape [nGps; SizeSpec.broadcastable; nTrnSmpls] beta
        let betaBetaT = betaBetaT |> Expr.dump "betaBetaT"

        // lkLkT = lk .* lk.T
        // [smpl, gp, trn_smpl, 1] .* [smpl, gp, 1, trn_smpl] ==> [smpl, gp, trn_smpl, trn_smpl]
        // is equivalent to: [smpl, gp, trn_smpl, 1*] * [smpl, gp, 1*, trn_smpl]
        let lkLkT =
            Expr.reshape [nSmpls; nGps; nTrnSmpls; SizeSpec.broadcastable] lk *
            Expr.reshape [nSmpls; nGps; SizeSpec.broadcastable; nTrnSmpls] lk
        let lkLkT = lkLkT |> Expr.dump "lkLkT"

        // Tr( (Kk_inv - betaBetaT) .*  L )
        // ([1*, gp, trn_smpl1, trn_smpl2] - [1*, gp, trn_smpl, trn_smpl]) .* [smpl, gp, trn_smpl1, trn_smpl2]
        //   ==> Tr ([smpl, gp, trn_smpl1, trn_smpl2]) ==> [smpl, gp]
        let var1 = Expr.padLeft (Kk_inv - betaBetaT) .* L  |> Expr.trace
        let var1 = var1 |> Expr.dump "var1"
        
        // Tr( lkLkT .* betaBeta.T ) 
        // [smpl, gp, trn_smpl, trn_smpl] .* [1*, gp, trn_smpl, trn_smpl] 
        //  ==> Tr ([smpl, gp, trn_smpl1, trn_smpl2]) ==> [smpl, gp]
        let var2 = lkLkT .* (Expr.padLeft betaBetaT) |> Expr.trace
        let var2 = var2 |> Expr.dump "var2"

        let pred_var = 1.0f - var1 - var2
        let pred_var = pred_var |> Expr.dump "pred_var"

        // T[smpl, gp1, gp2, trn_smpl1, trn_smpl2]
        //let T = Told nSmpls nGps nTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX
        let T = Tnew nSmpls nGps nTrnSmpls mu sigma !pars.Lengthscales !pars.TrnX
        let T = T |> Expr.dump "T"

        // calculate betaTbeta = beta.T .* T .* beta
        // beta[gp, trn_smpl]
        // T[smpl, gp1, gp2, trn_smpl1, trn_smpl2]
        // beta[gp1, trn_smpl1].T .* T[gp1,gp2, trn_smpl1, trn_smpl2] .* beta[gp2, trn_smpl2]
        // [1*, gp1, 1*, 1, trn_smpl1] .* [smpl, gp1, gp2, trn_smpl1, trn_smpl2] .* [1*, 1*, gp2, trn_smpl2, 1]
        // ==> [smpl, gp1, gp2, 1, 1]
        let bc = SizeSpec.broadcastable
        let one = SizeSpec.one
        let betaTbeta = 
            (Expr.reshape [bc; nGps; bc; one; nTrnSmpls] beta) .* T .* 
            (Expr.reshape [bc; bc; nGps; nTrnSmpls; one] beta)

        // [smpl, gp1, gp2, 1, 1] ==> [smpl, gp1, gp2]
        let betaTbeta =
            betaTbeta |> Expr.reshape [nSmpls; nGps; nGps]   
        let betaTbeta = betaTbeta |> Expr.dump "betaTbeta"     

        // calculate m_k * m_l
        // [smpl, gp1, 1*] * [smpl, 1*, gp2]
        // ==> [smpl, gp1, gp2]
        let mkml = 
            (Expr.reshape [nSmpls; nGps; bc] pred_mean) *
            (Expr.reshape [nSmpls; bc; nGps] pred_mean)
        let mkml = mkml |> Expr.dump "mkml"

        /// calculate pred_cov_without_var =  beta.T .* T .* beta - m_k * m_l
        let pred_cov_without_var = betaTbeta - mkml
        let pred_cov_without_var = pred_cov_without_var |> Expr.dump "pred_cov_without_var"

        // replace diagonal in pred_cov_without_var by pred_var
        let pred_cov = setCovDiag nSmpls nGps pred_cov_without_var pred_var
        let pred_cov = pred_cov |> Expr.dump "pred_cov"

        pred_mean , pred_cov



/// [["nInput"; "nBatch"]; ["nInput"; "nHidden"]]
module WeightLayer =
    type HyperPars = {
        /// number of inputs
        NInput:     SizeSpecT 

        /// number of units, i.e. number of GPs = Number of outputs
        NGPs:       SizeSpecT
    }

    /// Weight layer parameters.
    type Pars = {
        /// expression for the weights [nGPs,nInput]
        Weights:        ExprT<single> ref
        /// hyper-parameters
        HyperPars:      HyperPars
    }

    let internal initWeights seed (shp: int list) : ArrayNDHostT<single> = 
        let fanOut = shp.[0] |> single
        let fanIn = shp.[1] |> single
        let r = 4.0f * sqrt (6.0f / (fanIn + fanOut))
        let rng = System.Random seed
        
        rng.SeqSingle(-r, r)
        |> ArrayNDHost.ofSeqWithShape shp

    let pars (mb: ModelBuilder<_>) hp = {
        Weights   = mb.Param ("Weights", [hp.NGPs; hp.NInput], initWeights)
        HyperPars = hp
    }

    let transform pars (mu,sigma) =
        //[smpl,inp] .* [inp,gp] 
        //=>[smpl,gp]
        let newMu = mu .* (!pars.Weights).T
        //[1*,gp,inp] .* [smpl,inp,inp] => [smpl,gp,inp]
        //[smpl,gp,inp] .* [1*,inp,gp] => [smpl,gp,gp]
        //[1*,gp,inp] .* [smpl,inp,inp] .* [1*,inp,gp]
        //=> [smpl,gp,gp]
        let nGps = pars.HyperPars.NGPs
        let nInput = pars.HyperPars.NInput
        let bc = SizeSpec.broadcastable
        let newSigma =  (Expr.reshape [bc;nGps;nInput] !pars.Weights) .*
                        sigma .*
                        (Expr.reshape [bc;nInput;nGps] (!pars.Weights).T)
        newMu, newSigma

module GPTransferUnit = 
    type HyperPars = {
        /// number of inputs
        NInput:    SizeSpecT

        /// number of units, i.e. number of GPs = Number of outputs
        NGPs:       SizeSpecT

        /// number of training samples for each GP
        NTrnSmpls:  SizeSpecT

    }

        /// Weight layer parameters.
    type Pars = {
        // WeightLayer
        WeightL:        WeightLayer.Pars
        //MultiGPLayer
        MultiGPL:       MultiGPLayer.Pars
        /// hyper-parameters
        HyperPars:      HyperPars
    }

    let pars (mb: ModelBuilder<_>) (hp: HyperPars) = {
        WeightL = WeightLayer.pars (mb.Module "WeigltL") 
            {NInput = hp.NInput; NGPs = hp.NGPs}
        MultiGPL = MultiGPLayer.pars (mb.Module "MultiGPL")
            {NGPs = hp.NGPs; NTrnSmpls = hp.NTrnSmpls}
        HyperPars = hp
    }

    let pred (pars: Pars) input = 
        WeightLayer.transform pars.WeightL input 
        |> MultiGPLayer.pred pars.MultiGPL


module InitialLayer =

    let cov input =
        let nSmpls = (Expr.shapeOf input).[0]
        let nInput = (Expr.shapeOf input).[1]
        let bc = SizeSpec.broadcastable
        // [smpl,inp1,1] .* [smpl,1,in2] => [smpl,in1,in2]
        // is equivalent to [smpl,inp1,1*] * [smpl,1*,in2] => [smpl,in1,in2]
        Expr.reshape [nSmpls; nInput; bc] input *
        Expr.reshape [nSmpls; bc; nInput] input

    let transform input =
        input, (cov input)

//module MLGPT = 
//    
//    type HyperPars = {
//        /// a list of the hyper parameters of the layers
//        Layers: GPTransferUnit.HyperPars list
//        /// the loss measure
//        LossMeasure: LossLayer.Measures
//    }
//
//    type Pars<'T> = {
//        /// a lsit of the parameters of the GPTransfer Layers
//        Layers:     GPTransferUnit.Pars<'T> list
//        /// hyper-parameters
//        HyperPars:  HyperPars 
//    }
//
//    let pars (mb: ModelBuilder<_>) (hp:HyperPars) = {
//        Layers = hp.Layers
//        |>List.mapi (fun idx gphp -> 
//                    GPTransferUnit.pars (mb.Module (sprintf "Layer%d" idx)) gphp)
//        HyperPars = hp
//    } 
//    
//    let pred (pars: Pars<'T>) input = 
//        let inputDist = initialLayer.transform input
//        (inputDist, pars.Layers)
//        ||> List.fold (fun inp p -> GPTransferUnit.pred p inp)
//
//
//    let loss pars input target =
//        let predmu,predSigma = (pred pars input)
//        LossLayer.loss pars.HyperPars.LossMeasure predmu target