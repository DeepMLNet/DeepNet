namespace SmoothADP

open System

open RProvider
open RProvider.graphics
open RProvider.grDevices
open RTools


open ArrayNDNS
open SymTensor




module SmoothADP =

    type HyperPars = {
        /// number of support points
        NSupport:   SizeSpecT
    }

    type Pars = {
        SupportX:       ExprT<single> ref
        SupportY:       ExprT<single> ref
        Lengthscale:    ExprT<single> ref
        Sigma:          ExprT<single> ref
        HyperPars:      HyperPars    
    }

    let pars  (mb: ModelBuilder<_>) hp = {
        SupportX       = mb.Param ("SupportX",    [hp.NSupport])
        SupportY       = mb.Param ("SupportY",    [hp.NSupport])
        Lengthscale    = mb.Param ("Lengthscale", [])
        Sigma          = mb.Param ("Sigma",       [])
        HyperPars      = hp 
    }

    /// squared exponential covariance matrix between X1 and X2
    let K nSmpls1 nSmpls2 lengthscale sigma X1 X2 = 
        // input  X1[smpl1]
        //        X2[smpl2]
        //        l[]
        //        s[]
        // output K[smpl1, smpl2]
        let smpl1 = ElemExpr.idx 0
        let smpl2 = ElemExpr.idx 1
        let l = ElemExpr.arg0D 0
        let s = ElemExpr.arg0D 1
        let x1 = ElemExpr.arg1D 2
        let x2 = ElemExpr.arg1D 3

        let kse =
            exp (- ((x1.[smpl1] - x2.[smpl2])**2.0f) / (2.0f * l**2.0f) ) +
            ElemExpr.ifThenElse smpl1 smpl2 (s**2.0f) (ElemExpr.zero())    
        let K = Expr.elements [nSmpls1; nSmpls2] kse [lengthscale; sigma; X1; X2]

        let dKsedX1 = kse |> ElemExprDeriv.compute |> ElemExprDeriv.ofArgElem x1.[smpl1]
        let dKdX1 = Expr.elements [nSmpls1; nSmpls2] dKsedX1 [lengthscale; sigma; X1; X2]
        
        K, dKdX1
    

    /// Predicts mean and mean of derivative.
    /// Input testX [smpl]
    /// Returns (testY[smpl], dtestY_dtestX[smpl]).
    let gp_predict pars testX =
        let nTest    = (Expr.shapeOf testX).[0]
        let nSupport = pars.HyperPars.NSupport

        // covariance matrix between training samples
        let KXX, _ = K nSupport nSupport !pars.Lengthscale !pars.Sigma !pars.SupportX !pars.SupportX

        // covariance matrix between test and training samples
        let KxX, dKxX_dx = K nTest nSupport !pars.Lengthscale (Expr.zero ()) testX !pars.SupportX

        // tgt = KXX^-1 .* supportY
        // [nSupport; nSupport] .* [nSupport] => [nSupport] 
        let tgt = (Expr.invert KXX) .* !pars.SupportY

        // testY = KxX .* KXX^-1 .* supportY
        // [nTest; nSupport] .* [nSupport] => [nTest]
        let testY = KxX .* tgt

        // d(testY) / d(testX) [nTest]
        //let dtestY_dtestX = Deriv.compute testY |> Deriv.ofVar testX |> Expr.diag
        let dtestY_dtestX = dKxX_dx .* tgt    // formula from paper

        testY, dtestY_dtestX







module Main =

    [<EntryPoint>]
    let main argv = 
        let device = DevHost
        let mb = ModelBuilder<single> "Main"

        let nSupport = mb.Size "NSupport"
        let nTest    = mb.Size "nTest"

        let testX = mb.Var "TestX" [nTest]

        let pars = SmoothADP.pars mb {NSupport=nSupport}

        mb.SetSize nSupport 10
        let mi = mb.Instantiate device

        let testY, dtestY = SmoothADP.gp_predict pars testX
        let testFn = mi.Func (testY, dtestY) |> arg1 testX
        
        // setup quadratic function      
        let sx = ArrayNDHost.linSpaced 1.f 10.f 10
        let sy = sx ** 2.0f
        mi.ParameterStorage.[!pars.SupportX] <- sx
        mi.ParameterStorage.[!pars.SupportY] <- sy
        mi.ParameterStorage.[!pars.Lengthscale] <- ArrayNDHost.scalar 1.0f
        mi.ParameterStorage.[!pars.Sigma] <- ArrayNDHost.scalar (sqrt 0.1f)

        // select test points
        let tStart, tStop = -2.f, 13.f
        let tx = ArrayNDHost.linSpaced tStart tStop 100
        let ty, dty = testFn tx
        

        printfn "SupportX=\n%A" sx
        printfn "SupportY=\n%A" sy
        printfn ""
        printfn "TestX=\n%s" tx.Full
        printfn "TestY=\n%s" ty.Full
        printfn "dTestY / dTestX=\n%s" dty.Full

        let toList (ary: #ArrayNDT<single>) =
            (box ary) :?> ArrayNDHostT<single> |> ArrayNDHost.toList |> List.map float

        R.pdfPage ("plot.pdf", 2)

        R.plot2 ([float tStart; float tStop], [-5.; 110.], "GP", "x", "y")
        R.lines2 (tx |> toList, ty |> toList)
        R.points2 (sx |> toList, sy |> toList) 

        R.plot2 ([float tStart; float tStop], [-30.; 30.], "GP", "x", "dy")
        R.lines2 (tx |> toList, dty |> toList)


        R.dev_off() |> ignore

        0