namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda


module Program =

    //let testSquaredExponentialCovarianceMatrix () =
    //    let nGps = SizeSpec.symbol "nGps"
    //    let nTrnSmpls = SizeSpec.symbol "nTrnSmpls"
    //
    //    let trnX = Expr.var<single> "trnX" [nGps; nTrnSmpls]
    //    let lengthscale = Expr.var "lengthscale" [nGps]
    //
    //    let cm = GPTransferOps.squaredExponentialCovarianceMatrix trnX lengthscale
    //    let cmFn = Func.make DevHost.DefaultFactory cm |> arg2 trnX lengthscale
    //
    //    let derivs = Deriv.compute cm
    //    let cmWrtTrnX = derivs |> Deriv.ofVar trnX
    //    let cmWrtLengthscale = derivs |> Deriv.ofVar lengthscale
    //    let derivFn = Func.make2 DevHost.DefaultFactory cmWrtTrnX cmWrtLengthscale |> arg2 trnX lengthscale
    //
    //    let trnXVal = [[1.0f; 1.1f; 2.0f]] |> ArrayNDHost.ofList2D
    //    let lengthscaleVal = [0.5f] |> ArrayNDHost.ofList
    //    let cmVal = cmFn trnXVal lengthscaleVal
    //    let cmWrtTrnXVal, cmWrtLengthscaleVal = derivFn trnXVal lengthscaleVal 
    //
    //    printfn "trnXVal=\n%A" trnXVal
    //    printfn "lengthscaleVal=\n%A" lengthscaleVal
    //    printfn "cmVal=\n%A" cmVal
    //    printfn ""
    //    printfn "d(cmVal)/d(trnXVal)=\n%A" cmWrtTrnXVal
    //    printfn "d(cmVal)/d(lengthscaleVal)=\n%A" cmWrtLengthscaleVal
    

    let testMultiGPLayer () =
        let mb = ModelBuilder<single> "Test"

        let nSmpls    = mb.Size "nSmpls"
        let nGPs      = mb.Size "nGPs"
        let nTrnSmpls = mb.Size "nTrnSmpls"
        
        let mgp = 
            MultiGPLayer.pars (mb.Module "MGP") {NGPs=nGPs; NTrnSmpls=nTrnSmpls}

        let inp_mean  : ExprT<single> = mb.Var "inp_mean"  [nSmpls; nGPs]
        let inp_cov   : ExprT<single> = mb.Var "inp_cov"   [nSmpls; nGPs; nGPs]

        mb.SetSize nGPs      2
        mb.SetSize nTrnSmpls 3
        let mi = mb.Instantiate DevCuda

        let pred_mean, pred_cov = MultiGPLayer.pred mgp inp_mean inp_cov

        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg2 inp_mean inp_cov


        let inp_mean_val = [[1.0f; 2.0f]] |> ArrayNDHost.ofList2D
        let inp_cov_val =  Array3D.zeroCreate 1 2 2
        inp_cov_val.[0,0,0] <- 1.0f
        let inp_cov_val = inp_cov_val |> ArrayNDHost.ofArray3D

        pred_mean_cov_fn inp_mean_val inp_cov_val


    [<EntryPoint>]
    let main argv = 
        //MathInterface.doMathTest ()
        //MathInterface.doMathTest2 ()

        //testSquaredExponentialCovarianceMatrix ()

        testMultiGPLayer () |> ignore

        0


