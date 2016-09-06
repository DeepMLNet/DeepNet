namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda


module Program =

    let post device (x: ArrayNDT<'T>) =
        if device = DevCuda then ArrayNDCuda.toDev (x :?> ArrayNDHostT<'T>) :> ArrayNDT<'T>
        else x 
    
    let testMultiGPLayer device =
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
        let mi = mb.Instantiate device

        let pred_mean, pred_cov = MultiGPLayer.pred mgp inp_mean inp_cov
        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg2 inp_mean inp_cov



        mi.ParameterStorage.[!mgp.Lengthscales] <- [1.0f; 1.0f] 
                                                   |> ArrayNDHost.ofList
                                                   |> post device
        mi.ParameterStorage.[!mgp.TrnX] <- [[1.0f; 2.0f; 2.5f]
                                            [4.1f; 4.3f; 4.4f]] 
                                           |> ArrayNDHost.ofList2D
                                           |> post device
        mi.ParameterStorage.[!mgp.TrnT] <- [[1.0f; 2.0f; 2.5f]
                                            [4.1f; 4.3f; 4.4f]] 
                                           |> ArrayNDHost.ofList2D
                                           |> post device
        mi.ParameterStorage.[!mgp.TrnSigma] <- [[0.0f; 0.0f; 0.0f]
                                                [0.0f; 0.0f; 0.0f]] 
                                               |> ArrayNDHost.ofList2D
                                               |> post device

        let inp_mean_val = [[1.0f; 2.0f]] |> ArrayNDHost.ofList2D |> post device
        let inp_cov_val =  Array3D.zeroCreate 1 2 2
        //inp_cov_val.[0,0,0] <- 1.0f
        let inp_cov_val = inp_cov_val |> ArrayNDHost.ofArray3D |> post device

        let pred_mean, pred_cov = pred_mean_cov_fn inp_mean_val inp_cov_val

        printfn "Lengthscales=\n%A" mi.ParameterStorage.[!mgp.Lengthscales]
        printfn "TrnX=\n%A" mi.ParameterStorage.[!mgp.TrnX]
        printfn "TrnT=\n%A" mi.ParameterStorage.[!mgp.TrnT]
        printfn "TrnSigma=\n%A" mi.ParameterStorage.[!mgp.TrnSigma]
        printfn ""
        printfn "inp_mean=\n%A" inp_mean_val
        printfn "inp_cov=\n%A" inp_cov_val
        printfn ""
        printfn "pred_mean=\n%A" pred_mean
        printfn "pred_cov=\n%A" pred_cov



    [<EntryPoint>]
    let main argv = 
        TestUtils.evalHostCuda testMultiGPLayer
        //TestUtils.compareTraces testMultiGPLayer false |> ignore
        //testMultiGPLayer DevCuda |> ignore
        //testMultiGPLayer DevHost |> ignore

        let dataArray = ArrayNDHost.zeros<single> [100]

        let a = Expr.zeroMatrix (SizeSpec.fix 3) (SizeSpec.fix 3)

        let abelIp = Expr.createInterpolator1D dataArray -1.0f 3.0f 0.1f
        let abel = Expr.interpolate1D abelIp
        let c = abel a


        0


