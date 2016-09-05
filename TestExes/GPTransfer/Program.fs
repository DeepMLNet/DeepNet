namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda


module Program =

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

        printfn "%A" mi.ParameterStorage.[!mgp.TrnX]

        let pred_mean, pred_cov = MultiGPLayer.pred mgp inp_mean inp_cov

        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg2 inp_mean inp_cov


        let inp_mean_val = [[1.0f; 2.0f]] |> ArrayNDHost.ofList2D |> ArrayNDCuda.toDev
        let inp_cov_val =  Array3D.zeroCreate 1 2 2
        inp_cov_val.[0,0,0] <- 1.0f
        let inp_cov_val = inp_cov_val |> ArrayNDHost.ofArray3D |> ArrayNDCuda.toDev

        let pred_mean, pred_cov = pred_mean_cov_fn inp_mean_val inp_cov_val

        printfn "inp_mean=\n%A" inp_mean_val
        printfn "inp_cov=\n%A" inp_cov_val
        printfn ""
        printfn "pred_mean=\n%A" pred_mean
        printfn "pred_cov=\n%A" pred_cov



    [<EntryPoint>]
    let main argv = 
        testMultiGPLayer () |> ignore

        0


