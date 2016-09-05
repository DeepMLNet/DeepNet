namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System

module Program =

    let post device (x: ArrayNDT<'T>) =
        if device = DevCuda then ArrayNDCuda.toDev (x :?> ArrayNDHostT<'T>) :> ArrayNDT<'T>
        else x 
    
    let testMultiGPLayer device =
        
        let seed = 1
        let rand = Random(seed)

        let ngps = 3
        let ntraining = 10
        let ntest = 10

        let lhost = [1.0f; 1.5f; 2.0f] |> ArrayNDHost.ofList 
        let l = lhost |> ArrayNDCuda.toDev
        let trn_xhost =  rand.UniformArrayND (-5.0f ,5.0f) [ngps;ntraining] 


        let trn_x = trn_xhost  |> ArrayNDCuda.toDev
        let trn_thost = rand.UniformArrayND (-5.0f ,5.0f) [ngps;ntraining] 
        let trn_t = trn_thost  |> ArrayNDCuda.toDev
        let trn_sigmahost = rand.UniformArrayND (-5.0f ,5.0f) [ngps;ntraining] 
        let trn_sigma = trn_sigmahost  |> ArrayNDCuda.toDev
        
        
        let mb = ModelBuilder<single> "Test"

        let nSmpls    = mb.Size "nSmpls"
        let nGPs      = mb.Size "nGPs"
        let nTrnSmpls = mb.Size "nTrnSmpls"
        
        let mgp = 
            MultiGPLayer.pars (mb.Module "MGP") {NGPs=nGPs; NTrnSmpls=nTrnSmpls}
        let inp_mean  : ExprT<single> = mb.Var "inp_mean"  [nSmpls; nGPs]
        let inp_cov   : ExprT<single> = mb.Var "inp_cov"   [nSmpls; nGPs; nGPs]

        mb.SetSize nGPs      3
        mb.SetSize nTrnSmpls 10
        let mi = mb.Instantiate device

        let pred_mean, pred_cov = MultiGPLayer.pred mgp inp_mean inp_cov
        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg2 inp_mean inp_cov





        for i in [0..9] do

            let inp_meanhost = rand.UniformArrayND (-5.0f ,5.0f) [1;ngps]
            let inp_mean_val = inp_meanhost |> ArrayNDCuda.toDev

            let inp_covhost =  rand.UniformArrayND (-5.0f ,5.0f) [1;ngps;ngps]
            let inp_cov_val = inp_covhost |> ArrayNDCuda.toDev

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

        0


