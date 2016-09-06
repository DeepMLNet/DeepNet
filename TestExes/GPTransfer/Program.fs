namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System
open Datasets

module Program =

    let post device (x: ArrayNDT<'T>) =
        if device = DevCuda then ArrayNDCuda.toDev (x :?> ArrayNDHostT<'T>) :> ArrayNDT<'T>
        else x 
    
    type trainData ={
        Lengthscale:    ArrayNDT<single>
        Trn_X:          ArrayNDT<single>
        Trn_T:          ArrayNDT<single>
        Trn_Sigma:      ArrayNDT<single>
        }
    type predOutput = {
       In_Mean:         ArrayNDT<single>
       In_Cov:          ArrayNDT<single>
       Pred_Mean:       ArrayNDT<single>
       Pred_Cov:        ArrayNDT<single>
       }

    let testMultiGPLayer device =

        let seed = 1
        let rand = Random(seed)

        let ngps = 3
        let ntraining = 10
        let ntest = 10

        let m: ExprT<single> = Expr.var "m" [SizeSpec.symbol "nTstSmpls";SizeSpec.symbol "nGPs"; SizeSpec.symbol "nGPs"]
        let psd = m.* m
        let cmplr = DevHost.Compiler, CompileEnv.empty
        let makePsd = Func.make cmplr psd |> arg1 m

        let mb = ModelBuilder<single> "Test"

        let nSmpls    = mb.Size "nSmpls"
        let nGPs      = mb.Size "nGPs"
        let nTrnSmpls = mb.Size "nTrnSmpls"
        
        let mgp = 
            MultiGPLayer.pars (mb.Module "MGP") {NGPs=nGPs; NTrnSmpls=nTrnSmpls}
        let inp_mean  : ExprT<single> = mb.Var "inp_mean"  [nSmpls; nGPs]
        let inp_cov   : ExprT<single> = mb.Var "inp_cov"   [nSmpls; nGPs; nGPs]

        mb.SetSize nGPs      ngps
        mb.SetSize nTrnSmpls ntraining
        let mi = mb.Instantiate device

        let pred_mean, pred_cov = MultiGPLayer.pred mgp inp_mean inp_cov
        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg2 inp_mean inp_cov

        let ls_host = [1.0f; 1.5f; 2.0f] |> ArrayNDHost.ofList 
        let trn_x_host =  rand.UniformArrayND (-5.0f ,5.0f) [ngps;ntraining] 
        let trn_t_host = rand.UniformArrayND (-5.0f ,5.0f) [ngps;ntraining] 
        let trn_sigma_host = ArrayNDHost.zeros<single> [ngps;ntraining]

        let trainInp = {
            Lengthscale = ls_host;
            Trn_X = trn_x_host;
            Trn_T = trn_t_host;
            Trn_Sigma = trn_sigma_host}
        let trainData = [trainInp] |> Dataset.FromSamples
        trainData.Save("TrainData.h5")

        let ls_val = ls_host |> post device
        let trn_x_val = trn_x_host  |> post device
        let trn_t_val = trn_t_host  |> post device
        let trn_sigma_val = trn_sigma_host  |> post device

        mi.ParameterStorage.[!mgp.Lengthscales] <- ls_val
        mi.ParameterStorage.[!mgp.TrnX] <- trn_x_val
        mi.ParameterStorage.[!mgp.TrnT] <- trn_t_val
        mi.ParameterStorage.[!mgp.TrnSigma] <- trn_sigma_val

        let randomTest () =
            let inp_meanhost = rand.UniformArrayND (-5.0f ,5.0f) [1;ngps]
            let inp_mean_val = inp_meanhost |> post device

            let inp_covhost =  rand.UniformArrayND (0.0f ,2.0f) [1;ngps;ngps]
            let inp_covhost = makePsd inp_covhost
            let inp_cov_val = inp_covhost |> post device


        let pred_mean, pred_cov = pred_mean_cov_fn inp_mean_val inp_cov_val

            let testInOut = {
                In_Mean = inp_meanhost;
                In_Cov = inp_covhost;
                Pred_Mean = pred_mean;
                Pred_Cov = pred_cov}

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

            testInOut
        
        let testList = [1..ntest]
                       |> List.map (fun _-> randomTest () )



        let testData = testList |> Dataset.FromSamples
        testData.Save("TestData.h5")
    [<EntryPoint>]
    let main argv = 
        TestUtils.evalHostCuda testMultiGPLayer
        //TestUtils.compareTraces testMultiGPLayer false |> ignore
        //testMultiGPLayer DevCuda |> ignore
        //testMultiGPLayer DevHost |> ignore

        0


