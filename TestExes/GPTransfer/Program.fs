namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System
open Datasets

module Program =
    
    ///Transfers Arrays to device (either Host or DevCuda)
    let post device (x: ArrayNDT<'T>) =
        if device = DevCuda then ArrayNDCuda.toDev (x :?> ArrayNDHostT<'T>) :> ArrayNDT<'T>
        else x 
    
    ///Sampling type for Model training parameters.
    type trainData ={
        Lengthscale:    ArrayNDT<single>
        Trn_X:          ArrayNDT<single>
        Trn_T:          ArrayNDT<single>
        Trn_Sigma:      ArrayNDT<single>
        }

    ///Sampling type for Model input and prediction.
    type predOutput = {
       In_Mean:         ArrayNDT<single>
       In_Cov:          ArrayNDT<single>
       Pred_Mean:       ArrayNDT<single>
       Pred_Cov:        ArrayNDT<single>
       }
    
    ///Generates a random list of singles that is sorted.
    let randomSortedListOfLength (rand:Random) (minValue,maxValue) length =
        [1..length] |> List.map (fun _ -> rand.NextDouble())
        |> List.map (fun x -> (single x))
        |> List.map (fun x -> x  * (maxValue - minValue) + minValue)
        |> List.sort

    ///Generates multiple random sorted lists of singles in a 2D list
    let randomSortedLists (rand:Random) (minValue,maxValue) length = 
        List.map (fun _ -> randomSortedListOfLength rand (minValue,maxValue) length)
    
    let fsng x = single x
    let isng x = single x

    ///Generates a random single in the range (minValue,maxValue)
    let randSingleInRange (rand:Random) (minValue,maxValue) =
        let (imin,imax) = int minValue,int maxValue
        if (imin >= imax - 1) then
            failwith "range is too small"
        let rand1 = fsng (rand.NextDouble())
        let rand2 = isng(rand.Next(imin,imax-1))
        rand1+rand2

    ///Generates a random polynomial of maximal power 2
    let randPolynomial (rand:Random) list = 

        let fact1 = fsng (rand.NextDouble())
        let fact2 = fsng (rand.NextDouble())
        let pow1 = isng (rand.Next(1,2))
        let pow2 = isng (rand.Next(1,2))
        list |> List.map (fun x ->   fact1 *x** pow1 - fact2 *x** pow2)
   
   ///Turns a random matrix in the form of a covariance matrix into a Psd matrix. 
    let fPsd c =        
        let cov: ExprT<single> = Expr.var "cov" [SizeSpec.broadcastable;SizeSpec.symbol "nGPs"; SizeSpec.symbol "nGPs"]
        let psd = cov.* cov
        let cmplr = DevHost.Compiler, CompileEnv.empty
        let funPsd = Func.make cmplr psd |> arg1 cov
        funPsd c



    ///Tests multilayer GPs with random parameters and random inputs.
    ///Saves parameters and inputs in hdf5 files to compare with other implementations (especially gpsample.py).
    let testMultiGPLayer device =

        //initiating random number generator 
        let rand = Random()
        //defining size parameters
        let ngps = 3
        let ntraining = 30
        let ntest = 2


        //building the model
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

        //model outputs
        let pred_mean, pred_cov = MultiGPLayer.pred mgp inp_mean inp_cov
        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg2 inp_mean inp_cov


        //creating random training vectors
        let trn_x_list = [1..ngps] |> randomSortedLists rand (-5.0f,5.0f) ntraining 
        let trn_x_host = trn_x_list |> ArrayNDHost.ofList2D

        let trn_t_list = trn_x_list |> List.map(fun list -> randPolynomial rand list)
        let trn_t_host = trn_t_list |> ArrayNDHost.ofList2D

        printfn "Trn_x =\n%A" trn_x_host
        printfn"Trn_t =\n%A" trn_t_host

        ////lengthscale vectore hardcoded
        //let ls_host = [1.0f; 1.5f; 2.0f] |> ArrayNDHost.ofList 

        //random lengthscale vector
        let ls_host = [1..ngps] |> List.map (fun x -> randSingleInRange rand (0.0, 2.0)) |> ArrayNDHost.ofList
        
        //sigma vector hardcoded
        let trn_sigma_host = ArrayNDHost.zeros<single> [ngps;ntraining]


        //save train parameters
        let trainInp = {
            Lengthscale = ls_host;
            Trn_X = trn_x_host;
            Trn_T = trn_t_host;
            Trn_Sigma = trn_sigma_host}

        let trainData = [trainInp] |> Dataset.FromSamples
        let trainFileName = sprintf "TrainData.h5"
        trainData.Save(trainFileName)

        //transfer train parametters to device (Host or GPU)
        let ls_val = ls_host |> post device
        let trn_x_val = trn_x_host  |> post device
        let trn_t_val = trn_t_host  |> post device
        let trn_sigma_val = trn_sigma_host  |> post device

        mi.ParameterStorage.[!mgp.Lengthscales] <- ls_val
        mi.ParameterStorage.[!mgp.TrnX] <- trn_x_val
        mi.ParameterStorage.[!mgp.TrnT] <- trn_t_val
        mi.ParameterStorage.[!mgp.TrnSigma] <- trn_sigma_val


        ///run GpTransferModel with random test inputs
        let randomTest () =

            //generate random test inputs
            let inp_meanhost = rand.UniformArrayND (-5.0f ,5.0f) [1;ngps]
            let inp_mean_val = inp_meanhost |> post device

//            let inp_covhost =  rand.UniformArrayND (0.0f ,2.0f) [1;ngps;ngps]
//            let inp_covhost = fPsd inp_covhost
            let inp_covhost = ArrayNDHost.zeros<single> [1;ngps;ngps]
            let inp_cov_val = inp_covhost |> post device
            
            //calculate predicted mean and variance
            let pred_mean, pred_cov = pred_mean_cov_fn inp_mean_val inp_cov_val
            
            //save inputs and predictions in sample datatype
            let testInOut = {
                In_Mean = inp_meanhost;
                In_Cov = inp_covhost;
                Pred_Mean = pred_mean;
                Pred_Cov = pred_cov}

            //print inputs and predictions
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

            //return sample of inputs and predictions
            testInOut
        
        //run ntest tests and save samples in dataset
        printfn "Testing Multi GP Transfer Model"
        let testList = [1..ntest]
                       |> List.map (fun _-> randomTest () )



        let testData = testList |> Dataset.FromSamples
        let testFileName = sprintf "TestData.h5"
        testData.Save(testFileName)

    [<EntryPoint>]
    let main argv = 
        //testMultiGPLayer DevHost
        TestUtils.evalHostCuda testMultiGPLayer
        //TestUtils.compareTraces testMultiGPLayer false |> ignore
        //testMultiGPLayer DevCuda |> ignore
        //testMultiGPLayer DevHost |> ignore

        0


