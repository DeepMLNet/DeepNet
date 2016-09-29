namespace GPAct

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System
open Datasets
open Models

module TestFunctions =
    
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

    ///Generates a random polynomial of maximal power 2
    let randPolynomial (rand:Random) list = 

        let fact1 = fsng (rand.NextDouble())
        let fact2 = fsng (rand.NextDouble())
        let pow1 = isng (rand.Next(1,2))
        let pow2 = isng (rand.Next(1,2))
        list |> List.map (fun x ->   fact1 *x** pow1 - fact2 *x** pow2)
   
   ///Turns a random matrix in the form of a covariance matrix into a Psd matrix. 
    let makePsd (c: ArrayNDT<_>) =        
        c.T .* c



    ///Tests multilayer GPs with random parameters and random inputs.
    ///Saves parameters and inputs in hdf5 files to compare with other implementations (especially gpsample.py).
    let testMultiGPLayer device =

        //initiating random number generator 
        let rand = Random(1)
        //defining size parameters
        let ngps = 2
        let ntraining = 10
        let ntest = 1

        //building the model
        let mb = ModelBuilder<single> "Test"

        let nSmpls    = mb.Size "nSmpls"
        let nGPs      = mb.Size "nGPs"
        let nTrnSmpls = mb.Size "nTrnSmpls"
        
        let w =
            WeightTransform.pars (mb.Module "WL") {WeightTransform.defaultHyperPars with
                                                    NInput = nGPs; NOutput = nGPs}

        let mgp = 
            GPActivation.pars (mb.Module "MGP") {GPActivation.defaultHyperPars with
                                                  NGPs=nGPs; NTrnSmpls=nTrnSmpls}
        let inp_mean = mb.Var "inp_mean"  [nSmpls; nGPs]
        let inp_cov  = mb.Var "inp_cov"   [nSmpls; nGPs; nGPs]
        mb.SetSize nGPs      ngps
        mb.SetSize nTrnSmpls ntraining
        let mi = mb.Instantiate device

        //model outputs
//        let pred_mean = MultiGPLayer.pred mgp (WeightLayer.transform w (inp_mean, inp_cov))
        let pred_mean,pred_cov = GPActivation.pred mgp (inp_mean, inp_cov)
//        let pred_mean= mi.Func pred_mean |> arg2 inp_mean inp_cov
        


//        let pred_mean, pred_cov = MultiGPLayer.pred mgp inp_mean inp_cov
        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg2 inp_mean inp_cov

        //creating random training vectors
        let trn_x_list = [1..ngps] |> randomSortedLists rand (-5.0f,5.0f) ntraining 
        let trn_x_host = trn_x_list |> ArrayNDHost.ofList2D

        let trn_t_list = trn_x_list |> List.map(fun list -> randPolynomial rand list)
        let trn_t_host = trn_t_list |> ArrayNDHost.ofList2D

        printfn "Trn_x =\n%A" trn_x_host
        printfn "Trn_t =\n%A" trn_t_host

        //lengthscale vectore hardcoded
//        let ls_host = [1.0f; 1.5f; 2.0f] |> ArrayNDHost.ofList 
//        //random lengthscale vector
        let ls_host = rand.UniformArrayND (0.0f,3.0f) [ngps]

        //sigma vector hardcoded
        let trn_sigma_host = (ArrayNDHost.ones<single> [ngps;ntraining]) * sqrt 0.1f


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

        mi.ParameterStorage.[mgp.Lengthscales] <- ls_val
        mi.ParameterStorage.[mgp.TrnX] <- trn_x_val
        mi.ParameterStorage.[mgp.TrnT] <- trn_t_val
        mi.ParameterStorage.[mgp.TrnSigma] <- trn_sigma_val

        let transMean,transCov = WeightTransform.transform w (inp_mean,inp_cov)
        let transTestFn1 =  mi.Func transMean |> arg2 inp_mean inp_cov
        let transTestFn2 =  mi.Func transCov  |> arg2 inp_mean inp_cov
        let initLMean,initLCov = inp_mean, GPUtils.covZero inp_mean
        let initTestFn1 =  mi.Func initLMean |> arg1 inp_mean
        let initTestFn2 =  mi.Func initLCov |> arg1 inp_mean
        ///run GpTransferModel with random test inputs
        let randomTest () =

            //generate random test inputs
            let inp_mean_host = rand.UniformArrayND (-5.0f ,5.0f) [1;ngps]
            let inp_mean_val = inp_mean_host |> post device

            let inp_cov_host = ArrayNDHost.zeros<single> [1;ngps;ngps]
            //let inp_cov_host = 0.1f * ArrayNDHost.ones [1;ngps] |> ArrayND.diagMat

            //let inp_cov_host = rand.UniformArrayND (-2.0f, 2.0f) [1;ngps;ngps]
            let inp_covhost = makePsd inp_cov_host

            let inp_cov_val = inp_covhost |> post device

            //calculate predicted mean and variance
            let pred_mean,pred_cov = pred_mean_cov_fn inp_mean_val inp_cov_val

            //save inputs and predictions in sample datatype
            let testInOut = {
                In_Mean = inp_mean_host;
                In_Cov = inp_covhost;
                Pred_Mean = pred_mean;
                Pred_Cov = pred_cov}

            //print inputs and predictions
            printfn "Lengthscales=\n%A" mi.ParameterStorage.[mgp.Lengthscales]
            printfn "TrnX=\n%A" mi.ParameterStorage.[mgp.TrnX]
            printfn "TrnT=\n%A" mi.ParameterStorage.[mgp.TrnT]
            printfn "TrnSigma=\n%A" mi.ParameterStorage.[mgp.TrnSigma]
            printfn ""
            printfn "inp_mean=\n%A" inp_mean_val
            printfn "inp_cov=\n%A" inp_cov_val
            printfn ""
            printfn "pred_mean=\n%A" pred_mean
            printfn "pred_cov=\n%A" pred_cov

            //return sample of inputs and predictions
            testInOut

        //run ntest tests and save samples in dataset
        Dump.start "dump.h5"
        printfn "Testing Multi GP Transfer Model on %A" device
        let testList = [1..ntest]
                       |> List.map (fun n-> 
                            Dump.prefix <- sprintf "%d" n
                            randomTest () )
        Dump.stop ()

        let testData = testList |> Dataset.FromSamples
        let testFileName = sprintf "TestData.h5"
        testData.Save(testFileName)
    
    let TestGPTransferUnit device =
        //initiating random number generator 
        let rand = Random(1)
        //defining size parameters
        let ngps = 1
        let ninputs = 5
        let ntraining = 10
        let ntests = 20
        let batchSize = 1

        //building the model
        let mb = ModelBuilder<single> "GPTU_Test"

        let nSmpls       = mb.Size "nSmpls"
        let nInputs      = mb.Size "nInputs"
        let nGPs         = mb.Size "nGPs"
        let nTrnSmpls    = mb.Size "nTrnSmpls"

        let gptu = 
           GPActivationLayer.pars (mb.Module "GPTU") 
                {WeightTransform = {WeightTransform.defaultHyperPars with NInput=nInputs; NOutput=nGPs}
                 Activation      = {GPActivation.defaultHyperPars with NGPs=nGPs; NTrnSmpls=nTrnSmpls}}

        let inp_mean = mb.Var "inp_mean"  [nSmpls; nInputs]
        let pred     = mb.Var "Pred"      [nSmpls; nGPs]
        let target   = mb.Var "Target"    [nSmpls; nGPs]


        mb.SetSize  nGPs         ngps
        mb.SetSize  nTrnSmpls    ntraining
        mb.SetSize  nInputs      ninputs

        let mi = mb.Instantiate device

        let pred_mean, pred_cov = GPActivationLayer.pred gptu (inp_mean, GPUtils.covZero inp_mean)
        let pred_mean_cov_fn = mi.Func (pred_mean, pred_cov) |> arg1 inp_mean

//        let loss =  -target * log pred |> Expr.sumAxis 0 |> Expr.mean
//        let loss = loss |> Expr.dump "Loss"
//        let cmplr = DevCuda.Compiler, CompileEnv.empty
//        let loss_fn = Func.make cmplr loss |> arg2 pred_mean target

//        let dLoss = Deriv.compute loss |> Deriv.ofVar mi.ParameterVector  |> Expr.reshape (Expr.shapeOf mi.ParameterVector) 
//        let dLoss = dLoss |> Expr.dump "dLoss"
//        let dLoss_fn = mi.Func dLoss |> arg2 pred_mean target

        let randomTest () =

            //generate random test inputs
            let inp_mean_host = rand.UniformArrayND (-5.0f ,10.0f) [batchSize;ninputs]
            let inp_mean_val = inp_mean_host |> post device


            //calculate predicted mean and variance
            let pred_mean,pred_cov = pred_mean_cov_fn inp_mean_val
            let randOffset = rand.UniformArrayND (-0.2f ,0.2f) [batchSize;ngps] |> post device
            let target_val = pred_mean + randOffset
            //print inputs and predictions

//            let l = loss_fn pred_mean target_val
//            let dL = dLoss_fn pred_mean tar

            printfn "inp_mean=\n%A" inp_mean_val
            printfn ""
            printfn "pred_mean=\n%A" pred_mean
            printfn "pred_cov=\n%A" pred_cov
            printfn ""
//            printfn "loss=\n%A" l
//            printfn ""
//            printfn "dLoss=\n%A" dL
//            printfn ""
            //return sample of inputs and predictions
        Dump.start "gptudump.h5"
        let testList = [1..ntests]
                       |> List.map (fun n-> 
                            Dump.prefix <- sprintf "%d" n
                            randomTest () )
        Dump.stop ()
        ()