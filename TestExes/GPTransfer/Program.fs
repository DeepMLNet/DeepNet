namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System
open Datasets
open Models
open Optimizers
open Models
open Basics

module Program =
    
    ///Creates the datasets
    let parsClass = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = [8]}
    let fullClassificationData = CsvLoader.loadFile parsClass "abalone.txt" 
    let fullClassificationDataset = Dataset.FromSamples fullClassificationData
    let classificationDataHost = TrnValTst.Of(fullClassificationDataset)
    let classificationData = classificationDataHost.ToCuda()


    let parsReg = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = [7;8];IntTreatment=CsvLoader.IntAsNumerical}
    let fullRegressionData = CsvLoader.loadFile parsReg "abalone.txt" 
    let fullRegressionDataset = Dataset.FromSamples fullRegressionData
    let regressionData = TrnValTst.Of(fullRegressionDataset).ToCuda()

    ///Retruns the index of the maximum element in an ArrayND
    let maxPosition (inAry: ArrayNDHostT<single>) =
        let maxElem = inAry |> ArrayND.maxAxis 0
        let maxElem = Seq.head (ArrayND.allElems maxElem)
        let pos = ArrayND.allElems inAry |> Seq.findIndex (fun elem -> elem = maxElem)
        ArrayNDHost.scalar (single pos)

    ///Retruns the index of the maximum element alone one axis od an ArrayND
    let maxPositionAxis dim inAry= 
        ArrayND.axisReduce maxPosition dim inAry
    let aryToHost (ary: ArrayNDT<single>) =
        match ary with
        | :?  ArrayNDCudaT<single> as predCuda -> 
            predCuda.ToHost ()
        | :? ArrayNDHostT<single> as predHost -> 
            predHost
        | _ -> failwith "Array neither on Host nor on Cuda"

    let broadcastWithZeros size (ary:ArrayNDT<single>) = 
        let inSize = ary.Shape.[0]
        if inSize < size then
            let newShape = ary.Shape |> List.set 0 size
            let tempAry = ary |> ArrayND.newCOfSameType newShape
            tempAry.[0 .. inSize-1, Fill] <-  ary.[0 .. inSize-1, Fill]
            tempAry
        else ary

    ///Calculates the number of errors in one batch             
    let batchClassificationErrors batchSize (modelPred: ArrayNDT<single> -> ArrayNDT<single>) (input:ArrayNDT<single>) (target:ArrayNDT<single>) =
        let inSize = input.Shape.[0]
        if inSize < batchSize then
            let input = broadcastWithZeros batchSize input
            let target = broadcastWithZeros batchSize target
            let predAry = (modelPred input)
            predAry.[inSize..batchSize - 1,Fill] <- target.[inSize..batchSize - 1,Fill]

            let pred = aryToHost predAry
            let targ = aryToHost target
            let predClass = maxPositionAxis 1 pred
            let targetClass = maxPositionAxis 1 targ
            let errors = ArrayND.map2TC (fun a b -> if a = b then 0.0f else 1.0f) predClass targetClass |> ArrayND.sum
            let err = Seq.head (ArrayND.allElems errors)
            err
        else
            let predAry = (modelPred input)

            let pred = aryToHost predAry
            let targ = aryToHost target
            let predClass = maxPositionAxis 1 pred
            let targetClass = maxPositionAxis 1 targ
            let errors = ArrayND.map2TC (fun a b -> if a = b then 0.0f else 1.0f) predClass targetClass |> ArrayND.sum
            let err = Seq.head (ArrayND.allElems errors)
            err

    ///Calculates the number of errors in one dataset
    let setClassificationErrors batchSize (modelPred: ArrayNDT<single> -> ArrayNDT<single>) (inSeq: seq<CsvLoader.CsvSample>) =
        let errs =  inSeq
                    |>Seq.map (fun {Input = inp;Target = trg} -> 
                        batchClassificationErrors batchSize modelPred inp  trg )
                    |>Seq.fold (fun acc elem -> acc + elem) 0.0f
        errs

    ///Calculates the fraction of errors for train-, validation- and test-dataset
    let classificationErrors batchSize (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single>) =
        let nTrnSmpls = dataset.Trn.NSamples
        let nValSmpls = dataset.Val.NSamples
        let nTstSmpls = dataset.Tst.NSamples
        let trnBatches = dataset.Trn.Batches batchSize 
        let valBatches = dataset.Val.Batches batchSize 
        let tstBatches = dataset.Tst.Batches batchSize 
        let trnError = setClassificationErrors batchSize modelPred trnBatches  / (single nTrnSmpls)
        let valError = setClassificationErrors batchSize modelPred valBatches  / (single nValSmpls)                
        let tstError = setClassificationErrors batchSize modelPred tstBatches  / (single nTstSmpls)
        trnError,valError,tstError

    ///classification on abalone dataset using a single GPTransfer Unit
    let classificationGPTransferUnit()=
        printfn "Training one GPTransfer Unit on abalone dataset age classification"
        
        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let data = classificationData

        ///classified the dataset using a MLP with one hidden layer
        ///(analogous to Lern Mnist Project)
        let mb = ModelBuilder<single> "MultiGPModel"

        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nClass  = mb.Size "nClass"
        let nTrn = mb.Size "nTrn"

        let gptu = 
            GPTransferUnit.pars (mb.Module "GPTUP") 
                { NInput = nInput
                  NOutput = nClass
                  NTrnSmpls = nTrn}
                // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]


        mb.SetSize nInput (fullClassificationDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullClassificationDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nTrn 5

        printfn "nInput=%d  nClass=%d  nTrn=%d"
            (mb.GetSize nInput) (mb.GetSize nClass) (mb.GetSize nTrn)

        let mi = mb.Instantiate dev
        let pred,_ = GPTransferUnit.pred gptu (InputLayer.transform input)

        let softmax act = exp act / (Expr.sumKeepingAxis 1 (exp act) + 1e-3f)
        
//        let pred = softmax pred + 1e-3f
        let pred = pred|> Expr.dump "pred"
        let pred = pred |> Expr.checkFinite "pred"
//        let loss = -target * log pred |> Expr.sumAxis 0 |> Expr.mean
//        let loss = loss |> Expr.dump "loss"
        
        //let pred = max pred (Expr.scalar 1e-3f)
//        let pred = pred**2.0f + 1e-3f

        let pred_fun =  mi.Func pred |> arg1 input 

        // loss expression
        let loss = LossLayer.loss LossLayer.MSE pred.T target.T
        let loss = loss |> Expr.checkFinite "loss"
        let loss = loss |> Expr.dump "loss"
        // optimizer
        let opt = Adam (loss, mi.ParameterVector, dev)
        let optCfg = opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg
        
        let batchSize = 500

        let trainCfg = {Train.defaultCfg with   BatchSize          = batchSize
                                                Termination        = Train.ItersWithoutImprovement 100
                                                DumpPrefix         = None
                                                MaxIters           = Some 300
                                                }
        //let trnErr,valErr,tstErr = classificationErrors  batchSize data pred_fun
        //printfn"Classification errors before training:"
        //printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)
        let result = Train.train trainable data trainCfg
        //printfn "Training Time: %A" sw.Elapsed
        

        let trnErr,valErr,tstErr = classificationErrors  batchSize data pred_fun
        printfn"Classification errors after training:"
        printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)
        ()


    ///classification on abalone dataset using a network of GPTransfer Units
    let classificationMLMGP()=
        printfn "Training 2 Layer MLMGP on abalone dataset age classification"

        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let data = classificationData

        ///classified the dataset using a MLP with one hidden layer
        ///(analogous to Lern Mnist Project)
        let mb = ModelBuilder<single> "MultiGPModel"

        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nClass  = mb.Size "nClass"
        let nHidden = mb.Size "nHidden"
        let nTrn = mb.Size "nTrn"

        let mlmgp = 
            MLGPT.pars (mb.Module "MLMGP") 
                { Layers = [{NInput=nInput; NOutput=nHidden; NTrnSmpls=nTrn}
                            {NInput=nHidden; NOutput=nClass; NTrnSmpls=nTrn}]
                  LossMeasure = LossLayer.CrossEntropy }
                // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]

        mb.SetSize nInput (fullClassificationDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullClassificationDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nHidden 10
        mb.SetSize nTrn 20

        let mi = mb.Instantiate dev

        // loss expression
        let loss = MLGPT.loss mlmgp input target

        let pred,_ = MLGPT.pred mlmgp input
        let pred_fun =  mi.Func pred |> arg1 input 

        // optimizer
        let opt =  Adam (loss, mi.ParameterVector, DevCuda)
        let optCfg =opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target
        
        let batchSize = 500
        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg : Train.Cfg = {Train.defaultCfg with   BatchSize          = batchSize
                                                            Termination        = Train.ItersWithoutImprovement 100
                                                            }
        let trnErr,valErr,tstErr = classificationErrors  batchSize data pred_fun
        printfn"Classification errors before training:"
        printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)
                
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let result = Train.train trainable data trainCfg
        sw.Stop()
        printfn "Training Time: %A" sw.Elapsed

        let trnErr,valErr,tstErr = classificationErrors  batchSize data pred_fun
        printfn"Classification errors after training:"
        printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)

        ()
    
    ///classification on abalone dataset using a multilayer perceptron
    let classificationMLP()=
//        printfn "Training 2 Layer MLP on letterRecognition dataset"
//        let fullDataset = (DataParser.loadSingleDataset "letter-recognition.data.txt" [0] ',')
        
        printfn "Training 1 Layer MLP on abalone dataset age classification"

        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let data = classificationData
        ///classified the dataset using a MLP with one hidden layer
        ///(analogous to Lern Mnist Project)
        let mb = ModelBuilder<single> "NeuralNetModel"

        // define symbolic sizes
        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nClass  = mb.Size "nClass"
        let nHidden = mb.Size "nHidden"
       
       // define model parameters
        let mlp = 
            MLP.pars (mb.Module "MLP") 
                { Layers = [
//                            {NInput=nInput; NOutput=nHidden; TransferFunc=NeuralLayer.Tanh}
                            {NInput=nInput; NOutput=nClass; TransferFunc=NeuralLayer.SoftMax}]
                  LossMeasure = LossLayer.MSE }
        // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]


        // instantiate model
        mb.SetSize nInput (fullClassificationDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullClassificationDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nHidden 10
        let mi = mb.Instantiate dev
        // loss expression
        let loss = MLP.loss mlp input.T target.T
        
        let pred = MLP.pred mlp input.T
        let pred_fun =  mi.Func pred.T |> arg1 input 
        
        // optimizer
        let opt =  Adam (loss, mi.ParameterVector, DevCuda)
        let optCfg =opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg
        let batchSize = 500
        let trainCfg= {Train.defaultCfg with   BatchSize          = batchSize
                                               Termination        = Train.ItersWithoutImprovement 100
                                               MaxIters           = Some 300
//                                               DumpPrefix         = Some "MLP"
                                               }
        let trnErr,valErr,tstErr = classificationErrors  batchSize data pred_fun
        printfn"Classification errors before training:"
        printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)
                
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let result = Train.train trainable data trainCfg
        result.Save "result.json"
        printfn "Training Time: %A" sw.Elapsed

        let trnErr,valErr,tstErr = classificationErrors  batchSize data pred_fun
        printfn"Classification errors after training:"
        printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)
        ()
    
    ///regression on abalone dataset using a single GPTransfer Unit
    let regressionGPTransferUnit()=
        printfn "Training one GPTransfer Unit on abalone dataset age regression"
        
        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let data = regressionData

        ///classified the dataset using a MLP with one hidden layer
        ///(analogous to Lern Mnist Project)
        let mb = ModelBuilder<single> "GPTUModel"

        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nClass  = mb.Size "nClass"
        let nTrn = mb.Size "nTrn"

        let gptu = 
            GPTransferUnit.pars (mb.Module "GPTUP") 
                { NInput = nInput
                  NOutput = nClass
                  NTrnSmpls = nTrn}
                // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]


        mb.SetSize nInput (fullRegressionDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullRegressionDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nTrn 20

        printfn "shapeInputs = %A" (fullRegressionDataset.[0].Input |> ArrayND.shape)
        printfn "shapeTargets = %A" (fullRegressionDataset.[0].Target |> ArrayND.shape)
        let mi = mb.Instantiate dev
        let pred, _ = GPTransferUnit.pred gptu (InputLayer.transform input)

        //loss expression
        let loss = LossLayer.loss LossLayer.MSE pred.T target.T

        // optimizer
        let opt =  Adam (loss, mi.ParameterVector, dev)
        let optCfg =opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg = {Train.defaultCfg with   BatchSize          = 500
                                                Termination        = Train.ItersWithoutImprovement 100
                                                DumpPrefix         = None}

        let result = Train.train trainable data trainCfg
        result.Save "result.json"
        
        ()
    
    ///regressionn on abalone dataset using a multilayer perceptron
    let regressionMLP()=
//        printfn "Training 2 Layer MLP on letterRecognition dataset"
//        let fullDataset = (DataParser.loadSingleDataset "letter-recognition.data.txt" [0] ',')
        
        printfn "Training 2 Layer MLP on abalone dataset age regrression"

        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let data = regressionData
        ///classified the dataset using a MLP with one hidden layer
        ///(analogous to Lern Mnist Project)
        let mb = ModelBuilder<single> "NeuralNetModel"

        // define symbolic sizes
        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nClass  = mb.Size "nClass"
        let nHidden = mb.Size "nHidden"
       
       // define model parameters
        let mlp = 
            MLP.pars (mb.Module "MLP") 
                { Layers = [{NInput=nInput; NOutput=nHidden; TransferFunc=NeuralLayer.Tanh}
                            {NInput=nHidden; NOutput=nClass; TransferFunc=NeuralLayer.Identity}]
                  LossMeasure = LossLayer.MSE }
        // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]

        // instantiate model
        mb.SetSize nInput (fullRegressionDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullRegressionDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nHidden 4
        let mi = mb.Instantiate dev
        
        // loss expression
        let loss = MLP.loss mlp input.T target.T

        // optimizer
        let opt =  Adam (loss, mi.ParameterVector, DevCuda)
        let optCfg =opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg = {Train.defaultCfg with   BatchSize          = 500
                                                Termination        = Train.ItersWithoutImprovement 100
                                                DumpPrefix         = None}

        let result = Train.train trainable data trainCfg
        result.Save "result.json"

        ()
    [<EntryPoint>]
    let main argv = 

//        SymTensor.Debug.Timing <- true
//        SymTensor.Debug.TraceCompile <- true
        SymTensor.Debug.EnableCheckFinite <- false
//        SymTensor.Compiler.Cuda.Debug.Timing <- true
//        SymTensor.Compiler.Cuda.Debug.TraceCalls <- true
//        SymTensor.Compiler.Cuda.Debug.TraceCompile <- true
//        SymTensor.Compiler.Cuda.Debug.DebugCompile <- true
//        SymTensor.Compiler.Cuda.Debug.ResourceUsage <- true
        SymTensor.Compiler.Cuda.Debug.DisableStreams <- true
        SymTensor.Compiler.Cuda.Debug.TerminateWhenNonFinite <- false
        SymTensor.Compiler.Cuda.Debug.DumpCode <- true
//        SymTensor.Compiler.Cuda.Debug.TerminateAfterRecipeGeneration <- true

        //let trc = SymTensor.Trace.startSession "trace"

//        TestFunctions.testDatasetParser()

//        regressionMLP ()
//        regressionGPTransferUnit ()

//        Dump.start "gptraindump.h5"
//        Dump.prefix <- sprintf "pre"
//        classificationMLP ()
        classificationGPTransferUnit ()
//        classificationMLMGP ()
//        Dump.stop()
//        TestFunctions.testMultiGPLayer DevHost
//        TestFunctions.testMultiGPLayer DevCuda
            
//        TestFunctions.TestGPTransferUnit DevHost
//        TestFunctions.TestGPTransferUnit DevCuda


//        TestUtils.evalHostCuda TestFunctions.testMultiGPLayer
//        TestUtils.compareTraces TestFunctions.testMultiGPLayer false |> ignore


        //let tr = trc.End()
        Cuda.CudaSup.shutdown ()
        0


