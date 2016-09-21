namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System
open Datasets
open Models
open Optimizers
open Models

module Program =
    
    let parsClass = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = [8]}
    let fullClassificationData = CsvLoader.loadFile parsClass "abalone.txt" 
    let fullClassificationDataset = Dataset.FromSamples fullClassificationData
    let classificationData = TrnValTst.Of(fullClassificationDataset).ToCuda()
//
//    let fullRegressionDataset = fullClassificationDataset
//    let regressionData = classificationData
//
    let parsReg = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = [8];IntTreatment=CsvLoader.IntAsNumerical}
    let fullRegressionData = CsvLoader.loadFile parsReg "abalone.txt" 
    let fullRegressionDataset = Dataset.FromSamples fullRegressionData
    let regressionData = TrnValTst.Of(fullRegressionDataset).ToCuda()


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
                  NGPs = nClass
                  NTrnSmpls = nTrn}
                // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]


        mb.SetSize nInput (fullClassificationDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullClassificationDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nTrn 20


        let mi = mb.Instantiate dev
        let pred, _ = GPTransferUnit.pred gptu (InputLayer.transform input)

        let softmax act = exp act / Expr.sumKeepingAxis 1 (exp act)
        
        //let pred = softmax pred
//        let loss = -target * log pred |> Expr.sumAxis 0 |> Expr.mean
//        let loss = loss |> Expr.dump "loss"
        
        //let pred = max pred (Expr.scalar 1e-3f)
        let pred = pred**2.0f + 1e-3f

        // loss expression
        let loss = LossLayer.loss LossLayer.CrossEntropy pred.T target.T
        let loss = loss |> Expr.checkFinite "loss"

        // optimizer
        let opt = Adam (loss, mi.ParameterVector, dev)
        let optCfg = opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg : Train.Cfg = {    
            Seed               = 100   
            BatchSize          = 500 
            //BatchSize          = 10
            LossRecordInterval = 1                                   
            Termination        = Train.ItersWithoutImprovement 100
            MinImprovement     = 1e-7  
            TargetLoss         = None  
            MinIters           = Some 100 
            MaxIters           = None  
            LearningRates      = [1e-3; 1e-4; 1e-5; 1e-6]                               
            CheckpointDir      = None  
            DiscardCheckpoint  = false 
            DumpPrefix         = None
            }
//
        let result = Train.train trainable data trainCfg
        ()



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
                { Layers = [{NInput=nInput; NGPs=nHidden; NTrnSmpls=nTrn}
                            {NInput=nHidden; NGPs=nClass; NTrnSmpls=nTrn}]
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

        // optimizer
        let opt =  Adam (loss, mi.ParameterVector, DevCuda)
        let optCfg =opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg : Train.Cfg = {    
            Seed               = 100   
            BatchSize          = 500 
            LossRecordInterval = 10                                   
            Termination        = Train.ItersWithoutImprovement 100
            MinImprovement     = 1e-7  
            TargetLoss         = None  
            MinIters           = Some 100 
            MaxIters           = None  
            LearningRates      = [1e-3; 1e-4; 1e-5]                               
            CheckpointDir      = None  
            DiscardCheckpoint  = false 
            DumpPrefix         = None
            }
        let result = Train.train trainable data trainCfg


        ()

    let classificationMLP()=
//        printfn "Training 2 Layer MLP on letterRecognition dataset"
//        let fullDataset = (DataParser.loadSingleDataset "letter-recognition.data.txt" [0] ',')
        
        printfn "Training 2 Layer MLP on abalone dataset age classification"

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
                { Layers = [{NInput=nInput; NOutput=nHidden; TransferFunc=NeuralLayer.Tanh}
                            {NInput=nHidden; NOutput=nClass; TransferFunc=NeuralLayer.SoftMax}]
                  LossMeasure = LossLayer.CrossEntropy }
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

        // optimizer
        let opt =  Adam (loss, mi.ParameterVector, DevCuda)
        let optCfg =opt.DefaultCfg

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg : Train.Cfg = {    
            Seed               = 100   
            BatchSize          = 500 
            LossRecordInterval = 10                                   
            Termination        = Train.ItersWithoutImprovement 100
            MinImprovement     = 1e-7  
            TargetLoss         = None  
            MinIters           = Some 100 
            MaxIters           = None  
            LearningRates      = [1e-3; 1e-4; 1e-5]                               
            CheckpointDir      = None  
            DiscardCheckpoint  = false 
            DumpPrefix         = None
            }

        let result = Train.train trainable data trainCfg

        ()
    
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
                  NGPs = nClass
                  NTrnSmpls = nTrn}
                // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]


        mb.SetSize nInput (fullRegressionDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullRegressionDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nTrn 20

        printfn "nInput=\n%A" (fullRegressionDataset.[0].Input |> ArrayND.nElems)
        printfn "nClass=\n%A" (fullRegressionDataset.[0].Target |> ArrayND.nElems)
        printfn "mb=\n%A" mb.PrettyString

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

        let trainCfg : Train.Cfg = {    
            Seed               = 100   
            BatchSize          = 500 
            //BatchSize          = 10
            LossRecordInterval = 10                                   
            Termination        = Train.ItersWithoutImprovement 100
            MinImprovement     = 1e-7  
            TargetLoss         = None  
            MinIters           = Some 100 
            MaxIters           = None  
            LearningRates      = [1e-3; 1e-4; 1e-5]                               
            CheckpointDir      = None  
            DiscardCheckpoint  = false 
            DumpPrefix         = None
            }

        let result = Train.train trainable data trainCfg
        ()

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

        let trainCfg : Train.Cfg = {    
            Seed               = 100   
            BatchSize          = 500 
            LossRecordInterval = 10                                   
            Termination        = Train.ItersWithoutImprovement 100
            MinImprovement     = 1e-7  
            TargetLoss         = None  
            MinIters           = Some 100 
            MaxIters           = None  
            LearningRates      = [1e-3; 1e-4; 1e-5]                               
            CheckpointDir      = None  
            DiscardCheckpoint  = false 
            DumpPrefix         = None
            }

        let result = Train.train trainable data trainCfg


        ()
    [<EntryPoint>]
    let main argv = 

        SymTensor.Compiler.Cuda.Debug.Timing <- true
//        SymTensor.Compiler.Cuda.Debug.TraceCalls <- true
        SymTensor.Compiler.Cuda.Debug.TraceCompile <- true
//        SymTensor.Compiler.Cuda.Debug.DebugCompile <- true
        SymTensor.Compiler.Cuda.Debug.MemUsage <- true
//        SymTensor.Compiler.Cuda.Debug.DisableStreams <- true
//        SymTensor.Compiler.Cuda.Debug.DumpCode <- true

        //let trc = SymTensor.Trace.startSession "trace"

//        TestFunctions.testDatasetParser()

//        regressionMLP ()
//        regressionGPTransferUnit ()

//        Dump.start "gptraindump.h5"
        classificationGPTransferUnit ()
//        Dump.stop()
//        classificationMLMGP ()
//        classificationMLP ()

//        TestFunctions.testMultiGPLayer DevHost
//        TestFunctions.testMultiGPLayer DevCuda
            
//        TestFunctions.TestGPTransferUnit DevHost
//        TestFunctions.TestGPTransferUnit DevCuda


//        TestUtils.evalHostCuda TestFunctions.testMultiGPLayer
//        TestUtils.compareTraces TestFunctions.testMultiGPLayer false |> ignore


        //let tr = trc.End()
        0


