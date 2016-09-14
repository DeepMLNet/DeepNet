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
    

    let classificationGPTransferUnit()=
        printfn "Trainingone GPTransfer Unit on abalone dataset age classification"
        
        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let pars = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = [8]}
        let fullData = CsvLoader.loadFile pars "abalone.data" 
        let fullDataset = Dataset.FromSamples fullData
        let data = TrnValTst.Of(fullDataset).ToCuda()
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


        mb.SetSize nInput (fullDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullDataset.[0].Target |> ArrayND.nElems)
        mb.SetSize nTrn 20

        let mi = mb.Instantiate dev
        let pred, _ = GPTransferUnit.pred gptu (InputLayer.transform input)
        //loss expression
        let loss = LossLayer.loss LossLayer.CrossEntropy pred target

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
            }

        let result = Train.train trainable data trainCfg
        ()



    let classificationMLMGP()=
        printfn "Training 2 Layer MLMGP on abalone dataset age classification"

        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let pars = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = [8]}
        let fullData = CsvLoader.loadFile pars "abalone.data" 
        let fullDataset = Dataset.FromSamples fullData
        
        let data = TrnValTst.Of(fullDataset).ToCuda()
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
                            {NInput=nInput; NGPs=nClass; NTrnSmpls=nTrn}]
                  LossMeasure = LossLayer.CrossEntropy }
                // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]

        mb.SetSize nInput (fullDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullDataset.[0].Target |> ArrayND.nElems)
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
            }

        let result = Train.train trainable data trainCfg
        ()

    let classificationMLP()=
//        printfn "Training 2 Layer MLP on letterRecognition dataset"
//        let fullDataset = (DataParser.loadSingleDataset "letter-recognition.data.txt" [0] ',')
        
        printfn "Training 2 Layer MLP on abalone dataset age classification"

        let dev = DevCuda

        ///Load the ablone dataset, classify gender from data
        let pars = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = [8]}
        let fullData = CsvLoader.loadFile pars "abalone.data" 
        let fullDataset = Dataset.FromSamples fullData
        let data = TrnValTst.Of(fullDataset).ToCuda()
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
        mb.SetSize nInput (fullDataset.[0].Input |> ArrayND.nElems)
        mb.SetSize nClass (fullDataset.[0].Target |> ArrayND.nElems)
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
            }

        let result = Train.train trainable data trainCfg


        ()
 
    [<EntryPoint>]
    let main argv = 

//        TestFunctions.testDatasetParser()

//        classificationMLP ()
//        classificationGPTransferUnit ()
//        classificationMLMGP ()

        TestFunctions.testMultiGPLayer DevHost
//        TestFunctions.testMultiGPLayer DevCuda
            
        TestFunctions.TestGPTransferUnit DevHost
//        TestFunctions.TestGPTransferUnit DevCuda


//        TestUtils.evalHostCuda TestFunctions.testMultiGPLayer
//        TestUtils.compareTraces TestFunctions.testMultiGPLayer false |> ignore




        0
//

