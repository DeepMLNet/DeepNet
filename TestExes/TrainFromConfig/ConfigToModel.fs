namespace TrainFromConfig

open FSharp.Configuration
open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System.Text.RegularExpressions
open GPTransfer

[<AutoOpen>]
module ConfigToModel =
    ///Reading the config file
    type TrainConfig = YamlConfig<"DefaultConfig.yaml">
    ///Loads the dataset defined in the config file.
    let dataFromConfig (config:TrainConfig) =
        ///TODO furtehr options in configuration file
        let pars = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = config.Data.TargetCols |> List.ofSeq}
        let fullData = CsvLoader.loadFile pars config.Data.Path
        let fullDataset = Dataset.FromSamples fullData
        let splitDataHost = TrnValTst.Of(fullDataset)
        match config.Training.Device with
        | Cuda -> splitDataHost.ToCuda()
        | Host -> splitDataHost
    
    ///Instanciates a ModelBuilder on the IDevice defined in config
    let instantiateFromConfig (config:TrainConfig) (mb:ModelBuilder<single>) =
        match config.Training.Device with
        | Cuda -> mb.Instantiate DevCuda
        | Host->  mb.Instantiate DevHost
    
    ///calls the loss measure named in the config file
    let lossFromConfig (config:TrainConfig) =
        match config.Training.Loss with
        | "MSE" -> LossLayer.MSE
        | "CrossEntropy" ->  LossLayer.CrossEntropy
        | "BinaryCrossEntropy" -> LossLayer.BinaryCrossEntropy
        | _ -> failwithf "Loss meassure %s is not implemented" config.Training.Loss
    
    let transferFuncFromConfig (config:TrainConfig) idx =
        match config.Model.Hyperpars.Neural.[idx].TransferFunc with
        | "Tanh"        -> NeuralLayer.Tanh
        | "SoftMax"     -> NeuralLayer.SoftMax
        | "Identity"    -> NeuralLayer.Identity
        | _ -> failwithf "transfer function %s is not implemented" config.Model.Hyperpars.Neural.[idx].TransferFunc


    ///Builds the MultiGPTransferUnit Model from the config files
    ///returns prediction, loss and ModelInstance
    let mlgptFromConfig (config:TrainConfig) (data:TrnValTst<CsvLoader.CsvSample>)= 
        let mb = ModelBuilder<single> config.Model.Name
        let nBatch  = mb.Size "nBatch"
        let nLayers = config.Model.Hyperpars.GPTransfer.Count
        
        
       ///define symbolc sizes for the layers
        let mutable symbolicSizes:list<GPTransferUnit.HyperPars> = List.Empty
        let first:GPTransferUnit.HyperPars = {NInput = mb.Size "nInput"; NOutput = mb.Size "nHidden1"; NTrnSmpls = mb.Size "nTrn0"}
        let last:GPTransferUnit.HyperPars = {NInput = mb.Size (sprintf "nHidden%d" (nLayers-1));
                                            NOutput = mb.Size "nOutput";
                                            NTrnSmpls = mb.Size (sprintf "nTrn%d" (nLayers-1))}
        if nLayers = 1 then
            symbolicSizes <- [{NInput = mb.Size "nInput"; NOutput = mb.Size "nOutput"; NTrnSmpls = mb.Size "nTrn0"}]
        else if nLayers = 2 then
            symbolicSizes <- [first;last]
        else
            let middle: list<GPTransferUnit.HyperPars> = [1..nLayers-2] 
                                                         |> List.map (fun i -> 
                                                                {NInput = mb.Size (sprintf "nHidden%d" i);
                                                                 NOutput = mb.Size (sprintf "nHidden%d" (i+1));
                                                                 NTrnSmpls = mb.Size (sprintf "nTrn%d" i)})
            symbolicSizes <- [first]|> List.append symbolicSizes|> List.append [last]
        ///define symbolc sizes for the layers

        let mlmgp = 
            MLGPT.pars (mb.Module config.Model.Type) 
                { Layers = symbolicSizes
                  LossMeasure = lossFromConfig config }

        // define variables
        let input  = mb.Var "Input"  [nBatch; first.NInput]
        let target = mb.Var "Target" [nBatch; last.NOutput]

        ///compute input and output size from data set
        let inSize = data.Trn.[0].Input.Shape.[0]
        let outSize = data.Trn.[0].Target.Shape.[0]
        config.Model.Hyperpars.GPTransfer.[0].NInput <- inSize
        config.Model.Hyperpars.GPTransfer.[nLayers - 1].NOutput <- outSize
        config.Save(__SOURCE_DIRECTORY__ + @"\RuntimeConfig.yaml")
        ///set all model sizes
        let configSizes =  config.Model.Hyperpars.GPTransfer |> List.ofSeq
        let setSizes =   List.map2 (fun  ({NInput = nIn; NOutput = nOut;NTrnSmpls = nTrn}:GPTransferUnit.HyperPars)
                                         (hyperpars:TrainConfig.Model_Type.Hyperpars_Type.GPTransfer_Item_Type) ->
                                        mb.SetSize nIn hyperpars.NInput
                                        mb.SetSize nOut hyperpars.NOutput
                                        mb.SetSize nTrn hyperpars.NTrnSmpls)
                                    symbolicSizes 
                                    configSizes
        let pred,_ = MLGPT.pred mlmgp input
        let mi = instantiateFromConfig config mb
        let loss = MLGPT.loss mlmgp input target
        pred, loss, mi,input,target

    ///Builds the Multilayer Perceptron from the config files
    ///returns prediction, loss and ModelInstance
    let mlpFromConfig (config:TrainConfig) (data:TrnValTst<CsvLoader.CsvSample>) = 
        let mb = ModelBuilder<single> config.Model.Name
        let nBatch  = mb.Size "nBatch"
        let nLayers = config.Model.Hyperpars.GPTransfer.Count
        
        ///define symbolc sizes for the layers
        let mutable symbolicSizes:list<NeuralLayer.HyperPars> = List.Empty
        let first:NeuralLayer.HyperPars = {NInput = mb.Size "nInput"; NOutput = mb.Size "nHidden1"; TransferFunc = transferFuncFromConfig config 0}
        let last:NeuralLayer.HyperPars = {NInput = mb.Size (sprintf "nHidden%d" (nLayers-1));
                                          NOutput = mb.Size "nOutput";
                                          TransferFunc = transferFuncFromConfig config (nLayers-1)}
        if nLayers = 1 then
            symbolicSizes <- [{NInput = mb.Size "nInput"; NOutput = mb.Size "nOutput"; TransferFunc = transferFuncFromConfig config 0}]
        else if nLayers = 2 then
            symbolicSizes <- [first;last]
        else
            let middle:  list<NeuralLayer.HyperPars> = [1..nLayers-2] 
                                                            |> List.map (fun i -> 
                                                                {NInput = mb.Size (sprintf "nHidden%d" i);
                                                                 NOutput = mb.Size (sprintf "nHidden%d" (i+1));
                                                                 TransferFunc = transferFuncFromConfig config i})
            symbolicSizes <- [first]|> List.append symbolicSizes|> List.append [last]

        let mlp = 
            MLP.pars (mb.Module config.Model.Type) 
                { Layers = symbolicSizes
                  LossMeasure = lossFromConfig config }

        // define variables
        let input  = mb.Var "Input"  [nBatch; first.NInput]
        let target = mb.Var "Target" [nBatch; last.NOutput]

        ///compute input and output size from data set
        let inSize = data.Trn.[0].Input.Shape.[0]
        let outSize = data.Trn.[0].Target.Shape.[0]
        config.Model.Hyperpars.GPTransfer.[0].NInput <- inSize
        config.Model.Hyperpars.GPTransfer.[nLayers - 1].NOutput <- outSize
        config.Save(__SOURCE_DIRECTORY__ + @"\RuntimeConfig.yaml")
        ///set all model sizes
        let configSizes =  config.Model.Hyperpars.GPTransfer |> List.ofSeq
        let setSizes =   List.map2 (fun  ({NInput = nIn; NOutput = nOut;TransferFunc = _}:NeuralLayer.HyperPars)
                                         (hyperpars:TrainConfig.Model_Type.Hyperpars_Type.GPTransfer_Item_Type) ->
                                        mb.SetSize nIn hyperpars.NInput
                                        mb.SetSize nOut hyperpars.NOutput)
                                    symbolicSizes 
                                    configSizes

        let pred = MLP.pred mlp input.T
        let mi = instantiateFromConfig config mb
        let loss = MLP.loss mlp input.T target.T
        pred, loss, mi,input,target

    let modelFromConfig (config:TrainConfig) (data:TrnValTst<CsvLoader.CsvSample>) = 
        let model = config.Model
        match model.Type with
        | "MLGPTransfer" -> mlgptFromConfig config data
        | "MLPerceptron" -> mlpFromConfig config data
        | _ -> failwithf "Model type %s is not defined" model.Type
    
    ///generates the optimizer from the config file
    let optFromConfig (config:TrainConfig) loss pars=
        match config.Training.Opt.Optimizer with
        ///TODO optimizer config from config file
        | "Adam" -> 
            let opt =Adam<single> (loss,pars,DevCuda)
            let optcfg = opt.DefaultCfg
            opt, optcfg        
//        | "GradientDescent" -> 
//            let opt = GradientDescent<single>(loss,pars,DevCuda)
//            let optCfg = { Optimizers.GradientDescent.Step=1e-3f }
//            opt,optCfg
        ///TODO: compatibility with GD
        | _-> failwithf "Optimizer %s is not implemented" config.Training.Opt.Optimizer
    
    let trainConfigFromConfig (config:TrainConfig) =
        let trainCfg: Train.Cfg =  {Seed= config.Training.Config.Seed
                                    BatchSize = config.Training.Config.BatchSize
                                    LossRecordInterval = config.Training.Config.LossRecordInterval
                                    ///TODO:  from config
                                    Termination = Train.ItersWithoutImprovement 100
                                    MinImprovement = config.Training.Config.MinImprovement
                                    TargetLoss = inStringToFloatOption config.Training.Config.TargetLoss
                                    MinIters = inStringToIntOption config.Training.Config.MinIters
                                    MaxIters = inStringToIntOption config.Training.Config.MaxIters
                                    LearningRates =   config.Training.Config.LearningRates |> List.ofSeq
                                    CheckpointDir = inStringToStringOption config.Training.Config.CheckpointDir
                                    DiscardCheckpoint =  config.Training.Config.DiscardCheckpoint
                                    DumpPrefix = inStringToStringOption config.Training.Config.DumpPrefix
                                    }
        trainCfg

