// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace GPTransfer
open FSharp.Configuration
open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda

module ConfigTraining = 
    type TrainConfig = YamlConfig<"Config.yaml">

    type Hyperpars = {NInput:       int;
                      NGPs:         int;
                      NTrnSmpls:    int}

    ///Loads the dataset defined in the config file.
    let dataFromConfig (config:TrainConfig) =
        ///TODO furtehr options in configuration file
        let pars = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = config.Data.TargetCols |> List.ofSeq}
        let fullData = CsvLoader.loadFile pars config.Data.Path
        let fullDataset = Dataset.FromSamples fullData
        let splitDataHost = TrnValTst.Of(fullDataset)
        match config.Training.Device with
        | "DevCuda" -> splitDataHost.ToCuda()
        | _ -> splitDataHost
    
    ///Instanciates a ModelBuilder on the IDevice defined in config
    let instantiateFromConfig (config:TrainConfig) (mb:ModelBuilder<single>) =
        match config.Training.Device with
        | "DevCuda" -> mb.Instantiate DevCuda
        | _ ->  mb.Instantiate DevHost
    
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


    ///Builds the GPTransfer unit defined by the config file
    ///returns prediction, loss and ModelInstance
    let gptFromConfig (config:TrainConfig) (data:TrnValTst<CsvLoader.CsvSample>) = 
        let mb = ModelBuilder<single> config.Model.Name
        
        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nClass  = mb.Size "nClass"
        let nTrn = mb.Size "nTrn"
        
        let gptu = 
            GPTransferUnit.pars (mb.Module config.Model.Type) 
                { NInput = nInput
                  NOutput = nClass
                  NTrnSmpls = nTrn}

        let input  = mb.Var "Input"  [nBatch; nInput]
        let target = mb.Var "Target" [nBatch; nClass]

        let inSize = data.Trn.[0].Input.Shape.[0]
        let outSize = data.Trn.[0].Target.Shape.[0]
        let ntrn = config.Model.Hyperpars.GPTransfer.[0].NTrnSmpls

        let nLayers = config.Model.Hyperpars.GPTransfer.Count
        config.Model.Hyperpars.GPTransfer.[0].NInput <- inSize
        config.Model.Hyperpars.GPTransfer.[nLayers - 1].NOutput <- outSize

        mb.SetSize nInput inSize
        mb.SetSize nClass outSize
        mb.SetSize nTrn ntrn
        let pred,_ = GPTransferUnit.pred gptu (InputLayer.transform input)
        let mi = instantiateFromConfig config mb
        let lossMeasure = lossFromConfig config
        let loss = LossLayer.loss lossMeasure pred.T target.T
        pred,loss , mi,input,target


    ///Builds the MultiGPTransferUnit Model from the config files
    ///returns prediction, loss and ModelInstance
    let mlgptFromConfig (config:TrainConfig) (data:TrnValTst<CsvLoader.CsvSample>)= 
        let mb = ModelBuilder<single> config.Model.Name
        let nBatch  = mb.Size "nBatch"
        let nLayers = config.Model.Hyperpars.GPTransfer.Count
        ///define symbolc sizes for the layers
        let symbolicSizes: list<GPTransferUnit.HyperPars> = [1..nLayers-2] 
                                                            |> List.map (fun i -> 
                                                                {NInput = mb.Size (sprintf "nHidden%d" i);
                                                                 NOutput = mb.Size (sprintf "nHidden%d" (i+1));
                                                                 NTrnSmpls = mb.Size (sprintf "nTrn%d" i)})

        let first:GPTransferUnit.HyperPars = {NInput = mb.Size "nInput"; NOutput = mb.Size "nHidden1"; NTrnSmpls = mb.Size "nTrn0"}
        let last:GPTransferUnit.HyperPars = {NInput = mb.Size (sprintf "nHidden%d" (nLayers-1));
                                            NOutput = mb.Size "nOutput";
                                            NTrnSmpls = mb.Size (sprintf "nTrn%d" (nLayers-1))}
        let symbolicSizes = [first]|> List.append symbolicSizes|> List.append [last]

        let mlmgp = 
            MLGPT.pars (mb.Module config.Model.Type) 
                { Layers = symbolicSizes
                  LossMeasure = lossFromConfig config }

        // define variables
        let input  = mb.Var "Input"  [nBatch; first.NInput]
        let target = mb.Var "Target" [nBatch; last.NOutput]

        ///compute input and output size from data set
        let inSize = data.Trn.[0].Input.Shape.[1]
        let outSize = data.Trn.[0].Target.Shape.[1]
        config.Model.Hyperpars.GPTransfer.[0].NInput <- inSize
        config.Model.Hyperpars.GPTransfer.[nLayers - 1].NOutput <- outSize

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
        let symbolicSizes: list<NeuralLayer.HyperPars> = [1..nLayers-2] 
                                                            |> List.map (fun i -> 
                                                                {NInput = mb.Size (sprintf "nHidden%d" i);
                                                                 NOutput = mb.Size (sprintf "nHidden%d" (i+1));
                                                                 TransferFunc = transferFuncFromConfig config i})

        let first:NeuralLayer.HyperPars = {NInput = mb.Size "nInput"; NOutput = mb.Size "nHidden1"; TransferFunc = transferFuncFromConfig config 0}
        let last:NeuralLayer.HyperPars = {NInput = mb.Size (sprintf "nHidden%d" (nLayers-1));
                                          NOutput = mb.Size "nOutput";
                                          TransferFunc = transferFuncFromConfig config (nLayers-1)}
        let symbolicSizes = [first]|> List.append symbolicSizes|> List.append [last]

        let mlp = 
            MLP.pars (mb.Module config.Model.Type) 
                { Layers = symbolicSizes
                  LossMeasure = lossFromConfig config }

        // define variables
        let input  = mb.Var "Input"  [nBatch; first.NInput]
        let target = mb.Var "Target" [nBatch; last.NOutput]

        ///compute input and output size from data set
        let inSize = data.Trn.[0].Input.Shape.[1]
        let outSize = data.Trn.[0].Target.Shape.[1]
        config.Model.Hyperpars.GPTransfer.[0].NInput <- inSize
        config.Model.Hyperpars.GPTransfer.[nLayers - 1].NOutput <- outSize

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
        | "GPTransfer" -> gptFromConfig config data
        | "MLGPT" -> mlgptFromConfig config data
        | "MLP" -> mlpFromConfig config data
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
                                     ///TODO:  from config
                                    TargetLoss= None
                                     ///TODO:  from config
                                    MinIters = Some 100
                                     ///TODO:  from config
                                    MaxIters = None
                                    LearningRates =   config.Training.Config.LearningRates |> List.ofSeq
                                    ///TODO:  from config
                                    CheckpointDir = None
                                    DiscardCheckpoint =  config.Training.Config.DiscardCheckpoint
                                    ///TODO:  from config
                                    DumpPrefix = None
                                    }
        trainCfg


    [<EntryPoint>]
    let main argv = 
        let config = TrainConfig()
        let data = dataFromConfig config
        let pred,loss,mi,input,target = modelFromConfig config data

        let opt, optCfg = optFromConfig config loss mi.ParameterVector

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg = trainConfigFromConfig config

        let result = Train.train trainable data trainCfg
        result.Save "result.json"
        0 // return an integer exit code
