namespace GPAct

open System.IO
open Nessos.FsPickler.Json

open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda


[<AutoOpen>]
/// configuration types
module ConfigTypes =

    type Layer = 
        | NeuralLayer of NeuralLayer.HyperPars
        | GPActivationLayer of GPActivationLayer.HyperPars

    type FeedForwardModel = {
        Layers:     Layer list
        Loss:       LossLayer.Measures
    }

    type CsvDataCfg = {
        Path:       string
        Parameters: CsvLoader.Parameters
    }

    type Optimizer =
        | GradientDescent of GradientDescent.Cfg<single>
        | Adam of Adam.Cfg<single>

    type Cfg = {
        Model:                  FeedForwardModel
        Data:                   CsvDataCfg
        Optimizer:              Optimizer
        Training:               Train.Cfg
        SaveParsDuringTraining: bool
    }


/// model building from configuration file
module ConfigLoader =

    let notLoading () : SizeSpecT = failwith "not loading a config file"

    let mutable NInput = notLoading
    let mutable NOutput = notLoading

    /// Builds a trainable model from an F# configuration script file.
    /// The configuration script must assign the cfg variable of type ConfigTypes.Cfg.
    let buildModel cfgPath =

        let mb = ModelBuilder<single> "Model"
        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nOutput = mb.Size "nOutput"

        let input  = mb.Var "Input"  [nBatch; nInput]
        let target = mb.Var "Target" [nBatch; nOutput]

        // load config
        NInput <- (fun () -> nInput)
        NOutput <- (fun () -> nOutput)
        let cfgPath = Path.GetFullPath cfgPath
        let cfg : Cfg = Config.loadAndChdir cfgPath

        // dump config as JSON
        let json = FsPickler.CreateJsonSerializer(indent=true, omitHeader=true)
        let cfgDumpPath = Path.ChangeExtension (cfgPath, "json")
        use cfgDump = File.CreateText cfgDumpPath
        json.Serialize (cfgDump, cfg)

        // load data
        let fullData = CsvLoader.loadFile cfg.Data.Parameters cfg.Data.Path
        let fullDataset = Dataset.FromSamples fullData
        let dataset = TrnValTst.Of fullDataset |> TrnValTst.ToCuda
        
        // build model
        let predMean, predVar = 
            ((input, GPUtils.covZero input), List.indexed cfg.Model.Layers)
            ||> Seq.fold (fun (mean, var) (layerIdx, layer) ->
                match layer with
                | NeuralLayer hp ->
                    let pars = NeuralLayer.pars (mb.Module (sprintf "NeuralLayer%d" layerIdx)) hp
                    NeuralLayer.pred pars mean, GPUtils.covZero mean // TODO: implement variance prop
                | GPActivationLayer hp ->
                    let pars = GPActivationLayer.pars (mb.Module (sprintf "GPTransferLayer%d" layerIdx)) hp
                    GPActivationLayer.pred pars (mean, var))

        // build loss
        let loss = LossLayer.loss cfg.Model.Loss predMean target

        // instantiate model
        let mi = mb.Instantiate (DevCuda, 
                                 Map [nInput,  fullDataset.[0].Input.NElems
                                      nOutput, fullDataset.[0].Target.NElems]) 

        // build functions
        let predFn : ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single> = 
            mi.Func (predMean, predVar) |> arg1 input

        // build optimizer
        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.ofSeq [input, smpl.Input; target, smpl.Target]
        let trainable =
            match cfg.Optimizer with
            | GradientDescent cfg -> 
                Train.trainableFromLossExpr mi loss smplVarEnv GradientDescent.New cfg
            | Adam cfg ->
                Train.trainableFromLossExpr mi loss smplVarEnv Adam.New cfg

        // build training function
        let mutable trainCfg = cfg.Training
        if cfg.SaveParsDuringTraining then
            let savePars (state: TrainingLog.Entry) =
                let filename = sprintf "Pars%05d.h5" state.Iter
                mi.SavePars filename
            trainCfg <- {trainCfg with LossRecordFunc = savePars}
        let trainFn () = 
            Train.train trainable dataset trainCfg

        mi, predFn, trainFn


