namespace TrainFromConfig

open System.IO

open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open GPTransfer


[<AutoOpen>]
/// configuration types
module ConfigTypes =

    type Layer = 
        | NeuralLayer of NeuralLayer.HyperPars
        | GPTransferLayer of GPTransferUnit.HyperPars

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
        Model:      FeedForwardModel
        Data:       CsvDataCfg
        Optimizer:  Optimizer
        Training:   Train.Cfg
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

        NInput <- (fun () -> nInput)
        NOutput <- (fun () -> nOutput)
        let cfg : Cfg = Config.loadAndChdir cfgPath

        // load data
        let fullData = CsvLoader.loadFile cfg.Data.Parameters cfg.Data.Path
        let fullDataset = Dataset.FromSamples fullData
        let dataset = TrnValTst.Of fullDataset |> TrnValTst.ToCuda
        
        // build model
        let predMean, predVar = 
            ((input, InputLayer.cov input), List.indexed cfg.Model.Layers)
            ||> Seq.fold (fun (mean, var) (layerIdx, layer) ->
                match layer with
                | NeuralLayer hp ->
                    let pars = NeuralLayer.pars (mb.Module (sprintf "NeuralLayer%d" layerIdx)) hp
                    NeuralLayer.pred pars mean, InputLayer.cov mean // TODO: implement variance prop
                | GPTransferLayer hp ->
                    let pars = GPTransferUnit.pars (mb.Module (sprintf "GPTransferLayer%d" layerIdx)) hp
                    GPTransferUnit.pred pars (mean, var))

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
        let trainFn () = 
            Train.train trainable dataset cfg.Training

        mi, predFn, trainFn


