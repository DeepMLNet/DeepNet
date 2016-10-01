namespace FracSigmoid

open System.IO

open Basics
open ArrayNDNS
open Argu
open Models
open Datasets
open Optimizers
open SymTensor
open SymTensor.Compiler.Cuda


type GenerateArgs =
    | [<Mandatory>] XMin of float
    | [<Mandatory>] XMax of float
    | [<Mandatory>] XPoints of int 
    | [<Mandatory>] NMin of float
    | [<Mandatory>] NMax of float
    | [<Mandatory>] NPoints of int
    | [<MainCommand; ExactlyOnce; Last; Mandatory>] Filename of filename:string
    with
    interface IArgParserTemplate with member s.Usage = "self-explaining"
            

type TrainArgs =
    | [<MainCommand; ExactlyOnce; Last; Mandatory>] CfgFile of filename:string
    with
    interface IArgParserTemplate with member s.Usage = "self-explaining"


type CLIArguments = 
    | [<CliPrefix(CliPrefix.None)>] Generate of ParseResults<GenerateArgs>
    | [<CliPrefix(CliPrefix.None)>] Train of ParseResults<TrainArgs>
    with
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Generate _ -> "generate interpolation table"
            | Train _ -> "train model using specified config file"



[<AutoOpen>]
module ConfigTypes =
    type Layer = 
        | NeuralLayer of NeuralLayer.HyperPars
        | TableLayer of TableLayer.HyperPars

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




module Program =

    let notLoading () : SizeSpecT = failwith "not loading a config file"

    let mutable NInput = notLoading
    let mutable NOutput = notLoading

    let buildModel cfgPath =

        let mb = ModelBuilder<single> "FracSigmoid"
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

        // load data
        let fullData = 
            CsvLoader.loadFile cfg.Data.Parameters cfg.Data.Path
            |> Seq.shuffle 100
        let fullDataset = Dataset.FromSamples fullData
        let dataset = TrnValTst.Of fullDataset |> TrnValTst.ToCuda
        
        // build model
        let mutable tblLayers = Map.empty
        let pred = 
            (input, List.indexed cfg.Model.Layers)
            ||> Seq.fold (fun value (layerIdx, layer) ->
                match layer with
                | NeuralLayer hp ->
                    let pars = NeuralLayer.pars (mb.Module (sprintf "NeuralLayer%d" layerIdx)) hp
                    NeuralLayer.pred pars value
                | TableLayer hp ->
                    let name = sprintf "TableLayer%d" layerIdx
                    printfn "Creating %s with %A" name hp
                    let pars = TableLayer.pars (mb.Module name) hp
                    tblLayers <- tblLayers |> Map.add name pars
                    TableLayer.pred pars value
                )

        // build loss
        let predLoss = LossLayer.loss cfg.Model.Loss pred target

        // prevent n from leaving range
        let fracLoss =
            tblLayers
            |> Map.toSeq
            |> Seq.map (fun (name, pars) ->
                let over = Expr.maxElemwise (Expr.scalar 0.0f) (abs pars.Frac - 1.01f)                
                let under = Expr.maxElemwise (Expr.scalar 0.0f) (0.10f - abs pars.Frac)
                Expr.sum over + Expr.sum under
            )
            |> Seq.fold (+) (Expr.scalar 0.0f)

        let fracLoss = 1000.0f * fracLoss
        let prmLoss = predLoss + fracLoss

        // instantiate model
        let mi = mb.Instantiate (DevCuda, 
                                 Map [nInput,  fullDataset.[0].Input.NElems
                                      nOutput, fullDataset.[0].Target.NElems]) 

        // build functions
        let predFn : ArrayNDT<single> -> ArrayNDT<single> = mi.Func (pred) |> arg1 input

        // build optimizer
        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.ofSeq [input, smpl.Input; target, smpl.Target]
        let trainable =
            match cfg.Optimizer with
            | GradientDescent cfg -> 
                Train.trainableFromLossExprs mi [prmLoss; predLoss; fracLoss] smplVarEnv GradientDescent.New cfg
            | Adam cfg ->
                Train.trainableFromLossExprs mi [prmLoss; predLoss; fracLoss] smplVarEnv Adam.New cfg

        let lossRecordFn (state: TrainingLog.Entry) =
            if cfg.SaveParsDuringTraining then
                let filename = sprintf "Pars%05d.h5" state.Iter
                mi.SavePars filename
            
        // build training function
        let trainCfg = {cfg.Training with LossRecordFunc = lossRecordFn}        
        let trainFn () = 
            Train.train trainable dataset trainCfg

        mi, predFn, trainFn




    [<EntryPoint>]
    let main argv =
        let parser = ArgumentParser.Create<CLIArguments> (helpTextMessage="Fractional sigmoid network",
                                                          errorHandler = ProcessExiter())
        let results = parser.ParseCommandLine argv

        match results.GetSubCommand () with
        | Generate args ->
            ()
//            let info = {
//                NMin     = args.GetResult <@ NMin @>
//                NMax     = args.GetResult <@ NMax @>
//                NPoints  = args.GetResult <@ NPoints @>
//                XMin     = args.GetResult <@ XMin @>
//                XMax     = args.GetResult <@ XMax @>
//                XPoints  = args.GetResult <@ XPoints @>
//                Function = FracSigmoid
//            }         
//            printfn "Building FracExp interpolation table for\n%A" info
//            let tbl = FracSigmoidTable.generate info
//
//            // save the table
//            let path = (args.GetResult <@ Filename @>)
//            use hdf = HDF5.OpenWrite path
//            tbl |> FracSigmoidTable.save hdf "FracSigmoid"
//            printfn "Saved to %s" (Path.GetFullPath path)

        | Train args ->
            let cfgFile = args.GetResult <@ CfgFile @>
            let mi, predFn, trainFn = buildModel cfgFile
            let tr = trainFn ()
            printfn "%A" tr.Best
            printfn "Used config was %s" cfgFile

        0