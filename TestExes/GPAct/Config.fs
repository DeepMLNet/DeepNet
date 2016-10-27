namespace GPAct

open System.IO
open Nessos.FsPickler.Json

open Basics
open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open MLPlots


[<AutoOpen>]
/// configuration types
module ConfigTypes =
    
    /// Supported layer types.
    type Layer = 
        | NeuralLayer of NeuralLayer.HyperPars
        | GPActivationLayer of GPActivationLayer.HyperPars
        | MeanOnlyGPLayer of MeanOnlyGPLayer.HyperPars

    /// Configuration of the feed forward model.
    type FeedForwardModel = {
        /// list of layers from input to output
        Layers:     Layer list
        /// the loss meassure
        Loss:       LossLayer.Measures
    }

    /// Configuration of the dataset.
    type CsvDataCfg = {
        /// path of the dataset
        Path:       string
        /// CsvLoader parameters to load the dataset
        Parameters: CsvLoader.Parameters
    }

    /// Supported optimizer types.
    type Optimizer =
        | GradientDescent of GradientDescent.Cfg<single>
        | Adam of Adam.Cfg<single>
    
    /// Configuration of the whole experiment.
    type Cfg = {
        Model:                  FeedForwardModel
        Data:                   CsvDataCfg
        Optimizer:              Optimizer
        Training:               Train.Cfg
        SaveParsDuringTraining: bool
        PlotGPsDuringTraining:  bool
    }


/// model building from configuration file
module ConfigLoader =
    
    /// Placeholder to initialize mutable sizeSpecs. 
    let notLoading () : SizeSpecT = failwith "not loading a config file"

    let mutable NInput = notLoading
    let mutable NOutput = notLoading

    /// Builds a trainable model from an F# configuration script file.
    /// The configuration script must assign the cfg variable of type ConfigTypes.Cfg.
    let buildModel cfgPath =
        
        // create model builder
        let mb = ModelBuilder<single> "Model"
        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nOutput = mb.Size "nOutput"

        let input  = mb.Var<single> "Input"  [nBatch; nInput]
        let target = mb.Var<single> "Target" [nBatch; nOutput]

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
        let fullData = 
            CsvLoader.loadFile cfg.Data.Parameters cfg.Data.Path
            |> Seq.shuffle 100
        let fullDataset = Dataset.ofSamples fullData
        let dataset = fullDataset |> TrnValTst.ofDataset |> TrnValTst.toCuda
        
        // build model
        /// Map containing parameters of all meanOnlyGPLayers of the model (used for plotting)
        let mutable meanOnlyGPLayers = Map.empty
        /// Map containing parameters of all GPActivationLayers of the model (used for plotting)
        let mutable gpLayers = Map.empty
        let predMean, predVar, regularizationTerm = 
            ((input, GPUtils.covZero input, Expr.zeroOfSameType input), List.indexed cfg.Model.Layers)
            ||> Seq.fold (fun (mean, var, reg) (layerIdx, layer) ->
                match layer with
                | NeuralLayer hp ->
                    let pars = NeuralLayer.pars (mb.Module (sprintf "NeuralLayer%d" layerIdx)) hp
                    NeuralLayer.pred pars mean, GPUtils.covZero mean,reg + NeuralLayer.regularizationTerm pars // TODO: implement variance prop
                | GPActivationLayer hp ->
                    let name = sprintf "GPTransferLayer%d" layerIdx
                    let pars = GPActivationLayer.pars (mb.Module name) hp
                    gpLayers <- gpLayers |> Map.add name pars
                    let predMean, predVar, regGP = GPActivationLayer.pred pars (mean, var)
                    predMean, predVar, reg + regGP
                | MeanOnlyGPLayer hp ->
                    let name = (sprintf "MeanOnlyGPLayer%d" layerIdx)
                    let pars = MeanOnlyGPLayer.pars (mb.Module name) hp
                    meanOnlyGPLayers <- meanOnlyGPLayers |> Map.add name pars
                    MeanOnlyGPLayer.pred pars mean, GPUtils.covZero mean, reg + MeanOnlyGPLayer.regularizationTerm pars
                )

        // build loss
        let loss = (LossLayer.loss cfg.Model.Loss predMean target) +
                   regularizationTerm

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

        
        /// True if plot thread is already running.
        /// Due to RProvider only one plot thread can run at a time.
        let mutable plotInProgress = false
        /// Saves Parameters and plots GP activation functions if
        /// corresponding bools are true.
        let lossRecordFn (state: TrainingLog.Entry) =
            if cfg.SaveParsDuringTraining then
                let filename = sprintf "Pars%05d.h5" state.Iter
                mi.SavePars filename
            // plot GP activation functions
            if cfg.PlotGPsDuringTraining && state.Iter % 200 = 0 then
                let gpLayers = gpLayers |> Map.map (fun name pars -> 
                    let gpPars = pars.Activation
                    let l = mi.[gpPars.Lengthscales] |> ArrayND.copy
                    let s = mi.[gpPars.TrnSigma] |> ArrayND.copy
                    let x = mi.[gpPars.TrnX] |> ArrayND.copy
                    let t = mi.[gpPars.TrnT] |> ArrayND.copy
                    let meanFct = (fun x -> Expr.zerosLike x)
                    let cut = pars.HyperPars.Activation.CutOutsideRange
                    l, s, x, t, meanFct,cut)
                let moGPLayers  = meanOnlyGPLayers |> Map.map (fun name pars -> 
                    let l = mi.[pars.Lengthscales] |> ArrayND.copy
                    let s = mi.[pars.TrnSigma] |> ArrayND.copy
                    let x = mi.[pars.TrnX] |> ArrayND.copy
                    let t = mi.[pars.TrnT] |> ArrayND.copy
                    let meanFct = pars.HyperPars.MeanFunction
                    let cut = pars.HyperPars.CutOutsideRange
                    l, s, x, t, meanFct,cut)
                let join (p:Map<'a,'b>) (q:Map<'a,'b>) = 
                    Map(Seq.concat [ (Map.toSeq p) ; (Map.toSeq q) ])
                let gpLayers = join gpLayers moGPLayers       
                let plots = async {
                    Cuda.CudaSup.setContext ()
                    for KeyValue (name, (l, s, x, t, meanFct,cut)) in gpLayers do
                        let plots = [0..l.Shape.[0] - 1] |> List.map (fun gp ->
                            let ls = l.[gp] |> ArrayND.value
                            let hps =  {GaussianProcess.Kernel = GaussianProcess.SquaredExponential (ls,1.0f)
                                        GaussianProcess.MeanFunction = meanFct
                                        GaussianProcess.CutOutsideRange = cut}
                            let name = sprintf "node %d" gp
                            let plot = fun () ->
                                            GPPlots.Plots.simplePlot (hps, 
                                                                s.[gp, *],
                                                                x.[gp, *],
                                                                t.[gp, *],
                                                                200, -5.0f, 5.0f, -5.0f, 5.0f)
                            name,plot)
                        savePlot 1200 900 "." (sprintf "%s-%05d.pdf" name state.Iter) (fun () ->
                            plotgrid plots
                            ) 
                    plotInProgress <- false
                }
                if not plotInProgress then
                    plotInProgress <- true
                    Async.Start plots
        
        // For classification return Some (unit -> unit) function to print classification error
        // otherwise return None.
        let errorPrint = 
            if (cfg.Model.Loss = LossLayer.CrossEntropy) || (cfg.Model.Loss = LossLayer.BinaryCrossEntropy) then 
                let printFn () = ClassificationError.printErrors cfg.Training.BatchSize dataset predFn
                Some printFn
            else
                None
        // build training function
        let trainCfg = {cfg.Training with LossRecordFunc = lossRecordFn}        
        let trainFn () = 
            Train.train trainable dataset trainCfg

        mi, predFn, trainFn, errorPrint


