namespace Models

open System
open System.Diagnostics
open System.IO
open MBrace.FsPickler.Json

open DeepNet.Utils
open Tensor
open Datasets
open SymTensor
open Optimizers


[<AutoOpen>]
module TrainingTypes = 

    /// Partition of the dataset.
    type Partition =
        /// training parition of the dataset
        | Training
        /// validation partition of the dataset
        | Validation

    /// User-defined quality metric values
    type UserQuality = {
        /// quality on training set
        TrnQuality: float
        /// quality on test set
        ValQuality: float
        /// quality on validation set
        TstQuality: float
    }

    /// User-defined quality metrics
    type UserQualities = Map<string, UserQuality>


/// Training history module.
module TrainingLog =

    type Entry = {
        Iter:               int
        TrnLoss:            float
        ValLoss:            float
        TstLoss:            float
        MultiTrnLoss:       float list
        MultiValLoss:       float list
        MultiTstLoss:       float list
        UserQualities:      UserQualities
        LearningRate:       float
    }

    type Log<'P> = {
        MinImprovement:     float
        BestOn:             Partition
        Best:               (Entry * Tensor<'P>) option
        History:            Entry list
    }

    let create minImprovement bestOn =
        {MinImprovement=minImprovement; BestOn=bestOn; Best=None; History=[]}

    let relevantLoss bestOn (entry: Entry) =
        match bestOn with
        | Training -> entry.TrnLoss
        | Validation -> entry.ValLoss

    let record (entry: Entry) parVals (log: Log<_>) =
        let best =
            match log.Best with
            | None -> Some (entry, parVals)
            | Some (bestEntry, _) when
                    (relevantLoss log.BestOn entry) 
                     <= (relevantLoss log.BestOn bestEntry) - log.MinImprovement ->
                Some (entry, Tensor.copy parVals)
            | _ -> log.Best
        {log with Best=best; History=entry :: log.History}

    let lastIter (log: Log<_>) =
        match log.History with
        | {Iter=iter}::_ -> iter
        | [] -> -1

    let bestIter (log: Log<_>) =
        match log.Best with
        | Some ({Iter=iter}, _) -> iter
        | None -> 0

    let best (log: Log<_>) =
        log.Best

    let itersWithoutImprovement (log: Log<_>) =
        match log.Best with
        | None -> 0
        | Some ({Iter=bestIter}, _) -> lastIter log - bestIter

    let removeToIter iter (log: Log<_>) =
        {log with History = log.History |> List.skipWhile (fun {Iter=i} -> i > iter)}


/// Generic training module.
module Train =

    /// Training termination criterium
    type TerminationCriterium =
        /// terminates after the given number of iterations with improvement of the validation loss
        | ItersWithoutImprovement of int
        /// trains for IterGain*BestIterSoFar iterations
        | IterGain of float
        /// does not use this termination criterium
        | Forever

    /// Training configuration.
    type Cfg = {
        /// seed for parameter initialization
        Seed:                           int
        /// batch size
        BatchSize:                      int64
        /// time slot length for sequence training
        SlotSize:                       int64 option
        /// number of iterations between evaluation of the loss
        LossRecordInterval:             int
        /// function that is called after loss has been evaluated
        LossRecordFunc:                 TrainingLog.Entry -> unit
        /// Function that takes the current iteration number and 
        /// calculates one or more user-defined quality metrics 
        /// using the current model state.
        UserQualityFunc:                int -> UserQualities
        /// termination criterium
        Termination:                    TerminationCriterium
        /// minimum loss decrease to count as improvement
        MinImprovement:                 float
        /// partition to use for determination of best loss
        BestOn:                         Partition
        /// target loss that should lead to termination of training
        TargetLoss:                     float option
        /// minimum training iterations
        MinIters:                       int option
        /// maximum training iterations
        MaxIters:                       int option
        /// Learning rates that will be used. After training terminates with
        /// one learning rate, it continues using the next learning rate from this list.
        LearningRates:                  float list
        /// Path to a checkpoint file (HDF5 format).
        /// Used to save the training state if training is interrupted and/or periodically.
        /// If the file exists, the training state is loaded from it and training resumes.
        /// The string %ITER% in the filename is replaced with the iteration number.
        CheckpointFile:                 string option
        /// number of iterations between automatic writing of checkpoints
        CheckpointInterval:             int option
        /// If true, checkpoint is not loaded from disk.
        DiscardCheckpoint:              bool
        /// If specified, loads the checkpoint corresponding to the specified iteration.
        /// Otherwise, the latest checkpoint is loaded.
        LoadCheckpointIter:             int option
        /// If false, no training is performed after loading the checkpoint.
        PerformTraining:                bool
        /// If set, during each iteration the dump prefix will be set to the given string
        /// concatenated with the iteration number.
        DumpPrefix:                     string option
    } 

    /// Default training configuration.
    let defaultCfg = {
        Seed                        = 1
        BatchSize                   = 10000L
        SlotSize                    = None
        LossRecordInterval          = 10
        LossRecordFunc              = fun _ -> ()
        UserQualityFunc             = fun _ -> Map.empty
        Termination                 = IterGain 1.25
        MinImprovement              = 1e-7
        BestOn                      = Validation
        TargetLoss                  = None
        MinIters                    = Some 100
        MaxIters                    = None
        LearningRates               = [1e-3; 1e-4; 1e-5; 1e-6]
        CheckpointFile              = None
        CheckpointInterval          = None
        DiscardCheckpoint           = false
        LoadCheckpointIter          = None
        PerformTraining             = true
        DumpPrefix                  = None
    }

    /// training faith
    type Faith =
        | Continue
        | NoImprovement
        | IterLimitReached
        | TargetLossReached
        | UserTerminated
        | CheckpointRequested
        | CheckpointIntervalReached
        | NaNEncountered

    /// Result of training
    type TrainingResult = {
        Best:               TrainingLog.Entry option
        TerminationReason:  Faith
        Duration:           TimeSpan
        History:            TrainingLog.Entry list
    } with
        /// save as JSON file
        member this.Save path = Json.save path this
        /// load from JSON file
        static member Load path : TrainingResult = Json.load path           


    /// Interface for a trainable model.
    type ITrainable<'Smpl, 'T> =
        /// Loss of given sample.
        abstract member Losses: sample:'Smpl -> float list
        /// Perform an optimization step with the given learning rate and sample and return the loss.
        abstract member Optimize: learningRate:float -> sample:'Smpl -> Lazy<float list>
        /// Prints information about the model.
        abstract member PrintInfo: unit -> unit
        /// Initializes the model using the given random seed.
        abstract member InitModel: seed:int -> unit
        /// Load model parameters from specified file.
        abstract member LoadModel: hdf:HDF5 -> prefix:string -> unit
        /// Save model parameters to specified file.
        abstract member SaveModel: hdf:HDF5 -> prefix:string -> unit
        /// Model parameter values (i.e. weights).
        abstract member ModelParameters: Tensor<'T> with get, set
        /// Resets the internal model state. (for example the latent state of an RNN)
        abstract member ResetModelState: unit -> unit
        /// Initialize optimizer state.
        abstract member InitOptState: unit -> unit
        /// Load optimizer state from specified file.
        abstract member LoadOptState: hdf:HDF5 -> prefix:string -> unit
        /// Save optimizer state to specified file.
        abstract member SaveOptState: hdf:HDF5 -> prefix:string -> unit

    /// Constructs an ITrainable<_> from expressions.
    let internal newTrainable
            (modelInstance: ModelInstance<'T>) 
            (losses: ExprT list) 
            (nextStateExpr: ExprT option)
            (varEnvBuilder: Tensor<'T> option -> 'Smpl -> VarEnvT)
            (optNew: ExprT -> ExprT -> IDevice -> IOptimizer<'T, 'OptCfg, 'OptState>)
            (optCfg: 'OptCfg) =         
   
        let usingState = Option.isSome nextStateExpr
        let mutable modelState = None

        let losses = losses |> List.map modelInstance.Use 
        let mainLoss = losses.Head
        let stateAndLosses = 
            if usingState then nextStateExpr.Value :: losses else losses
            
        let opt = optNew mainLoss modelInstance.ParameterVector modelInstance.Device       
        let mutable optState = opt.InitialState optCfg modelInstance.ParameterValues

        let updateStateAndGetLosses result = 
            match result with
            | nextState :: losses when usingState -> 
                modelState <- Some nextState
                losses
            | losses when not usingState -> losses
            | _ -> failwith "unexpected result"

        let lossesFn =
            let fn = modelInstance.Func stateAndLosses
            fun smpl ->                
                fn <| varEnvBuilder modelState smpl 
                |> updateStateAndGetLosses

        let lossesOptFn = 
            let fn = modelInstance.Func (opt.OptStepExpr :: stateAndLosses) |> opt.Use
            fun smpl optCfg optState ->
                fn <| varEnvBuilder modelState smpl <| optCfg <| optState
                |> List.tail |> updateStateAndGetLosses
   
        {new ITrainable<'Smpl, 'T> with
            member this.Losses sample = lossesFn sample |> List.map (Tensor.value >> conv<float>)
            member this.Optimize learningRate sample = 
                let losses = lossesOptFn sample (opt.CfgWithLearningRate learningRate optCfg) optState
                lazy (losses |> List.map (Tensor.value >> conv<float>))
            member this.PrintInfo () = modelInstance.ParameterStorage.PrintShapes ()
            member this.InitModel seed = modelInstance.InitPars seed
            member this.LoadModel hdf prefix = modelInstance.LoadPars (hdf, prefix)
            member this.SaveModel hdf prefix = modelInstance.SavePars (hdf, prefix)
            member this.ModelParameters
                with get () = modelInstance.ParameterValues
                and set (value) = modelInstance.ParameterValues <- value
            member this.ResetModelState () = modelState <- None
            member this.InitOptState () = optState <- opt.InitialState optCfg modelInstance.ParameterValues
            member this.LoadOptState hdf prefix = optState <- opt.LoadState hdf prefix
            member this.SaveOptState hdf prefix = opt.SaveState hdf prefix optState    
        }

    /// Constructs an ITrainable<_> for the given stateful model using the specified loss
    /// expressions, state update expression and optimizer.
    let newStatefulTrainable modelInstance losses nextState varEnvBuilder optNew optCfg =
        newTrainable modelInstance losses (Some nextState) varEnvBuilder optNew optCfg

    /// Constructs an ITrainable<_> for the given model instance, loss expressions and optimizer.
    let trainableFromLossExprs modelInstance losses varEnvBuilder optNew optCfg =
        newTrainable modelInstance losses None (fun _ -> varEnvBuilder) optNew optCfg

    /// Constructs an ITrainable<_> for the given model instance, loss expression and optimizer.
    let trainableFromLossExpr modelInstance loss varEnvBuilder optNew optCfg =     
        trainableFromLossExprs modelInstance [loss] varEnvBuilder optNew optCfg


    /// Current training state for checkpoints.
    type private TrainState = {
        History:            TrainingLog.Entry list
        BestEntry:          TrainingLog.Entry option
        LearningRates:      float list
        Duration:           TimeSpan
        Faith:              Faith
    }

    /// Trains a model instance using the given loss and optimization functions on the given dataset.
    /// Returns the training history.
    let train (trainable: ITrainable<'Smpl, 'T>) (dataset: TrnValTst<'Smpl>) (cfg: Cfg) =
        // checkpoint data
        let mutable checkpointRequested = false
        use ctrlCHandler = Console.CancelKeyPress.Subscribe (fun evt ->            
            match cfg.CheckpointFile with
            | Some _ when evt.SpecialKey = ConsoleSpecialKey.ControlBreak ->
                checkpointRequested <- true
                evt.Cancel <- true
            | _ -> ())
        let checkpointFilename iter = 
            match cfg.CheckpointFile with
            | Some cpBasename -> 
                match iter with
                | Some iter -> cpBasename.Replace("%ITER%", sprintf "%06d" iter)
                | None -> cpBasename.Replace("%ITER%", "latest")
            | None -> failwith "checkpointing is off"

        // batches
        let getBatches part = 
            match cfg.SlotSize with
            | Some slotSize -> part |> Dataset.slotBatches cfg.BatchSize slotSize
            | None -> part |> Dataset.batches cfg.BatchSize
        let trnBatches = getBatches dataset.Trn
        let valBatches = getBatches dataset.Val
        let tstBatches = getBatches dataset.Tst

        if Seq.isEmpty trnBatches then failwith "the training set is empty"
        if Seq.isEmpty valBatches then failwith "the validation set is empty"
        if Seq.isEmpty tstBatches then failwith "the test set is empty"

        // initialize model parameters
        printfn "Initializing model parameters for training"
        trainable.InitModel cfg.Seed
        trainable.InitOptState ()
        trainable.PrintInfo ()

        // dumping helpers
        let origDumpPrefix = Dump.prefix
        let setDumpPrefix iter partition =
            match cfg.DumpPrefix with
            | Some dp -> Dump.prefix <- sprintf "%s/%d/%s" dp iter partition
            | None -> ()            

        /// training function
        let rec doTrain iter learningRate log =

            if not Console.IsInputRedirected then printf "%6d \r" iter

            // execute training and calculate training loss
            trainable.ResetModelState ()
            setDumpPrefix iter "trn"
            let trnLosses = trnBatches |> Seq.map (trainable.Optimize learningRate) |> Seq.toList

            // record loss if needed
            let recordBecauseCP = 
                match cfg.CheckpointInterval with 
                | Some interval -> iter % interval = 0
                | None -> false
            if iter % cfg.LossRecordInterval = 0 || recordBecauseCP then

                // compute validation & test losses
                let multiAvg lls =
                    let mutable n = 1
                    lls
                    |> Seq.reduce (fun ll1 ll2 ->
                        n <- n + 1
                        List.zip ll1 ll2
                        |> List.map (fun (l1, l2) -> l1 + l2))
                    |> List.map (fun l -> l / float n)
                let multiTrnLosses = trnLosses |> List.map (fun v -> v.Force()) |> multiAvg
                trainable.ResetModelState ()
                setDumpPrefix iter "val"
                let multiValLosses = valBatches |> Seq.map trainable.Losses |> multiAvg
                trainable.ResetModelState ()
                setDumpPrefix iter "tst"
                let multiTstLosses = tstBatches |> Seq.map trainable.Losses |> multiAvg

                // compute user qualities
                trainable.ResetModelState ()
                setDumpPrefix iter "userQuality"
                let userQualities = cfg.UserQualityFunc iter

                // log primary losses and user quality
                let entry = {
                    TrainingLog.Iter          = iter
                    TrainingLog.TrnLoss       = multiTrnLosses.Head
                    TrainingLog.ValLoss       = multiValLosses.Head
                    TrainingLog.TstLoss       = multiTstLosses.Head
                    TrainingLog.MultiTrnLoss  = multiTrnLosses
                    TrainingLog.MultiValLoss  = multiValLosses
                    TrainingLog.MultiTstLoss  = multiTstLosses
                    TrainingLog.UserQualities = userQualities
                    TrainingLog.LearningRate  = learningRate
                }
                let log = log |> TrainingLog.record entry trainable.ModelParameters
                cfg.LossRecordFunc entry

                // display primary and secondary losses
                printf "%6d:  trn=%7.4f  val=%7.4f  tst=%7.4f   " iter entry.TrnLoss entry.ValLoss entry.TstLoss
                match multiTrnLosses, multiValLosses, multiTstLosses with
                | [_], [_], [_] -> printf ""
                | _::secTrnLosses, _::secValLosses, _::secTstLosses ->
                    printf "("
                    for secTrnLoss, secValLoss, secTstLoss in 
                            List.zip3 secTrnLosses secValLosses secTstLosses do
                        printf "trn=%7.4f  val=%7.4f  tst=%7.4f; " secTrnLoss secValLoss secTstLoss
                    printf ")"
                | _ -> failwith "inconsistent losses"

                // display user qualities
                for KeyValue(name, qual) in entry.UserQualities do
                    printf "[%s:  trn=%7.4f  val=%7.4f  tst=%7.4f] " 
                           name qual.TrnQuality qual.ValQuality qual.TstQuality
                printfn "   "

                // check termination criteria
                let mutable faith = Continue

                match cfg.TargetLoss with
                | Some targetLoss when entry.ValLoss <= targetLoss -> 
                    printfn "Target loss reached"
                    faith <- TargetLossReached
                | _ -> ()

                match cfg.Termination with
                | ItersWithoutImprovement fiwi when 
                        TrainingLog.itersWithoutImprovement log > fiwi -> 
                    printfn "Trained for %d iterations without improvement" (TrainingLog.itersWithoutImprovement log)
                    faith <- NoImprovement
                | IterGain ig when
                        float iter > ig * float (TrainingLog.bestIter log) -> 
                    printfn "Trained for IterGain * %d = %d iterations" (TrainingLog.bestIter log) iter
                    faith <- NoImprovement
                | _ -> ()

                match cfg.MinIters with
                | Some minIters when iter < minIters -> 
                    if faith <> Continue then
                        printfn "But continuing since minimum number of iterations %d is not yet reached"
                            minIters
                    faith <- Continue
                | _ -> ()
                match cfg.MaxIters with
                | Some maxIters when iter >= maxIters -> 
                    printfn "Maximum number of iterations reached"
                    faith <- IterLimitReached
                | _ -> ()

                let isNan x = Double.IsInfinity x || Double.IsNaN x
                if isNan entry.TrnLoss || isNan entry.ValLoss || isNan entry.TstLoss then
                    faith <- NaNEncountered

                // process user input
                match Util.getKey () with
                | Some 'q' ->
                    printfn "Termination by user"
                    faith <- UserTerminated
                | Some 'd' ->
                    printfn "Learning rate decrease by user"
                    faith <- NoImprovement
                | _ -> ()

                // process checkpoint request
                match cfg.CheckpointInterval with
                | Some interval when iter % interval = 0 && faith = Continue -> 
                    faith <- CheckpointIntervalReached
                | _ -> ()
                if checkpointRequested then faith <- CheckpointRequested

                match faith with
                | Continue -> doTrain (iter + 1) learningRate log
                | _ -> log, faith
            else
                doTrain (iter + 1) learningRate log

        // training loop with decreasing learning rate
        let rec trainLoop prevFaith log learningRates = 
            match learningRates with
            | learningRate::rLearningRates ->
                // train
                if prevFaith <> CheckpointIntervalReached then
                    printfn "Using learning rate %g" learningRate
                let log, faith = doTrain (TrainingLog.lastIter log + 1) learningRate log

                if faith <> CheckpointRequested then
                    // restore best parameter values
                    match log.Best with
                    | Some (_, bestPv) -> trainable.ModelParameters <- bestPv
                    | None -> ()
                    trainable.InitOptState ()

                match faith with
                | NoImprovement 
                | NaNEncountered when not rLearningRates.IsEmpty -> 
                    // reset log to best iteration so far 
                    let log = log |> TrainingLog.removeToIter (TrainingLog.bestIter log)
                    // continue with lower learning rate
                    trainLoop faith log rLearningRates
                | _ -> log, faith, learningRates
            | [] -> failwith "no learning rates"
        
        // initialize or load checkpoint
        let cpLoadFilename = 
            match cfg.CheckpointFile, cfg.LoadCheckpointIter with
            | _ when cfg.DiscardCheckpoint -> None
            | Some _, Some _ -> 
                if File.Exists (checkpointFilename cfg.LoadCheckpointIter) then
                    Some (checkpointFilename cfg.LoadCheckpointIter)
                else failwithf "Checkpoint %s does not exist." 
                               (checkpointFilename cfg.LoadCheckpointIter)
            | Some _, None -> 
                if File.Exists (checkpointFilename None) then
                    Some (checkpointFilename None)
                else None
            | None, _ -> None
        let log, learningRates, duration, faith =
            match cpLoadFilename with
            | Some filename ->
                printfn "Loading checkpoint from %s" filename
                use cp = HDF5.OpenRead filename               
                let state : TrainState = cp.GetAttribute ("/", "TrainState") |> Json.deserialize
                let best =
                    match state.BestEntry with
                    | Some bestEntry ->
                        trainable.LoadModel cp "BestModel"
                        Some (bestEntry, trainable.ModelParameters |> Tensor.copy)
                    | None -> None
                let log = {TrainingLog.create cfg.MinImprovement cfg.BestOn with 
                            Best=best; History=state.History}
                trainable.LoadOptState cp "OptState"
                trainable.LoadModel cp "Model"               
                log, state.LearningRates, state.Duration, state.Faith
            | _ ->
                TrainingLog.create cfg.MinImprovement cfg.BestOn, cfg.LearningRates, TimeSpan.Zero, Continue

        // outer training loop with checkpoint saving
        let rec checkpointLoop log learningRates duration faith =
            match faith with
            | Continue | CheckpointRequested | CheckpointIntervalReached ->
                // train
                if faith <> CheckpointIntervalReached then
                    printfn "Training with %A" dataset
                let watch = Stopwatch.StartNew()
                let log, faith, learningRates = trainLoop faith log learningRates
                let duration = duration + watch.Elapsed

                // save checkpoint 
                match cfg.CheckpointFile with
                | Some _ ->
                    let cpFilename = checkpointFilename None
                    if faith <> CheckpointIntervalReached then
                        printfn "Saving checkpoint to %s" cpFilename

                    // write to temporary file
                    let cpTmpFilename = cpFilename + ".tmp"
                    using (HDF5.OpenWrite cpTmpFilename) (fun cp ->
                        trainable.SaveOptState cp "OptState"
                        trainable.SaveModel cp "Model"
                        let bestEntry =
                            match log.Best with
                            | Some (bestEntry, bestPars) ->
                                let curPars = trainable.ModelParameters |> Tensor.copy
                                trainable.ModelParameters <- bestPars
                                trainable.SaveModel cp "BestModel"
                                trainable.ModelParameters <- curPars
                                Some bestEntry
                            | None -> None
                        let trainState = {
                            History            = log.History
                            BestEntry          = bestEntry
                            LearningRates      = learningRates
                            Duration           = duration
                            Faith              = faith
                        }
                        cp.SetAttribute ("/", "TrainState", Json.serialize trainState))

                    // rename to checkpoint file
                    if File.Exists cpFilename then File.Replace (cpTmpFilename, cpFilename, null)
                    else File.Move (cpTmpFilename, cpFilename)

                    // copy to checkpoint iteration file
                    let cpIterFilename = checkpointFilename (Some (TrainingLog.lastIter log))
                    if faith = CheckpointIntervalReached && cpFilename <> cpIterFilename then
                        File.Copy (cpFilename, cpIterFilename, true)

                | None -> ()

                match faith with
                | CheckpointRequested -> exit 10
                | CheckpointIntervalReached -> checkpointLoop log learningRates duration faith
                | _ -> log, learningRates, duration, faith
            | _ ->
                // training already finished in loaded checkpoint
                log, learningRates, duration, faith
        
        let faith =
            if cfg.PerformTraining then faith
            else UserTerminated

        let log, learningRates, duration, faith = checkpointLoop log learningRates duration faith
        match TrainingLog.best log with
        | Some (bestEntry, _) ->
            printfn "Training completed after %d iterations in %A because %A with best losses:" 
                    bestEntry.Iter duration faith
            printfn "  trn=%7.4f  val=%7.4f  tst=%7.4f   " 
                    bestEntry.TrnLoss bestEntry.ValLoss bestEntry.TstLoss
        | None ->
            printfn "No training was performed."

        // restore original dump prefix
        Dump.prefix <- origDumpPrefix

        {
            History             = List.rev log.History
            Best                = log |> TrainingLog.best |> Option.map fst
            TerminationReason   = faith
            Duration            = duration
        }
        
                                   
        


