namespace Tensor.Expr.ML

open System
open System.Diagnostics
open System.IO
//open MBrace.FsPickler.Json

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.ML.Opt


/// Training, validation and test values.
type PartVals<'T> = {
    Trn: 'T
    Val: 'T
    Tst: 'T
}


/// Collection of primary and custom losses.
type Losses = {
    Primary: float
    Custom: Map<string, float>
} with

    static member (+) (a: Losses, b: Losses) = {
        Primary = a.Primary + b.Primary
        Custom = a.Custom |> Map.map (fun key av -> av + b.Custom.[key])
    }

    static member (/) (a: Losses, n: int) = {
        Primary = a.Primary / float n
        Custom = a.Custom |> Map.map (fun _ av -> av / float n)
    }

    static member average (ls: Losses seq) =
        let ls = Seq.cache ls
        let sum = ls |> Seq.reduce (+)
        sum / Seq.length ls
            

/// Training, validation and test losses.
type PartLosses = PartVals<Losses>


/// Training history.
module TrainingLog =

    /// Training history entry.
    type Entry = {
        Iter:               int
        Loss:               PartLosses
        Quality:            Map<string, PartVals<float>>
        LearningRate:       float
    }


/// Training history.
type TrainingLog = {
    MinImprovement:     float
    BestOn:             Partition
    Best:               (TrainingLog.Entry * ParSetInst) option
    History:            TrainingLog.Entry list
} with

    static member create minImprovement bestOn = {
        MinImprovement=minImprovement
        BestOn=bestOn
        Best=None
        History=[]
    }

    static member relevantLoss bestOn (entry: TrainingLog.Entry) =
        match bestOn with
        | Partition.Trn -> entry.Loss.Trn.Primary
        | Partition.Val -> entry.Loss.Val.Primary
        | Partition.Tst -> failwith "The termination criterium cannot depend on the test loss."

    static member record (entry: TrainingLog.Entry) parVals (log: TrainingLog) =
        let best =
            match log.Best with
            | None -> Some (entry, parVals)
            | Some (bestEntry, _) when
                    (TrainingLog.relevantLoss log.BestOn entry) 
                     <= (TrainingLog.relevantLoss log.BestOn bestEntry) - log.MinImprovement ->
                Some (entry, ParSetInst.copy parVals)
            | _ -> log.Best
        {log with Best=best; History=entry :: log.History}

    static member lastIter (log: TrainingLog) =
        match log.History with
        | {Iter=iter}::_ -> iter
        | [] -> -1

    static member bestIter (log: TrainingLog) =
        match log.Best with
        | Some ({Iter=iter}, _) -> iter
        | None -> 0

    static member best (log: TrainingLog) =
        log.Best

    static member itersWithoutImprovement (log: TrainingLog) =
        match log.Best with
        | None -> 0
        | Some ({Iter=bestIter}, _) -> TrainingLog.lastIter log - bestIter

    static member removeToIter iter (log: TrainingLog) =
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
        UserQualityFunc:                int -> Map<string, PartVals<float>>
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
        BestOn                      = Partition.Val
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
    type ITrainable<'Smpl> =
        /// Losses of given sample.
        abstract Losses: sample:'Smpl -> Losses
        /// Perform an optimization step and return the losses.
        abstract Step: sample:'Smpl -> Lazy<Losses>
        /// Learning rate.
        abstract LearningRate: float with set
        /// Prints information about the model.
        abstract PrintInfo: unit -> unit
        /// Initializes the model using the given random seed.
        abstract InitModel: unit -> unit
        /// Load model parameters from specified file.
        abstract LoadModel: hdf:HDF5 -> prefix:string -> unit
        /// Save model parameters to specified file.
        abstract SaveModel: hdf:HDF5 -> prefix:string -> unit
        /// Model parameter values (i.e. weights).
        abstract ModelParameters: ParSetInst with get, set
        /// Model state. (for example the latent state of an RNN)
        abstract ModelState: Option<ParSetInst> with get, set
        /// Resets the internal model state. (for example the latent state of an RNN)
        abstract ResetModelState: unit -> unit
        /// Initialize optimizer state.
        abstract InitOptState: unit -> unit
        /// Load optimizer state from specified file.
        abstract LoadOptState: hdf:HDF5 -> prefix:string -> unit
        /// Save optimizer state to specified file.
        abstract SaveOptState: hdf:HDF5 -> prefix:string -> unit

    type TrainableFromExpr<'Smpl> = {
        /// Primary loss function.
        /// It is minimized during training.
        PrimaryLoss: Expr<float>
        /// Secondary loss functions that are evaluated during training.
        /// They do not enter the optimization objective.
        CustomLoss: Map<string, Expr<float>>
        /// Parameter set instance that contains the model parameters.
        /// Optimization takes place w.r.t. these parameters.
        Pars: ParSetInst
        /// Optional model state that should persist between samples and its assoicated
        /// update expression.
        /// (For example the latent state of an RNN).
        State: (ParSetInst * EvalUpdateBundle) option
        /// Function that takes a sample (mini-batch) and returns the corresponding VarEnv.
        VarEnvForSample: 'Smpl -> VarEnv
        /// Optimizer configuration.
        OptCfg: IOptimizerCfg
    } with
        static member simple loss pars varEnvForSample optCfg = {
            PrimaryLoss = loss
            CustomLoss = Map.empty
            Pars = pars
            State = None
            VarEnvForSample = varEnvForSample
            OptCfg = optCfg
        }

    /// Constructs an ITrainable<_> from expressions.
    let internal newTrainable (spec: TrainableFromExpr<'Smpl>) =         
   
        // state variable environment
        let stateEnv =
            match spec.State with
            | Some (state, _) -> state.VarEnv
            | None -> VarEnv.empty
        
        // state update bundle
        let stateBndl =
            match spec.State with
            | Some (state, stateBndl) -> 
                stateBndl
                |> spec.Pars.Use
                |> state.Use
            | None -> EvalUpdateBundle.empty

        /// use state and parameter ParSets in an expression und EvalUpdateBundle
        let useStateAndPars (x: Expr<float>) =
            let x = spec.Pars.Use x
            match spec.State with 
            | Some (state, _) -> state.Use x
            | None -> x

        // losses
        let primaryLoss = useStateAndPars spec.PrimaryLoss  
        let customLoss = spec.CustomLoss |> Map.map (fun _ loss -> useStateAndPars loss)
           
        // loss bundle
        let lossBndl = 
            (EvalUpdateBundle.empty, Map.toSeq customLoss)
            ||> Seq.fold (fun bndl (_, expr) -> bndl |> EvalUpdateBundle.addExpr expr)
            |> EvalUpdateBundle.addExpr primaryLoss

        // loss claculation and state update bundle
        let lossStateBndl = EvalUpdateBundle.merge lossBndl stateBndl

        // optimizer
        let opt = spec.OptCfg.NewOptimizer primaryLoss.Untyped spec.Pars
        let optStep = opt.Step
        let optLossStateBndl = EvalUpdateBundle.merge lossStateBndl optStep

        // function that extracts losses from evaluated expression values
        let extractLosses (res: ExprVals) = {
            Primary = res.Get primaryLoss |> Tensor.value
            Custom = customLoss |> Map.map (fun _ expr -> res.Get expr |> Tensor.value)
        }
   
        {new ITrainable<'Smpl> with
            member this.Losses smpl = 
                let varEnv = VarEnv.joinMany [spec.VarEnvForSample smpl; spec.Pars.VarEnv; stateEnv]
                lossStateBndl |> EvalUpdateBundle.exec varEnv |> extractLosses

            member this.Step smpl = 
                let varEnv = VarEnv.joinMany [spec.VarEnvForSample smpl; spec.Pars.VarEnv; stateEnv]
                let res = optLossStateBndl |> EvalUpdateBundle.exec varEnv
                lazy (extractLosses res)

            member this.LearningRate
                with set lr = opt.Cfg <- opt.Cfg.SetLearningRate lr

            member this.PrintInfo () = printfn "%A" spec.Pars

            member this.InitModel () = spec.Pars.Init()
            member this.LoadModel hdf prefix = spec.Pars.Load (hdf, prefix)
            member this.SaveModel hdf prefix = spec.Pars.Save (hdf, prefix)
            member this.ModelParameters
                with get () = spec.Pars
                and set (value) = spec.Pars.CopyFrom value

            member this.ModelState
                with get () = spec.State |> Option.map fst
                and set (value) = 
                    match spec.State, value with
                    | Some (statePars, _), Some value -> statePars.CopyFrom value
                    | None, None -> ()
                    | _ -> failwith "Trainable state mismatch."
            member this.ResetModelState () = 
                match spec.State with
                | Some (statePars, _) -> statePars.Init()
                | None -> ()

            member this.InitOptState () = 
                opt.State <- opt.State.Initial()
            member this.LoadOptState hdf prefix = 
                let state = opt.State
                state.Load (hdf, prefix)
                opt.State <- state
            member this.SaveOptState hdf prefix = 
                opt.State.Save (hdf, prefix)
        }

    ///// Constructs an ITrainable<_> for the given stateful model using the specified loss
    ///// expressions, state update expression and optimizer.
    //let newStatefulTrainable modelInstance losses nextState varEnvBuilder optNew optCfg =
    //    newTrainable modelInstance losses (Some nextState) varEnvBuilder optNew optCfg

    ///// Constructs an ITrainable<_> for the given model instance, loss expressions and optimizer.
    //let trainableFromLossExprs modelInstance losses varEnvBuilder optNew optCfg =
    //    newTrainable modelInstance losses None (fun _ -> varEnvBuilder) optNew optCfg

    ///// Constructs an ITrainable<_> for the given model instance, loss expression and optimizer.
    //let trainableFromLossExpr modelInstance loss varEnvBuilder optNew optCfg =     
    //    trainableFromLossExprs modelInstance [loss] varEnvBuilder optNew optCfg


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
    let train (trainable: ITrainable<'Smpl>) (dataset: TrnValTst<'Smpl>) (cfg: Cfg) =
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
        trainable.InitModel ()
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
            trainable.LearningRate <- learningRate
            trainable.ResetModelState ()
            setDumpPrefix iter "trn"
            let trnLosses = trnBatches |> Seq.map trainable.Step |> Seq.toList

            // record loss if needed
            let recordBecauseCP = 
                match cfg.CheckpointInterval with 
                | Some interval -> iter % interval = 0
                | None -> false
            if iter % cfg.LossRecordInterval = 0 || recordBecauseCP then

                // compute validation & test losses
                let multiTrnLosses = trnLosses |> List.map (fun v -> v.Force()) |> Losses.average
                trainable.ResetModelState ()
                setDumpPrefix iter "val"
                let multiValLosses = valBatches |> Seq.map trainable.Losses |> Losses.average
                trainable.ResetModelState ()
                setDumpPrefix iter "tst"
                let multiTstLosses = tstBatches |> Seq.map trainable.Losses |> Losses.average

                // compute user qualities
                trainable.ResetModelState ()
                setDumpPrefix iter "userQuality"
                let userQualities = cfg.UserQualityFunc iter

                // log primary losses and user quality
                let entry = {
                    TrainingLog.Iter          = iter
                    TrainingLog.Loss          = {Trn=multiTrnLosses; Val=multiValLosses; Tst=multiTstLosses}
                    TrainingLog.Quality       = userQualities
                    TrainingLog.LearningRate  = learningRate
                }
                let log = log |> TrainingLog.record entry trainable.ModelParameters
                cfg.LossRecordFunc entry

                // display primary and secondary losses
                printf "%6d:  trn=%7.4f  val=%7.4f  tst=%7.4f   " 
                    iter entry.Loss.Trn.Primary entry.Loss.Val.Primary entry.Loss.Tst.Primary
                if not entry.Loss.Trn.Custom.IsEmpty then
                    printf "("
                    for KeyValue(name, trn) in entry.Loss.Trn.Custom do
                        printf "%s: trn=%7.4f  val=%7.4f  tst=%7.4f; " 
                            name trn entry.Loss.Val.Custom.[name] entry.Loss.Tst.Custom.[name]
                    printf ")"

                // display user qualities
                for KeyValue(name, qual) in entry.Quality do
                    printf "[%s:  trn=%7.4f  val=%7.4f  tst=%7.4f] " 
                           name qual.Trn qual.Val qual.Tst
                printfn "   "

                // check termination criteria
                let mutable faith = Continue

                match cfg.TargetLoss with
                | Some targetLoss when entry.Loss.Val.Primary <= targetLoss -> 
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
                if isNan entry.Loss.Trn.Primary || isNan entry.Loss.Val.Primary || isNan entry.Loss.Tst.Primary then
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
                        Some (bestEntry, trainable.ModelParameters |> ParSetInst.copy)
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
                                let curPars = trainable.ModelParameters |> ParSetInst.copy
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
                    bestEntry.Loss.Trn.Primary bestEntry.Loss.Val.Primary bestEntry.Loss.Tst.Primary
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
        
                                   
        


