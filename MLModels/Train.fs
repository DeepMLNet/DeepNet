namespace Models

open System
open System.Diagnostics
open System.IO
open Nessos.FsPickler.Json

open Basics
open ArrayNDNS
open Datasets
open SymTensor
open Optimizers


type Partition =
    | Training
    | Validation


/// Training history module.
module TrainingLog =

    type Entry = {
        Iter:               int
        TrnLoss:            float
        ValLoss:            float
        TstLoss:            float
        LearningRate:       float
    }

    type Log<'P> = {
        MinImprovement:     float
        BestOn:             Partition
        Best:               (Entry * ArrayNDT<'P>) option
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
                Some (entry, ArrayND.copy parVals)
            | _ -> log.Best
        {log with Best=best; History=entry :: log.History}

    let lastIter (log: Log<_>) =
        match log.History with
        | {Iter=iter}::_ -> iter
        | [] -> 0

    let bestIter (log: Log<_>) =
        match log.Best with
        | Some ({Iter=iter}, _) -> iter
        | None -> 0

    let best (log: Log<_>) =
        log.Best.Value

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
        BatchSize:                      int
        /// number of iterations between evaluation of the loss
        LossRecordInterval:             int
        /// function that is called after loss has been evaluated
        LossRecordFunc:                 TrainingLog.Entry -> unit
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
        /// Checkpoint storage directory. (is created if it does not exist)
        CheckpointDir:                  string option
        /// If true, checkpoint is not loaded from disk.
        DiscardCheckpoint:              bool
        /// If set, during each iteration the dump prefix will be set to the given string
        /// concatenated with the iteration number.
        DumpPrefix:                     string option
    } 

    /// Default training configuration.
    let defaultCfg = {
        Seed                        = 1
        BatchSize                   = 10000
        LossRecordInterval          = 10
        LossRecordFunc              = fun _ -> ()
        Termination                 = IterGain 1.25
        MinImprovement              = 1e-7
        BestOn                      = Validation
        TargetLoss                  = None
        MinIters                    = Some 100
        MaxIters                    = None
        LearningRates               = [1e-3; 1e-4; 1e-5; 1e-6]
        CheckpointDir               = None
        DiscardCheckpoint           = false
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
        | NaNEncountered

    /// Result of training
    type TrainingResult = {
        Best:               TrainingLog.Entry
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
        abstract member LoadModel: path:string -> unit
        /// Save model parameters to specified file.
        abstract member SaveModel: path:string -> unit
        /// Model parameter values (i.e. weights).
        abstract member ModelParameters: ArrayNDT<'T> with get, set
        /// Initialize optimizer state.
        abstract member InitOptState: unit -> unit
        /// Load optimizer state from specified file.
        abstract member LoadOptState: path: string -> unit
        /// Save optimizer state to specified file.
        abstract member SaveOptState: path: string -> unit

    /// Constructs an ITrainable<_> for the given model instance, loss expressions and optimizer.
    let trainableFromLossExprs
            (modelInstance: ModelInstance<'T>) 
            (losses: ExprT list) 
            (varEnvBuilder: 'Smpl -> VarEnvT)
            (optNew: ExprT -> ExprT -> IDevice -> IOptimizer<'T, 'OptCfg, 'OptState>)
            (optCfg: 'OptCfg) =         
   
        let losses = losses |> List.map modelInstance.Use 
        let mainLoss = losses.Head
        let opt = optNew mainLoss modelInstance.ParameterVector modelInstance.Device
        let lossesFn = modelInstance.Func losses << varEnvBuilder
        let lossesOptFn = modelInstance.Func (losses @ [opt.OptStepExpr]) |> opt.Use << varEnvBuilder

        let mutable optState = opt.InitialState optCfg modelInstance.ParameterValues
    
        {new ITrainable<'Smpl, 'T> with
            member this.Losses sample = lossesFn sample |> List.map (ArrayND.value >> conv<float>)
            member this.Optimize learningRate sample = 
                let lossesAndOpt = lossesOptFn sample (opt.CfgWithLearningRate learningRate optCfg) optState
                let losses = lossesAndOpt |> List.take (lossesAndOpt.Length - 1)
                lazy (losses |> List.map (ArrayND.value >> conv<float>))
            member this.PrintInfo () = modelInstance.ParameterStorage.PrintShapes ()
            member this.InitModel seed = modelInstance.InitPars seed
            member this.LoadModel path = modelInstance.LoadPars path
            member this.SaveModel path = modelInstance.SavePars path
            member this.ModelParameters
                with get () = modelInstance.ParameterValues
                and set (value) = modelInstance.ParameterValues <- value
            member this.InitOptState () = optState <- opt.InitialState optCfg modelInstance.ParameterValues
            member this.LoadOptState path = optState <- opt.LoadState path
            member this.SaveOptState path = opt.SaveState path optState    
        }

    /// Constructs an ITrainable<_> for the given model instance, loss expression and optimizer.
    let trainableFromLossExpr modelInstance loss varEnvBuilder optNew optCfg =     
        trainableFromLossExprs modelInstance [loss] varEnvBuilder optNew optCfg

    type private CheckpointFiles (cfg: Cfg) = 
        let dir = Path.GetFullPath cfg.CheckpointDir.Value    
        member this.Directory = dir       
        member this.ModelFile = Path.Combine (dir, "model.h5")
        member this.BestModelFile = Path.Combine (dir, "bestmodel.h5")
        member this.OptStateFile = Path.Combine (dir, "optstate.h5")
        member this.TrainStateFile = Path.Combine (dir, "trainstate.json")
        member this.Exists = 
            File.Exists this.ModelFile && File.Exists this.BestModelFile && 
            File.Exists this.OptStateFile && File.Exists this.TrainStateFile 
        member this.Mkdir () = Directory.CreateDirectory dir |> ignore
        member this.Remove () =
            if File.Exists this.ModelFile then File.Delete this.ModelFile
            if File.Exists this.BestModelFile then File.Delete this.BestModelFile
            if File.Exists this.OptStateFile then File.Delete this.OptStateFile
            if File.Exists this.TrainStateFile then File.Delete this.TrainStateFile
            try Directory.Delete (dir, false) with :? IOException -> ()

    type private TrainState = {
        History:        TrainingLog.Entry list
        BestEntry:      TrainingLog.Entry option
        LearningRates:  float list
        Duration:       TimeSpan
        Faith:          Faith
    }

    /// Trains a model instance using the given loss and optimization functions on the given dataset.
    /// Returns the training history.
    let train (trainable: ITrainable<'Smpl, 'T>) (dataset: TrnValTst<'Smpl>) (cfg: Cfg) =
        // checkpoint data
        let cp =
            match cfg.CheckpointDir with
            | Some _ -> 
                let cp = CheckpointFiles cfg
                if cfg.DiscardCheckpoint then cp.Remove ()
                Some cp
            | None -> None
        let mutable checkpointRequested = false
        use ctrlCHandler = Console.CancelKeyPress.Subscribe (fun evt ->
            match cp with
            | Some _ ->
                checkpointRequested <- true
                evt.Cancel <- true
            | None -> ()
        )

        // batches
        let trnBatches = dataset.Trn.Batches cfg.BatchSize
        let valBatches = dataset.Val.Batches cfg.BatchSize
        let tstBatches = dataset.Tst.Batches cfg.BatchSize

        if Seq.isEmpty trnBatches then failwith "the training set is empty"
        if Seq.isEmpty valBatches then failwith "the validation set is empty"
        if Seq.isEmpty tstBatches then failwith "the test set is empty"

        // initialize model parameters
        printfn "Initializing model parameters for training"
        trainable.InitModel cfg.Seed
        trainable.PrintInfo ()

        /// training function
        let rec doTrain iter learningRate log =

            /// set dump prefix
            match cfg.DumpPrefix with
            | Some dp -> Dump.prefix <- sprintf "%s%d" dp iter
            | None -> ()

            // execute training
            let trnLosses = trnBatches |> Seq.map (trainable.Optimize learningRate) |> Seq.toList

            // record loss
            if iter % cfg.LossRecordInterval = 0 then

                let multiAvg lls =
                    let mutable n = 1
                    lls
                    |> Seq.reduce (fun ll1 ll2 ->
                        n <- n + 1
                        List.zip ll1 ll2
                        |> List.map (fun (l1, l2) -> l1 + l2))
                    |> List.map (fun l -> l / float n)

                let multiTrnLosses = trnLosses |> List.map (fun v -> v.Force()) |> multiAvg
                let multiValLosses = valBatches |> Seq.map trainable.Losses |> multiAvg
                let multiTstLosses = tstBatches |> Seq.map trainable.Losses |> multiAvg

                // compute and log primary validation & test loss
                let entry = {
                    TrainingLog.Iter    = iter
                    TrainingLog.TrnLoss = multiTrnLosses.Head
                    TrainingLog.ValLoss = multiValLosses.Head
                    TrainingLog.TstLoss = multiTstLosses.Head
                    TrainingLog.LearningRate = learningRate
                }
                let log = log |> TrainingLog.record entry trainable.ModelParameters
                printf "%6d:  trn=%7.4f  val=%7.4f  tst=%7.4f   " iter entry.TrnLoss entry.ValLoss entry.TstLoss
                cfg.LossRecordFunc entry

                // print secondary losses
                match multiTrnLosses, multiValLosses, multiTstLosses with
                | [_], [_], [_] -> printfn ""
                | _::secTrnLosses, _::secValLosses, _::secTstLosses ->
                    printf "("
                    for secTrnLoss, secValLoss, secTstLoss in 
                            List.zip3 secTrnLosses secValLosses secTstLosses do
                        printf "trn=%7.4f  val=%7.4f  tst=%7.4f; " secTrnLoss secValLoss secTstLoss
                    printfn ")"
                | _ -> failwith "inconsistent losses"

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
                if checkpointRequested then faith <- CheckpointRequested

                match faith with
                | Continue -> doTrain (iter + 1) learningRate log
                | _ -> log, faith
            else
                doTrain (iter + 1) learningRate log

        // training loop with decreasing learning rate
        let rec trainLoop log learningRates = 
            match learningRates with
            | learningRate::rLearningRates ->
                // train
                printfn "Using learning rate %g" learningRate
                let log, faith = doTrain (TrainingLog.lastIter log + 1) learningRate log

                if faith <> CheckpointRequested then
                    // restore best parameter values
                    match log.Best with
                    | Some (_, bestPv) -> trainable.ModelParameters <- bestPv
                    | None -> ()

                match faith with
                | NoImprovement 
                | NaNEncountered when not rLearningRates.IsEmpty -> 
                    // reset log to best iteration so far 
                    let log = log |> TrainingLog.removeToIter (TrainingLog.bestIter log)
                    // continue with lower learning rate
                    trainLoop log rLearningRates
                | _ -> log, faith, learningRates
            | [] -> failwith "no learning rates"
        
        // initialize or load checkpoint
        let log, learningRates, duration, faith =
            match cp with
            | Some cp when cp.Exists ->
                printfn "Loading checkpoint from %s" cp.Directory
                let state : TrainState = Json.load cp.TrainStateFile
                let best =
                    match state.BestEntry with
                    | Some bestEntry ->
                        trainable.LoadModel cp.BestModelFile
                        Some (bestEntry, trainable.ModelParameters |> ArrayND.copy)
                    | None -> None
                let log = {TrainingLog.create cfg.MinImprovement cfg.BestOn with 
                            Best=best; History=state.History}
                trainable.LoadOptState cp.OptStateFile
                trainable.LoadModel cp.ModelFile
                log, state.LearningRates, state.Duration, state.Faith
            | _ ->
                TrainingLog.create cfg.MinImprovement cfg.BestOn, cfg.LearningRates, TimeSpan.Zero, Continue
        
        // train
        let log, duration, faith = 
            match faith with
            | Continue | CheckpointRequested ->
                // train
                printfn "Training with %A" dataset
                let watch = Stopwatch.StartNew()
                let log, faith, learningRates = trainLoop log learningRates
                let duration = duration + watch.Elapsed

                // save checkpoint 
                match cp with
                | Some cp ->
                    printfn "Saving checkpoint to %s" cp.Directory
                    cp.Mkdir ()
                    trainable.SaveOptState cp.OptStateFile
                    trainable.SaveModel cp.ModelFile
                    let bestEntry =
                        match log.Best with
                        | Some (bestEntry, bestPars) ->
                            let curPars = trainable.ModelParameters |> ArrayND.copy
                            trainable.ModelParameters <- bestPars
                            trainable.SaveModel cp.BestModelFile
                            trainable.ModelParameters <- curPars
                            Some bestEntry
                        | None -> None
                    Json.save cp.TrainStateFile {
                        History = log.History
                        BestEntry = bestEntry
                        LearningRates = learningRates
                        Duration = duration
                        Faith = faith
                    }
                    if faith = CheckpointRequested then exit 10
                | None -> ()

                log, duration, faith
            | _ ->
                // training already finished in loaded checkpoint
                printfn "Training finished in loaded checkpoint"
                log, duration, faith

        let bestEntry, _ = TrainingLog.best log
        printfn "Training completed after %d iterations in %A because %A with best losses:" 
            bestEntry.Iter duration faith
        printfn "  trn=%7.4f  val=%7.4f  tst=%7.4f   " 
            bestEntry.TrnLoss bestEntry.ValLoss bestEntry.TstLoss

        {
            History             = List.rev log.History
            Best                = bestEntry
            TerminationReason   = faith
            Duration            = duration
        }
        
                                   
        


