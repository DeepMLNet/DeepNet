﻿namespace Models

open Basics
open ArrayNDNS
open Datasets
open SymTensor


/// Training history functions.
module TrainingLog =

    type Entry = {
        Iter:               int
        TrnLoss:            single
        ValLoss:            single
        TstLoss:            single
    }

    type Log<'P> = {
        MinImprovement:     single
        Best:               (Entry * ArrayNDT<'P>) option
        History:            Entry list
    }

    let create minImprovement =
        {MinImprovement=minImprovement; Best=None; History=[]}

    let record (entry: Entry) parVals (log: Log<_>) =
        let best =
            match log.Best with
            | None -> Some (entry, parVals)
            | Some (bestEntry, _) when entry.ValLoss <= bestEntry.ValLoss - log.MinImprovement ->
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

    let itersWithoutImprovement (log: Log<_>) =
        match log.Best with
        | None -> 0
        | Some ({Iter=bestIter}, _) -> lastIter log - bestIter

    let removeToIter iter (log: Log<_>) =
        {log with History = log.History |> List.skipWhile (fun {Iter=i} -> i > iter)}


/// Training functions.
module Train =

    /// Training termination criterium
    type TerminationCriterium =
        /// terminates after the given number of iterations with improvement of the validation loss
        | FixedItersWithoutImprovement of int
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
        /// number of interations between evaluation of the loss
        LossRecordInterval:             int
        /// termination criterium
        Termination:                    TerminationCriterium
        /// minimum loss decrease to count as improvement
        MinImprovement:                 single
        /// target loss that should lead to termination of training
        TargetLoss:                     single option
        /// minimum training iterations
        MinIters:                       int option
        /// maximum training iterations
        MaxIters:                       int option
        /// Learning rates that will be used. After training terminates with
        /// one learning rate, it continues using the next learning rate from this list.
        LearningRates:                  single list
    }

    /// Default training configuration.
    let defaultCfg = {
        Seed                        = 1
        BatchSize                   = 10000
        LossRecordInterval          = 10
        Termination                 = IterGain 1.25
        MinImprovement              = 1e-7f
        TargetLoss                  = None
        MinIters                    = Some 100
        MaxIters                    = None
        LearningRates               = [1e-3f; 1e-4f; 1e-5f; 1e-6f]
    }

    /// Trains a model instance using the given loss and optimization functions on the given dataset.
    /// Returns the training history.
    let train (modelInstance: ModelInstance<'P>) (lossFn: 'S -> single) (optFn: single -> 'S -> single Lazy) 
              (dataset: TrnValTst<'S>) (cfg: Cfg) =
        
        // batches
        let trnBatches = dataset.Trn.Batches cfg.BatchSize
        let valBatches = dataset.Val.Batches cfg.BatchSize
        let tstBatches = dataset.Tst.Batches cfg.BatchSize

        // initialize model parameters
        printfn "Initializing model parameters for training"
        modelInstance.InitPars cfg.Seed

        /// training function
        let rec doTrain iter learningRate log =
            // execute training
            let trnLosses = trnBatches () |> Seq.map (optFn learningRate) 

            // record loss
            if iter % cfg.LossRecordInterval = 0 then
                // compute and log validation & test losses
                let entry = {
                    TrainingLog.Iter    = iter
                    TrainingLog.TrnLoss = trnLosses |> Seq.averageBy (fun v -> v.Force())
                    TrainingLog.ValLoss = valBatches () |> Seq.map lossFn |> Seq.average
                    TrainingLog.TstLoss = tstBatches () |> Seq.map lossFn |> Seq.average
                }
                let log = log |> TrainingLog.record entry modelInstance.ParameterValues
                printfn "%5d:  trn=%3.4f  val=%3.4f  tst=%3.4f" iter entry.TrnLoss entry.ValLoss entry.TstLoss

                // check termination criteria
                let mutable terminate = false

                match cfg.TargetLoss with
                | Some targetLoss when entry.ValLoss <= targetLoss -> 
                    printfn "Target loss reached"
                    terminate <- true
                | _ -> ()

                match cfg.Termination with
                | FixedItersWithoutImprovement fiwi when 
                        TrainingLog.itersWithoutImprovement log > fiwi -> 
                    printfn "Trained for %d iterations without improvement" (TrainingLog.itersWithoutImprovement log)
                    terminate <- true
                | IterGain ig when
                        float iter > ig * float (TrainingLog.bestIter log) -> 
                    printfn "Trained for IterGain * %d = %d iterations" (TrainingLog.bestIter log) iter
                    terminate <- true
                | _ -> ()

                match cfg.MaxIters with
                | Some maxIters when iter >= maxIters -> 
                    printfn "Maximum number of iterations reached"
                    terminate <- true
                | _ -> ()
                match cfg.MinIters with
                | Some minIters when iter < minIters -> terminate <- false
                | _ -> ()

                if terminate then log
                else doTrain (iter + 1) learningRate log
            else
                doTrain (iter + 1) learningRate log

        // train with decreasing learning rate
        printfn "Training with dataset %A" dataset
        let log =
            (TrainingLog.create cfg.MinImprovement, cfg.LearningRates)
            ||> Seq.fold (fun log lr ->
                // rest log to best iteration so far
                let log = log |> TrainingLog.removeToIter (TrainingLog.bestIter log)
                // train
                printfn "Training with learning rate %g" lr
                let log = doTrain (TrainingLog.lastIter log) lr log
                // restore best parameter values
                match log.Best with
                | Some (_, bestPv) -> modelInstance.ParameterValues.[Fill] <- bestPv
                | None -> ()

                log
            )           
        printfn "Training completed"

        log
                                  

        
        


