namespace GPAct
open SymTensor
open ArrayNDNS
open Datasets
open Models

module ClassificationError =

    /// Calculates the number of errors in one batch.           
    let errorsInBatch (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) (input:ArrayNDT<single>) (target:ArrayNDT<single>) =
        let predClass,predCov = modelPred input 
        let predClass = predClass |> ArrayNDHost.fetch
        let targ = target |> ArrayNDHost.fetch
        let predClass = predClass |> ArrayND.argMaxAxis 1
        let targetClass = targ |> ArrayND.argMaxAxis 1

        ArrayND.ifThenElse (predClass ==== targetClass) (ArrayNDHost.scalar 0.0f) (ArrayNDHost.scalar 1.0f)
        |> ArrayND.sum
        |> ArrayND.value 

    ///Calculates the number of errors in one dataset.
    let errorsInSet (modelPred: ArrayNDT<single> -> ArrayNDT<single>  * ArrayNDT<single>) (inSeq: seq<CsvLoader.CsvSample>) =
        inSeq
        |>Seq.map (fun {Input = inp; Target = trg} -> 
                    errorsInBatch modelPred inp trg)
        |>Seq.sum

    ///Calculates the fraction of errors for train-, validation- and test-dataset.
    let classificationErrors batchSize (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) =
        let trnBatches = dataset.Trn.Batches batchSize 
        let valBatches = dataset.Val.Batches batchSize 
        let tstBatches = dataset.Tst.Batches batchSize 
        let trnError = errorsInSet modelPred trnBatches  / (single dataset.Trn.NSamples)
        let valError = errorsInSet modelPred valBatches  / (single dataset.Val.NSamples)                
        let tstError = errorsInSet modelPred tstBatches  / (single dataset.Tst.NSamples)
        trnError,valError,tstError

    /// Prints the percentage of errors for train- validation and test-dataset.
    let printErrors batchSize (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) =
        let trnErr,valErr,tstErr = classificationErrors  batchSize dataset modelPred
        printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)

module RegressionError =
    
    type ErrorMeasure =
        MSE | RMSE  
    let mse (pred:ArrayNDT<single>) (target:ArrayNDT<single>) = (pred - target) ** 2.0f |> ArrayND.mean
    ///Calculates the number of error in one dataset.
    
    let subsetError (modelPred: ArrayNDT<single> -> ArrayNDT<single>  * ArrayNDT<single>) (inData: CsvLoader.CsvSample) (errorMeasure:ErrorMeasure) =
        let errorFunction =
            match errorMeasure with
            | MSE -> mse
            | RMSE -> (fun x y ->sqrt( mse x y))
        let {CsvLoader.Input = inp; CsvLoader.Target = trg} = inData
        let mean,cov = modelPred inp
        errorFunction mean trg |> ArrayND.value


    let calculateErrors (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) (errorMeasure:ErrorMeasure) =
        let trnError = subsetError modelPred dataset.Trn.All errorMeasure
        let valError = subsetError modelPred dataset.Val.All errorMeasure
        let tstError = subsetError modelPred dataset.Tst.All errorMeasure
        trnError, valError, tstError
    
    let printRMSEs (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) =
        let trnError, valError, tstError = calculateErrors dataset modelPred RMSE
        printfn "Train RMSE = %f, Validation RMSE = %f, Test RMSE =%f" trnError valError tstError

    let printMSEs (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) =
        let trnError, valError, tstError = calculateErrors dataset modelPred MSE
        printfn "Train MSE = %f, Validation MSE = %f, Test MSE =%f" trnError valError tstError