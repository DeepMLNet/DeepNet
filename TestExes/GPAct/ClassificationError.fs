namespace GPAct
open SymTensor
open ArrayNDNS
open Datasets

module ClassificationError =

    /// Calculates the number of errors in one batch.           
    let errorsInBatch batchSize (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) (input:ArrayNDT<single>) (target:ArrayNDT<single>) =
        let predClass,predCov = modelPred input 
        let predClass = predClass |> ArrayNDHost.fetch
        let targ = target |> ArrayNDHost.fetch
        let predClass = predClass |> ArrayND.argMaxAxis 1
        let targetClass = targ |> ArrayND.argMaxAxis 1

        ArrayND.ifThenElse (predClass ==== targetClass) (ArrayNDHost.scalar 0.0f) (ArrayNDHost.scalar 1.0f)
        |> ArrayND.sum
        |> ArrayND.value 

    ///Calculates the number of errors in one dataset.
    let errorsInSet batchSize (modelPred: ArrayNDT<single> -> ArrayNDT<single>  * ArrayNDT<single>) (inSeq: seq<CsvLoader.CsvSample>) =
        inSeq
        |>Seq.map (fun {Input = inp; Target = trg} -> 
                    errorsInBatch batchSize modelPred inp trg)
        |>Seq.sum

    ///Calculates the fraction of errors for train-, validation- and test-dataset.
    let classificationErrors batchSize (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) =
        let trnBatches = dataset.Trn.Batches batchSize 
        let valBatches = dataset.Val.Batches batchSize 
        let tstBatches = dataset.Tst.Batches batchSize 
        let trnError = errorsInSet batchSize modelPred trnBatches  / (single dataset.Trn.NSamples)
        let valError = errorsInSet batchSize modelPred valBatches  / (single dataset.Val.NSamples)                
        let tstError = errorsInSet batchSize modelPred tstBatches  / (single dataset.Tst.NSamples)
        trnError,valError,tstError

    /// Prints the percentage of errors for train- validation and test-dataset.
    let printErrors batchSize (dataset:TrnValTst<CsvLoader.CsvSample>) (modelPred: ArrayNDT<single> -> ArrayNDT<single> * ArrayNDT<single>) =
        let trnErr,valErr,tstErr = classificationErrors  batchSize dataset modelPred
        printfn "Train Error = %f%%, Validation Error = %f%%, Test Error =%f%% " (trnErr*100.0f) (valErr*100.0f) (tstErr*100.0f)
