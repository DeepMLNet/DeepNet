namespace Models 

open Basics
open Tensor
open Datasets


/// Accuracy, defined by probability of correct classification.
module Accuracy =

    /// Calculates the number of correctly classified samples.
    /// Target must be in one-hot encoding.
    /// Shapes: pred[smpl, class], target[smpl, class]
    let correctlyClassified (trgt: Tensor<'T>) (pred: Tensor<'T>) =     
        if Tensor.nDims pred <> 2 || Tensor.nDims trgt <> 2 then
            failwith "pred and target must be two-dimensional"      
        let pred = pred |> HostTensor.transfer
        let trgt = trgt |> HostTensor.transfer

        let predClass = pred |> Tensor.argMaxAxis 1
        let trgtClass = trgt |> Tensor.argMaxAxis 1

        Tensor.ifThenElse (predClass ==== trgtClass) (HostTensor.scalar 1.0) (HostTensor.scalar 0.0)
        |> Tensor.sum
        |> Tensor.value 

    /// Calculates the accuracies of a classifier on the training, validation and test sets.
    let ofClassifier (dataset: TrnValTst<'S>) batchSize 
            (trgtFn: 'S -> Tensor<'T>) (predFn: 'S -> Tensor<'T>) =
        dataset |> TrnValTst.apply (fun part ->
            part 
            |> Dataset.batches batchSize 
            |> Seq.map (fun batch -> correctlyClassified (trgtFn batch) (predFn batch))
            |> Seq.sum
            |> fun correct -> correct / float part.NSamples
        )


/// Sum squared error.
module SSE =

    /// Calculates the sum squared error.
    /// Shapes: pred[smpl, ...], target[smpl, ...]
    let error (trgt: Tensor<'T>) (pred: Tensor<'T>) =
        let pred = pred |> HostTensor.transfer |> Tensor.float
        let trgt = trgt |> HostTensor.transfer |> Tensor.float

        (pred - trgt) ** 2.0
        |> Tensor.sum
        |> Tensor.value


/// Mean over samples of squared error.
module MSE =

    /// Calculates the mean squared error.
    /// The mean is taken over the samples. The error is summed over all
    /// other dimensions.
    /// Shapes: pred[smpl, ...], target[smpl, ...]
    let error (trgt: Tensor<'T>) (pred: Tensor<'T>) =
        SSE.error trgt pred / float trgt.Shape.[0]
        
    /// Calculates the MSE of a predictor on the training, validation and test sets.
    let ofPredictor (dataset: TrnValTst<'S>) batchSize 
            (trgtFn: 'S -> Tensor<'T>) (predFn: 'S -> Tensor<'T>) =
        dataset |> TrnValTst.apply (fun part ->
            part 
            |> Dataset.batches batchSize 
            |> Seq.map (fun batch -> SSE.error (trgtFn batch) (predFn batch))
            |> Seq.sum
            |> fun error -> error / float part.NSamples
        )


/// Root of mean over samples of squared error.
module RMSE =

    /// Calculates the root mean squared error.
    /// The mean is taken over the samples. The error is summed over all
    /// other dimensions.
    /// Shapes: pred[smpl, ...], target[smpl, ...]
    let error trgt pred =
        MSE.error trgt pred |> sqrt

    /// Calculates the RMSE of a predictor on the training, validation and test sets.
    let ofPredictor dataset batchSize trgtFn predFn =
        let trnMSE, valMSE, tstMSE = MSE.ofPredictor dataset batchSize trgtFn predFn
        sqrt trnMSE, sqrt valMSE, sqrt tstMSE


    
    
