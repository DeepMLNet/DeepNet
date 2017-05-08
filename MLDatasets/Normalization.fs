namespace Datasets

open Basics
open Tensor
open Util


[<AutoOpen>]
/// Dataset normalization types.
module NormalizationTypes =

    /// normalization operation to perform
    type Normalizer =
        /// no normalization
        | NoNormalizer
        /// rescale the range of every feature to [0, 1]
        | Rescaling
        /// Make each feature have zero mean and unit variance.
        /// If `keepZeroOne` is true, then features that contain only the values 0 and 1 are left untouched.
        | Standardization of keepZeroOne:bool
        /// scale the feature vector so that it has L2-norm one
        | ScaleToUnitLength
        /// Apply Principal Component Analysis (PCA) whitening. 
        /// Optionally specify how many components to keep.
        | PCAWhitening of nComponents:int64 option
        /// Apply ZCA whitening.
        /// Optionally specify how many dimensions to keep.
        | ZCAWhitening 

    /// type-neutral interface for Normalization<'T>
    type INormalization =
        interface end

    /// performed normalization operation
    type Normalization<'T> =
        | NotNormalized
        | Rescaled of minVals:Tensor<'T> * maxVals:Tensor<'T>
        | Standardized of means:Tensor<'T> * stds:Tensor<'T> * onlyZeroOne:Tensor<bool> option
        | ScaledToUnitLength of lengths:Tensor<'T> 
        | PCAWhitened of Decomposition.PCAInfo<'T>
        | ZCAWhitened of Decomposition.PCAInfo<'T>

        interface INormalization


/// Dataset normalization functions.
module Normalization =

    let private performField normalizer (data: Tensor<'T>)  =
        let epsilon = HostTensor.scalar (conv<'T> 1e-5)
        match normalizer with 
        | NoNormalizer ->
            NotNormalized, data
        | Rescaling -> 
            let minVals = data |> Tensor.minAxis 0
            let maxVals = data |> Tensor.maxAxis 0
            let maxVals = Tensor.maxElemwise maxVals (minVals + epsilon)
            Rescaled (minVals, maxVals), (data - minVals.[NewAxis, *]) / (maxVals - minVals).[NewAxis, *]
        | Standardization keepZeroOne ->
            let zero = HostTensor.scalar (conv<'T> 0)
            let one = HostTensor.scalar (conv<'T> 1)
            let means = data |> Tensor.meanAxis 0
            let stds = Tensor.stdAxis(0, data) + epsilon
            let standardized = (data - means.[NewAxis, *]) / stds.[NewAxis, *]
            let res, onlyZeroOne = 
                if keepZeroOne then
                    let onlyZeroOne = (data ==== zero) |||| (data ==== one) |> Tensor.allAxis 0
                    Tensor.ifThenElse onlyZeroOne.[NewAxis, *] data standardized, Some onlyZeroOne
                else standardized, None
            Standardized (means, stds, onlyZeroOne), res
        | ScaleToUnitLength ->
            let lengths = Tensor.normAxis(1, data) + epsilon
            ScaledToUnitLength lengths, data / lengths.[*, NewAxis]
        | PCAWhitening nComps ->
            let whitened, info = Decomposition.PCA.Perform (data, ?nComps=nComps)
            PCAWhitened info, whitened
        | ZCAWhitening ->
            let whitened, info = Decomposition.ZCA.Perform data
            ZCAWhitened info, whitened

    let private reverseField normalization (nData: Tensor<'T>) =
        match normalization with
        | NotNormalized ->
            nData
        | Rescaled (minVals, maxVals) ->
            nData * (maxVals - minVals).[NewAxis, *] + minVals.[NewAxis, *]
        | Standardized (means, stds, onlyZeroOne) ->
            let unstd = nData * stds.[NewAxis, *] + means.[NewAxis, *]
            match onlyZeroOne with
            | Some onlyZeroOne -> Tensor.ifThenElse onlyZeroOne.[NewAxis, *] nData unstd
            | None -> unstd            
        | ScaledToUnitLength lengths ->
            nData * lengths.[*, NewAxis]
        | PCAWhitened info ->
            Decomposition.PCA.Reverse (nData, info)
        | ZCAWhitened info ->
            Decomposition.ZCA.Reverse (nData, info)

    let private performFieldUntyped n (fs: ITensor) =
        match fs with
        | :? Tensor<single> as fs -> 
            let info, res = performField n fs in info :> INormalization, res :> ITensor
        | :? Tensor<double> as fs -> 
            let info, res = performField n fs in info :> INormalization, res :> ITensor
        | _ -> failwithf "normalization requires single or double data type"

    let private reverseFieldUntyped (n: INormalization) (fs: ITensor) =
        match fs with
        | :? Tensor<single> as fs -> 
            reverseField (n :?> Normalization<single>) fs :> ITensor
        | :? Tensor<double> as fs -> 
            reverseField (n :?> Normalization<double>) fs :> ITensor
        | _ -> failwithf "unnormalization requires single or double data type"

    /// Normalizes each field of the specified Dataset using the specified normalizier.
    let perform (normalizers: Normalizer list) (dataset: Dataset<'S>) =
        if normalizers.Length <> dataset.FieldStorages.Length then
            failwith "normalization requires one normalizer per field of dataset"

        let infos, nfs =            
            List.zip normalizers dataset.FieldStorages
            |> List.map (fun (n, fs) -> performFieldUntyped n fs)
            |> List.unzip
        infos, Dataset<'S> (nfs, dataset.IsSeq)
        
    /// Reverses the normalization performed by the 'perform' function.
    let reverse (normalizations: INormalization list) (dataset: Dataset<'S>) =
        if normalizations.Length <> dataset.FieldStorages.Length then
            failwith "reversation of normalization requires one normalization info per field of dataset"
        let fs =
            List.zip normalizations dataset.FieldStorages
            |> List.map (fun (info, fs) -> reverseFieldUntyped info fs)
        Dataset<'S> (fs, dataset.IsSeq)            


