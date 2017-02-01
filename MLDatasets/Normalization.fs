namespace Datasets

open Basics
open ArrayNDNS
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
        | Rescaled of minVals:ArrayNDHostT<'T> * maxVals:ArrayNDHostT<'T>
        | Standardized of means:ArrayNDHostT<'T> * stds:ArrayNDHostT<'T> * onlyZeroOne:ArrayNDHostT<bool> option
        | ScaledToUnitLength of lengths:ArrayNDHostT<'T> 
        | PCAWhitened of Decomposition.PCAInfo<'T>
        | ZCAWhitened of Decomposition.PCAInfo<'T>

        interface INormalization


/// Dataset normalization functions.
module Normalization =

    let private performField normalizer (data: ArrayNDHostT<'T>)  =
        let epsilon = ArrayNDHost.scalar (conv<'T> 1e-5)
        match normalizer with 
        | NoNormalizer ->
            NotNormalized, data
        | Rescaling -> 
            let minVals = data |> ArrayND.minAxis 0
            let maxVals = data |> ArrayND.maxAxis 0
            let maxVals = ArrayND.maxElemwise maxVals (minVals + epsilon)
            Rescaled (minVals, maxVals), (data - minVals.[NewAxis, *]) / (maxVals - minVals).[NewAxis, *]
        | Standardization keepZeroOne ->
            let zero = ArrayND.scalarOfSameType data (conv<'T> 0)
            let one = ArrayND.scalarOfSameType data (conv<'T> 1)
            let means = data |> ArrayND.meanAxis 0
            let stds = (data |> ArrayND.stdAxis 0) + epsilon
            let standardized = (data - means.[NewAxis, *]) / stds.[NewAxis, *]
            let res, onlyZeroOne = 
                if keepZeroOne then
                    let onlyZeroOne = (data ==== zero) |||| (data ==== one) |> ArrayND.allAxis 0
                    ArrayND.ifThenElse onlyZeroOne.[NewAxis, *] data standardized, Some onlyZeroOne
                else standardized, None
            Standardized (means, stds, onlyZeroOne), res
        | ScaleToUnitLength ->
            let lengths = (data |> ArrayND.normAxis 1) + epsilon
            ScaledToUnitLength lengths, data / lengths.[*, NewAxis]
        | PCAWhitening nComps ->
            let whitened, info = Decomposition.PCA.Perform (data, ?nComps=nComps)
            PCAWhitened info, whitened
        | ZCAWhitening ->
            let whitened, info = Decomposition.ZCA.Perform data
            ZCAWhitened info, whitened

    let private reverseField normalization (nData: ArrayNDHostT<'T>) =
        match normalization with
        | NotNormalized ->
            nData
        | Rescaled (minVals, maxVals) ->
            nData * (maxVals - minVals).[NewAxis, *] + minVals.[NewAxis, *]
        | Standardized (means, stds, onlyZeroOne) ->
            let unstd = nData * stds.[NewAxis, *] + means.[NewAxis, *]
            match onlyZeroOne with
            | Some onlyZeroOne -> ArrayND.ifThenElse onlyZeroOne.[NewAxis, *] nData unstd
            | None -> unstd            
        | ScaledToUnitLength lengths ->
            nData * lengths.[*, NewAxis]
        | PCAWhitened info ->
            Decomposition.PCA.Reverse (nData, info)
        | ZCAWhitened info ->
            Decomposition.ZCA.Reverse (nData, info)

    let private performFieldUntyped n (fs: IArrayNDT) =
        match fs with
        | :? ArrayNDHostT<single> as fs -> 
            let info, res = performField n fs in info :> INormalization, res :> IArrayNDT
        | :? ArrayNDHostT<double> as fs -> 
            let info, res = performField n fs in info :> INormalization, res :> IArrayNDT
        | _ -> failwithf "normalization requires a dataset stored in CPU memory"

    let private reverseFieldUntyped (n: INormalization) (fs: IArrayNDT) =
        match fs with
        | :? ArrayNDHostT<single> as fs -> 
            reverseField (n :?> Normalization<single>) fs :> IArrayNDT
        | :? ArrayNDHostT<double> as fs -> 
            reverseField (n :?> Normalization<double>) fs :> IArrayNDT
        | _ -> failwithf "normalization requires a dataset stored in CPU memory"

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


