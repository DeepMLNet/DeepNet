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
        /// make each feature have zero mean and unit variance
        | Standardization
        /// scale the feature vector so that it has L2-norm one
        | ScaleToUnitLength
        /// Apply Principal Component Analysis (PCA) whitening. 
        /// Optionally specify how many components to keep.
        | PCAWhitening of nComponents:int option
        /// Apply ZCA whitening.
        /// Optionally specify how many dimensions to keep.
        | ZCAWhitening of nComponents:int option

    /// performed normalization operation
    type Normalization =
        | NotNormalized
        | Rescaled of (single * single) list
        | Standardized of (single * single) list
        | ScaledToUnitLength of single list
        | PCAWhitened of means:ArrayNDHostT<single> * variances:ArrayNDHostT<single> * 
                         axes:ArrayNDHostT<single> 
        | ZCAWhitened of means:ArrayNDHostT<single> * variances:ArrayNDHostT<single> * 
                         axes:ArrayNDHostT<single> 

/// Dataset normalization functions.
module Normalization =

    let rec private performField normalizer (data: ArrayNDHostT<single>)  =
        match normalizer with 
        | NoNormalizer ->
            NotNormalized, data
        | Rescaling -> // TODO: change to broadcasting
            let nData = ArrayNDHost.zeros data.Shape
            let info = List.init data.Shape.[1] (fun f ->
                let minVal = data.[*, f] |> ArrayND.min |> ArrayND.value
                let maxVal = data.[*, f] |> ArrayND.max |> ArrayND.value
                nData.[*, f] <- (data.[*, f] - minVal) / (maxVal - minVal)
                minVal, maxVal)
            Rescaled info, nData
        | Standardization ->
            let nData = ArrayNDHost.zeros data.Shape
            let info = List.init data.Shape.[1] (fun f ->
                let mean = data.[*, f] |> ArrayND.mean |> ArrayND.value
                let std = data.[*, f] |> ArrayND.std |> ArrayND.value
                nData.[*, f] <- (data.[*, f] - mean) / std
                mean, std)
            Standardized info, nData        
        | ScaleToUnitLength ->
            let nData = ArrayNDHost.zeros data.Shape
            let lengths = List.init data.Shape.[0] (fun s ->
                let length = data.[s, *] |> ArrayND.norm |> ArrayND.value
                nData.[s, *] <- data.[s, *] / length
                length)
            ScaledToUnitLength lengths, nData
        | PCAWhitening nComps ->
            // center data
            let means = data |> ArrayND.meanAxis 0
            let centered = data - means.[NewAxis, *] // centered[smpl, feature]

            // compute covariance matrix and its eigen decomposition
            let cov = (ArrayND.transpose centered .* centered) / (single data.Shape.[0])
            let variances, axes = ArrayND.symmetricEigenDecomposition cov 

            // sort axes by their variances in descending order
            let sortIdx = 
                variances 
                |> ArrayNDHost.toList 
                |> List.indexed 
                |> List.sortByDescending snd
                |> List.map fst
                |> ArrayNDHost.ofList
            let variances = variances |> ArrayND.gather [Some sortIdx]
            let axes = axes |> ArrayND.gather [None; Some sortIdx]

            // limit number of components if desired
            let variances, axes =
                match nComps with
                | Some nComps -> variances.[0 .. nComps-1], axes.[*, 0 .. nComps-1]
                | None -> variances, axes

            // transform data into new coordinate system
            // [smpl, feature] .* [feature, comp]
            let pcaed = centered .* axes
            
            // scale axes so that each has unit variance
            let whitened = pcaed / sqrt variances.[NewAxis, *]
            PCAWhitened (means, variances, axes), whitened
        | ZCAWhitening nComps ->
            match performField (PCAWhitening nComps) data with
            | PCAWhitened (means, variances, axes), whitened ->
                let zcaed = whitened .* ArrayND.transpose axes  
                ZCAWhitened (means, variances, axes), zcaed
            | _ -> failwith "impossible"

    let rec private reverseField normalization (nData: ArrayNDHostT<single>) =
        match normalization with
        | NotNormalized ->
            nData
        | Rescaled info ->
            let data = ArrayNDHost.zeros nData.Shape
            for f, (minVal, maxVal) in List.indexed info do
                data.[*, f] <- nData.[*, f] * (maxVal - minVal) + minVal
            data
        | Standardized info ->
            let data = ArrayNDHost.zeros nData.Shape
            for f, (mean, std) in List.indexed info do
                data.[*, f] <- nData.[*, f] * std + mean
            data
        | ScaledToUnitLength lengths ->
            let data = ArrayNDHost.zeros nData.Shape
            for s, length in List.indexed lengths do
                data.[s, *] <- nData.[s, *] * length
            data
        | PCAWhitened (means, variances, axes) ->
            let whitened = nData
            let pcaed = whitened * sqrt variances.[NewAxis, *]
            let centered = pcaed .* ArrayND.transpose axes // [smpl, comp] .* [comp, feature]
            let data = centered + means.[NewAxis, *]
            data
        | ZCAWhitened (means, variances, axes) ->
            let zcaed = nData
            let whitened = zcaed .* axes
            reverseField (PCAWhitened (means, variances, axes)) whitened

    let private extractFieldStorage (fs: IArrayNDT) =
        match fs with
        | :? ArrayNDHostT<single> as fs -> fs
        | _ -> failwithf "normalization requires a dataset stored in CPU memory with single data type"

    /// Normalizes each field of the specified Dataset using the specified normalizier.
    let perform (normalizers: Normalizer list) (dataset: Dataset<'S>) =
        let infos, nfs =            
            List.zip normalizers dataset.FieldStorages
            |> List.map (fun (n, fs) -> 
                let info, nfs = fs |> extractFieldStorage |> performField n
                info, nfs :> IArrayNDT)
            |> List.unzip
        infos, Dataset<'S> (nfs, dataset.IsSeq)
        
    /// Reverses the normalization performed by the 'perform' function.
    let reverse (normalizations: Normalization list) (dataset: Dataset<'S>) =
        let fs =
            List.zip normalizations dataset.FieldStorages
            |> List.map (fun (info, fs) -> fs |> extractFieldStorage |> reverseField info :> IArrayNDT)
        Dataset<'S> (fs, dataset.IsSeq)            


