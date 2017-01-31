namespace ArrayNDNS

open Basics


/// Matrix decomposition functions.
module Decomposition =
        
    /// Information about a performed PCA or ZCA.
    type PCAInfo<'T> = {
        /// the means of the features (for centering the data)
        Means:      ArrayNDHostT<'T>
        /// the variances of the principal components in descending order
        Variances:  ArrayNDHostT<'T>
        /// the axes corresponding to the principal components
        Axes:       ArrayNDHostT<'T>
    }

    /// Principal Component Analysis (PCA)
    type PCA() =

        /// Apply Principal Component Analysis (PCA) whitening. 
        /// `data` must be of the form [sample, feature].
        /// `nComps` optionally specifies how many components to keep.
        /// Returns a tensor of the form [sample, component].
        static member Perform (data: ArrayNDHostT<'T>, ?nComps) =
            if data.NDims <> 2 then
                invalidArg "data" "data must be a matrix"            
            let nFeatures = data.Shape.[1]
            let nComps = defaultArg nComps nFeatures
            if not (0L < nComps && nComps <= nFeatures) then
                invalidArg "nComps" "nComps must be between 0 and the number of features"
             
            // center data
            let means = data |> ArrayND.meanAxis 0
            let centered = data - means.[NewAxis, *] // centered[smpl, feature]

            // compute covariance matrix and its eigen decomposition
            let n = ArrayNDHost.scalar (conv<'T> data.Shape.[0])
            let cov = (ArrayND.transpose centered .* centered) / n 
            let variances, axes = ArrayND.symmetricEigenDecomposition cov 

            // sort axes by their variances in descending order
            let sortIdx = 
                variances 
                |> ArrayNDHost.toList 
                |> List.indexed 
                |> List.sortByDescending snd
                |> List.map fst
                |> List.map int64
                |> ArrayNDHost.ofList
            let variances = variances |> ArrayND.gather [Some sortIdx]
            let axesIdx = ArrayND.replicate 0 axes.Shape.[0] sortIdx.[NewAxis, *]
            let axes = axes |> ArrayND.gather [None; Some axesIdx]

            // limit number of components if desired
            let variances = variances.[0L .. nComps-1L]
            let axes = axes.[*, 0L .. nComps-1L]

            // transform data into new coordinate system
            // [smpl, feature] .* [feature, comp]
            let pcaed = centered .* axes
            
            // scale axes so that each has unit variance
            let whitened = pcaed / sqrt variances.[NewAxis, *]
            whitened, {Means=means; Variances=variances; Axes=axes}

        /// Reverses PCA whitening.
        /// `whitened` must be of the form [sample, component].
        static member Reverse (whitened: ArrayNDHostT<'T>, 
                               {Means=means; Variances=variances; Axes=axes}) =
            if whitened.NDims <> 2 then
                invalidArg "whitened" "whitened must be a matrix" 
            let pcaed = whitened * sqrt variances.[NewAxis, *]
            let centered = pcaed .* ArrayND.transpose axes // [smpl, comp] .* [comp, feature]
            centered + means.[NewAxis, *]


    /// ZCA whitening
    type ZCA() =

        /// Apply ZCA whitening. 
        /// `data` must be of the form [sample, feature].
        /// Returns a tensor of the form [sample, component].
        static member Perform (data: ArrayNDHostT<'T>) =        
            let whitened, info = PCA.Perform data
            whitened .* ArrayND.transpose info.Axes, info

        /// Reverses ZCA whitening.
        /// `whitened` must be of the form [sample, component].
        static member Reverse (zcaed: ArrayNDHostT<'T>, info: PCAInfo<'T>) =
            PCA.Reverse (zcaed .* info.Axes, info)
