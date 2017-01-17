module NormalizationTests

open Xunit
open FsUnit.Xunit
open System.IO

open Basics
open ArrayNDNS
open Datasets

type TestSample = {Data: ArrayNDT<double>}

type CurveNormalizationTests () =
    let dataFile = NPZFile.Open (Util.assemblyDirectory + "/../../TestData/PCA.npz")
    let data : ArrayNDHostT<double> = dataFile.Get "data"
    let refPCAWhitenedFull : ArrayNDHostT<double> = dataFile.Get "pca_whitened_full"
    let refPCAWhitened10 : ArrayNDHostT<double> = dataFile.Get "pca_whitened_10"
    let refZCAWhitened : ArrayNDHostT<double> = dataFile.Get "zca_whitened"
    let dataset = Dataset<TestSample> ([data])

    //do printfn "CurveNormalization Dataset shape: %A" dataset.All.Data.Shape

    [<Fact>]
    member this.``PCA Whitening Full`` () =
        let normalizers = [PCAWhitening None]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        use hdf = HDF5.OpenWrite "PCADebug.h5"
        ArrayNDHDF.write hdf "data" data
        ArrayNDHDF.write hdf "normalized" (normalized.All.Data :?> ArrayNDHostT<double>)
        ArrayNDHDF.write hdf "reversed" (reversed.All.Data :?> ArrayNDHostT<double>)
        ArrayNDHDF.write hdf "refPCAWhitenedFull" refPCAWhitenedFull

        ArrayND.almostEqualWithTol 1e-5 1e-4 dataset.All.Data reversed.All.Data |> ArrayND.value |> should equal true
        ArrayND.almostEqualWithTol 1e-5 1e-4 normalized.All.Data refPCAWhitenedFull |> ArrayND.value |> should equal true
        
    [<Fact>]
    member this.``PCA Whitening 10`` () =
        let normalizers = [PCAWhitening (Some 10)]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        use hdf = HDF5.OpenWrite "PCADebug2.h5"
        ArrayNDHDF.write hdf "data" data
        ArrayNDHDF.write hdf "normalized" (normalized.All.Data :?> ArrayNDHostT<double>)
        ArrayNDHDF.write hdf "reversed" (reversed.All.Data :?> ArrayNDHostT<double>)
        ArrayNDHDF.write hdf "refPCAWhitened10" refPCAWhitened10

        //ArrayND.almostEqualWithTol 1e-5 1e-4 dataset.All.Data reversed.All.Data |> ArrayND.value |> should equal true
        ArrayND.almostEqualWithTol 1e-5 1e-4 normalized.All.Data refPCAWhitened10 |> ArrayND.value |> should equal true

    [<Fact>]
    member this.``ZCA Whitening`` () =
        let normalizers = [ZCAWhitening]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        use hdf = HDF5.OpenWrite "ZCADebug.h5"
        ArrayNDHDF.write hdf "data" data
        ArrayNDHDF.write hdf "normalized" (normalized.All.Data :?> ArrayNDHostT<double>)
        ArrayNDHDF.write hdf "reversed" (reversed.All.Data :?> ArrayNDHostT<double>)
        ArrayNDHDF.write hdf "refZCAWhitened10" refZCAWhitened

        ArrayND.almostEqualWithTol 1e-5 1e-4 dataset.All.Data reversed.All.Data |> ArrayND.value |> should equal true
        ArrayND.almostEqualWithTol 1e-5 1e-4 normalized.All.Data refZCAWhitened |> ArrayND.value |> should equal true

    [<Fact>]
    member this.``Rescaling`` () =
        let normalizers = [Rescaling]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        let min = ArrayND.min normalized.All.Data 
        let max = ArrayND.max normalized.All.Data 
        
        ArrayND.almostEqualWithTol 1e-5 1e-4 min (ArrayNDHost.scalar 0.0) |> ArrayND.value |> should equal true
        ArrayND.almostEqualWithTol 1e-5 1e-4 max (ArrayNDHost.scalar 1.0) |> ArrayND.value |> should equal true
        ArrayND.almostEqualWithTol 1e-5 1e-4 dataset.All.Data reversed.All.Data |> ArrayND.value |> should equal true

    [<Fact>]
    member this.``Standardization`` () =
        let normalizers = [Standardization true]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        let means = ArrayND.meanAxis 0 normalized.All.Data 
        let stdevs = ArrayND.stdAxis 0 normalized.All.Data

        printfn "after standardization:"
        printfn "means=\n%A" means.Full
        printfn "stdevs=\n%A" stdevs.Full

        ArrayND.almostEqualWithTol 1e-3 1e-4 means (ArrayND.zerosLike means) |> ArrayND.value |> should equal true
        ArrayND.almostEqualWithTol 1e-3 1e-4 stdevs (ArrayND.onesLike stdevs) |> ArrayND.value |> should equal true
        ArrayND.almostEqualWithTol 1e-5 1e-4 dataset.All.Data reversed.All.Data |> ArrayND.value |> should equal true

