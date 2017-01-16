module NormalizationTests

open Xunit
open FsUnit.Xunit
open System.IO

open Basics
open ArrayNDNS
open Datasets

type TestSample = {Data: ArrayNDT<single>}

type CurveNormalizationTests () =
    let dataFile = NPZFile.Open (Util.assemblyDirectory + "/../../TestData/PCA.npz")
    let data : ArrayNDHostT<single> = dataFile.Get "data"
    let refPCAWhitenedFull : ArrayNDHostT<single> = dataFile.Get "pca_whitened_full"
    let refPCAWhitened10 : ArrayNDHostT<single> = dataFile.Get "pca_whitened_10"
    let refZCAWhitened : ArrayNDHostT<single> = dataFile.Get "zca_whitened"
    let dataset = Dataset<TestSample> ([data])

    do printfn "CurveNormalization Dataset shape: %A" dataset.All.Data.Shape

    [<Fact>]
    member this.``PCA Whitening Full`` () =
        let normalizers = [PCAWhitening None]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        use hdf = HDF5.OpenWrite "PCADebug.h5"
        ArrayNDHDF.write hdf "data" data
        ArrayNDHDF.write hdf "normalized" (normalized.All.Data :?> ArrayNDHostT<single>)
        ArrayNDHDF.write hdf "reversed" (reversed.All.Data :?> ArrayNDHostT<single>)
        ArrayNDHDF.write hdf "refPCAWhitenedFull" refPCAWhitenedFull

        //ArrayND.almostEqualWithTol 1e-5f 1e-4f dataset.All.Data reversed.All.Data |> should equal true
        ArrayND.almostEqualWithTol 1.0f 1e-5f normalized.All.Data refPCAWhitenedFull |> should equal true
        

