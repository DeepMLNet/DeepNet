module NormalizationTests

open Xunit
open FsUnit.Xunit
open Xunit.Abstractions
open System.IO

open DeepNet.Utils
open Tensor
open Tensor.Algorithm
open Tensor.Expr.ML


type TestSample = {Data: Tensor<double>}

type CurveNormalizationTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let dataFile = NPZFile.Open (Util.assemblyDir + "/TestData/PCA.npz")
    let data : Tensor<double> = dataFile.Get "data"
    let refPCAWhitenedFull : Tensor<double> = dataFile.Get "pca_whitened_full"
    let refPCAWhitened10 : Tensor<double> = dataFile.Get "pca_whitened_10"
    let refZCAWhitened : Tensor<double> = dataFile.Get "zca_whitened"
    let dataset = Dataset<TestSample> ([data])

    //do printfn "CurveNormalization Dataset shape: %A" dataset.All.Data.Shape

    [<Fact>]
    member this.``PCA Whitening Full`` () =
        let normalizers = [PCAWhitening None]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        use hdf = HDF5.OpenWrite "PCADebug.h5"
        HostTensor.write hdf "data" data
        HostTensor.write hdf "normalized" normalized.All.Data
        HostTensor.write hdf "reversed" reversed.All.Data 
        HostTensor.write hdf "refPCAWhitenedFull" refPCAWhitenedFull

        // PCA axes are not sign unique.
        Tensor.almostEqual (dataset.All.Data, reversed.All.Data, absTol=1e-5, relTol=1e-4) |> should equal true
        Tensor.almostEqual (abs normalized.All.Data, abs refPCAWhitenedFull, absTol=1e-5, relTol=1e-4) |> should equal true
        
    [<Fact>]
    member this.``PCA Whitening 10`` () =
        let normalizers = [PCAWhitening (Some 10L)]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        use hdf = HDF5.OpenWrite "PCADebug2.h5"
        HostTensor.write hdf "data" data
        HostTensor.write hdf "normalized" normalized.All.Data
        HostTensor.write hdf "reversed" reversed.All.Data
        HostTensor.write hdf "refPCAWhitened10" refPCAWhitened10

        // PCA axes are not sign unique.
        //ArrayND.almostEqualWithTol 1e-5 1e-4 dataset.All.Data reversed.All.Data |> ArrayND.value |> should equal true
        Tensor.almostEqual (abs normalized.All.Data, abs refPCAWhitened10, absTol=1e-5, relTol=1e-4) |> should equal true

    [<Fact>]
    member this.``ZCA Whitening`` () =
        let normalizers = [ZCAWhitening]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        use hdf = HDF5.OpenWrite "ZCADebug.h5"
        HostTensor.write hdf "data" data
        HostTensor.write hdf "normalized" normalized.All.Data 
        HostTensor.write hdf "reversed" reversed.All.Data 
        HostTensor.write hdf "refZCAWhitened10" refZCAWhitened

        Tensor.almostEqual (dataset.All.Data, reversed.All.Data, absTol=1e-5, relTol=1e-4) |> should equal true
        Tensor.almostEqual (normalized.All.Data, refZCAWhitened, absTol=1e-5, relTol=1e-4) |> should equal true

    [<Fact>]
    member this.``Rescaling`` () =
        let normalizers = [Rescaling]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        let min = Tensor.minTensor normalized.All.Data 
        let max = Tensor.maxTensor normalized.All.Data 
        
        Tensor.almostEqual (min, HostTensor.scalar 0.0, absTol=1e-5, relTol=1e-4) |> should equal true
        Tensor.almostEqual (max, HostTensor.scalar 1.0, absTol=1e-5, relTol=1e-4) |> should equal true
        Tensor.almostEqual (dataset.All.Data, reversed.All.Data, absTol=1e-5, relTol=1e-4) |> should equal true

    [<Fact>]
    member this.``Standardization`` () =
        let normalizers = [Standardization true]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        let means = Tensor.meanAxis 0 normalized.All.Data 
        let stdevs = Tensor.stdAxis (0, normalized.All.Data)

        printfn "after standardization:"
        printfn "means=\n%A" means.Full
        printfn "stdevs=\n%A" stdevs.Full

        Tensor.almostEqual (means, Tensor.zerosLike means, absTol=1e-3, relTol=1e-4) |> should equal true
        Tensor.almostEqual (stdevs, Tensor.onesLike stdevs, absTol=1e-3, relTol=1e-4) |> should equal true
        Tensor.almostEqual (dataset.All.Data, reversed.All.Data, absTol=1e-3, relTol=1e-4) |> should equal true

    [<Fact>]
    member this.``ScaleToUnitLength`` () =
        let normalizers = [ScaleToUnitLength]
        let infos, normalized = dataset |> Normalization.perform normalizers
        let reversed = normalized |> Normalization.reverse infos

        let lengths = Tensor.normAxis (1, normalized.All.Data)

        printfn "after standardization:"
        printfn "lengths=\n%A" lengths.Full

        Tensor.almostEqual (lengths, Tensor.onesLike lengths, absTol=1e-3, relTol=1e-4) |> should equal true
        Tensor.almostEqual (dataset.All.Data, reversed.All.Data, absTol=1e-3, relTol=1e-4) |> should equal true