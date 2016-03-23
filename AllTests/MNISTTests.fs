module MNISTTests

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open Datasets


[<Fact>]
let ``Load MNIST`` () =
    let mnist = Mnist.load @"..\..\..\Data\MNIST"
    printfn "MNIST shape: %A" (ArrayND.shape mnist.TrnImgs)

[<Fact>]
let ``Save MNIST as HDF5`` () =
    let mnist = Mnist.load @"..\..\..\Data\MNIST"

    use hdf = new HDF5 (@"MNIST-TestWrite.h5", HDF5Overwrite)
    ArrayNDHDF.write hdf "TrnImgs" mnist.TrnImgs
    ArrayNDHDF.write hdf "TrnLbls" mnist.TrnLbls
    ArrayNDHDF.write hdf "TstImgs" mnist.TstImgs
    ArrayNDHDF.write hdf "TstLbls" mnist.TstLbls