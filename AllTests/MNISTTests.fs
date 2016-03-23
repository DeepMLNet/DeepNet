module MNISTTests

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open Datasets

let mnistPath = @"C:\Local\surban\dev\fexpr\Data\MNIST"

[<Fact>]
let ``Load MNIST`` () =
    let mnist = Mnist.load mnistPath
    printfn "MNIST shape: %A" (ArrayND.shape mnist.TrnImgs)

[<Fact>]
let ``Save MNIST as HDF5`` () =
    let mnist = Mnist.load mnistPath

    use hdf = new HDF5 (@"MNIST-TestWrite.h5", HDF5Overwrite)
    ArrayNDHDF.write hdf "TrnImgs" mnist.TrnImgs
    ArrayNDHDF.write hdf "TrnLbls" mnist.TrnLbls
    ArrayNDHDF.write hdf "TstImgs" mnist.TstImgs
    ArrayNDHDF.write hdf "TstLbls" mnist.TstLbls