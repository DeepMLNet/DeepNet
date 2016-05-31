module MNISTTests

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open Datasets

let mnistPath = Util.assemblyDirectory + "../../../../Data/MNIST"

[<Fact>]
let ``Load MNIST`` () =
    let mnist = Mnist.load mnistPath
    printfn "MNIST shape: %A" (ArrayND.shape mnist.TrnImgs)

[<Fact>]
let ``Save MNIST as HDF5`` () =
    let mnist = Mnist.load mnistPath

    use hdf = new HDF5 (@"MNIST-TestWrite.h5", HDF5Overwrite)
    ArrayNDHDF.write hdf "TrnImgs" (mnist.TrnImgs :?> ArrayNDHostT<single>)
    ArrayNDHDF.write hdf "TrnLbls" (mnist.TrnLbls :?> ArrayNDHostT<single>)
    ArrayNDHDF.write hdf "TstImgs" (mnist.TstImgs :?> ArrayNDHostT<single>)
    ArrayNDHDF.write hdf "TstLbls" (mnist.TstLbls :?> ArrayNDHostT<single>)