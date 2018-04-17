module MNISTTests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor
open Datasets

let mnistPath = Util.assemblyDirectory + "../../../../Data/MNIST"

[<Fact>]
let ``Load MNIST`` () =
    let mnist = Mnist.loadRaw mnistPath
    printfn "MNIST shape: %A" (Tensor.shape mnist.TrnImgs)

[<Fact>]
let ``Save MNIST as HDF5`` () =
    let mnist = Mnist.loadRaw mnistPath

    use hdf = new HDF5 (@"MNIST-TestWrite.h5", HDF5Overwrite)
    HostTensor.write hdf "TrnImgs" mnist.TrnImgs
    HostTensor.write hdf "TrnLbls" mnist.TrnLbls
    HostTensor.write hdf "TstImgs" mnist.TstImgs
    HostTensor.write hdf "TstLbls" mnist.TstLbls

