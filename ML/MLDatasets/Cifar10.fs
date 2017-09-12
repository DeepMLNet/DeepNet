namespace Datasets

open System
open System.IO
open System.IO.Compression
open Tensor.Utils
open Tensor

/// Raw MNIST dataset
type CifarRawT = {
    /// 2d training image data of shape [50000;3072]
    TrnData:      Tensor<single>
    /// one-hot training labels of shape [50000; 10]
    TrnLbls:      Tensor<single>

    /// 2d test images data of shape [10000;3072]
    TstData:      Tensor<single>
    /// one-hot test labels of shape [10000; 10]
    TstLbls:      Tensor<single>   
} with 
    /// copies this dataset to the CUDA GPU
    member this.ToCuda () =
        {TrnImgs = this.TrnData |> CudaTensor.transfer
         TrnLbls = this.TrnLbls |> CudaTensor.transfer
         TstImgs = this.TstData |> CudaTensor.transfer
         TstLbls = this.TstLbls |> CudaTensor.transfer}

/// Module containing functions to load the Cifar 10 dataset from the binary version.
/// The dataset can be found at https://www.cs.toronto.edu/~kriz/cifar.html
module Cifar10 =
    
    let private assemble dataSeq =
        let data = List.ofSeq dataSeq
        let nSamples = List.length data |> int64

        let dataShape = Tensor.shape data.[0]
        let ds = HostTensor.zeros (nSamples :: dataShape)

        data |> List.iteri (fun smpl d -> ds.[int64 smpl, Fill] <- d)
        ds

    let private sampleList (directory:string) (batchName:String) =
        let zipPath = Path.Combine (directory, "cifar-10-binary.tar.gz")
        let archive = ZipFile.OpenRead(zipPath)
        let batch = archive.GetEntry(batchName)
        let batchStream = batch.Open()
        let batchReader = new BinaryReader(batchStream)
        let nSamples = 10000
        let imgSize =  (32 * 32 * 3)
        let sampleList = [1..nSamples] 
                         |> List.map (fun x ->
                                let label = batchReader.ReadByte ()
                                let image = batchReader.ReadBytes imgSize
                                HostTensor.scalar (single label), HostTensor.ofArray(Array.map single image)
                                )
        sampleList

    let private doLoadRaw (directory:string) =
        let trnLbls,trnData = sampleList directory "data_batch_1.bin"
                                |> List.append (sampleList directory "data_batch_2.bin")
                                |> List.append (sampleList directory "data_batch_3.bin")
                                |> List.append (sampleList directory "data_batch_4.bin")
                                |> List.append (sampleList directory "data_batch_5.bin")
                                |> List.unzip
                                
        let tstLbls,tstData = sampleList directory "test_batch.bin"
                                |> List.unzip

        {TrnData = trnData |> assemble; TrnLbls = trnLbls |> assemble
         TstData = tstData |> assemble; TstLbls = tstLbls |> assemble}
    
    /// Loads the Cifar10 dataset and returns it as type MnistRawT.
    /// Use only if you need raw access to the Cifar10 data.
    let loadRaw directory =
        let hdfPath = Path.Combine (directory,"Cifar10")
        if File.Exists hdfPath then
            use hdf = new HDF5 (hdfPath, HDF5Read)
            {TrnData = HostTensor.read hdf "TrnData"; 
             TrnLbls = HostTensor.read hdf "TrnLbls";
             TstData = HostTensor.read hdf "TstData"; 
             TstLbls = HostTensor.read hdf "TstLbls";}
        else
            printf "Converting Cifar10 to HDF5..."
            let cifar10 = doLoadRaw directory
            use hdf = new HDF5 (hdfPath, HDF5Overwrite)
            HostTensor.write hdf "TrnImgs" cifar10.TrnData 
            HostTensor.write hdf "TrnLbls" cifar10.TrnLbls 
            HostTensor.write hdf "TstImgs" cifar10.TstData 
            HostTensor.write hdf "TstLbls" cifar10.TstLbls 
            printfn "Done."
            cifar10

    /// Loads the Cifar10 dataset and splits the original training set into
    /// a training and validation set using the ratio `valRatio` (between 0 and 1)
    /// for the validation set.
    let load directory valRatio =
        if not (0.0 <= valRatio && valRatio <= 1.0) then
            invalidArg "valRatio" "valRatio must be between 0.0 and 1.0"
        
        let raw = loadRaw directory
        let trnData = raw.TrnData 
        let tstData = raw.TstData 

//        let orgTrn = Dataset<MnistT> [trnImgsFlat; raw.TrnLbls]
        let orgTrn = Dataset<InputTargetSampleT> [trnData; raw.TrnLbls]

        let trn, vali =
            match orgTrn |> Dataset.partition [1. - valRatio; valRatio] with
            | [trn; vali] -> trn, vali
            | _ -> failwith "impossible"
//        let tst = Dataset<MnistT> [tstImgsFlat; raw.TstLbls]
        let tst = Dataset<InputTargetSampleT> [tstData; raw.TstLbls]


        {Trn=trn; Val=vali; Tst=tst}