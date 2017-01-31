namespace Datasets

open System
open System.IO
open System.IO.Compression

open Basics
open ArrayNDNS


/// Raw MNIST dataset
type MnistRawT = {
    /// 2d training images of shape [60000; 28; 28]
    TrnImgs:      ArrayNDT<single>
    /// one-hot training labels of shape [60000; 10]
    TrnLbls:      ArrayNDT<single>

    /// 2d test images of shape [10000; 28; 28]
    TstImgs:      ArrayNDT<single>
    /// one-hot test labels of shape [10000; 10]
    TstLbls:      ArrayNDT<single>   
} with 
    /// copies this dataset to the CUDA GPU
    member this.ToCuda () =
        {TrnImgs = this.TrnImgs :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TrnLbls = this.TrnLbls :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TstImgs = this.TstImgs :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TstLbls = this.TstLbls :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev}


/// Module containing functions to load the MNIST dataset.
module Mnist = 

    [<Literal>]
    let private TestDataset = false

    let private assemble dataSeq =
        let data = List.ofSeq dataSeq
        let nSamples = List.length data |> int64

        let dataShape = ArrayND.shape data.[0]
        let ds = ArrayNDHost.zeros (nSamples :: dataShape)

        data |> List.iteri (fun smpl d -> ds.[int64 smpl, Fill] <- d)
        ds

    let private swapEndians (value: int32) =
        let bytes = BitConverter.GetBytes(value)
        let revBytes = Array.rev bytes
        BitConverter.ToInt32 (revBytes, 0)

    let private sampleSeq labelPath imagePath = seq {
        use imageGzStream = File.OpenRead imagePath
        use labelGzStream = File.OpenRead labelPath
        use imageStream = new GZipStream (imageGzStream, CompressionMode.Decompress)
        use labelStream = new GZipStream (labelGzStream, CompressionMode.Decompress)
        use imageReader = new BinaryReader (imageStream)
        use labelReader = new BinaryReader (labelStream)

        if labelReader.ReadInt32 () |> swapEndians <> 2049 then failwith "invalid MNIST label file"
        if imageReader.ReadInt32 () |> swapEndians <> 2051 then failwith "invalid MNIST image file"

        let nSamples = labelReader.ReadInt32() |> swapEndians
        if imageReader.ReadInt32() |> swapEndians <> nSamples then failwith "number of samples mismatch in MNIST"

        let nRows = imageReader.ReadInt32() |> swapEndians 
        let nCols = imageReader.ReadInt32() |> swapEndians 

        let nSamples = 
            if TestDataset then min 2000 nSamples
            else nSamples

        for smpl in 0 .. nSamples - 1 do
            let label = labelReader.ReadByte() |> int64
            let labelHot : ArrayNDHostT<single> = ArrayNDHost.zeros [10L];
            labelHot.[[label]] <- 1.0f

            let image = imageReader.ReadBytes (nRows * nCols)           
            let imageSingle = Array.map (fun p -> single p / 255.0f) image
            let imageMat = ArrayNDHost.ofArray imageSingle |> ArrayND.reshape [int64 nRows; int64 nCols]

            yield labelHot, imageMat
    }
        
    let private dataset labelPath imagePath =
        let labelSeq, imageSeq = sampleSeq labelPath imagePath |> Seq.toList |> List.unzip
        assemble labelSeq, assemble imageSeq    

    let private doLoadRaw directory =
        let trnLbls, trnImgs = 
            dataset (Path.Combine (directory, "train-labels-idx1-ubyte.gz")) 
                    (Path.Combine (directory, "train-images-idx3-ubyte.gz"))
        let tstLbls, tstImgs = 
            dataset (Path.Combine (directory, "t10k-labels-idx1-ubyte.gz")) 
                    (Path.Combine (directory, "t10k-images-idx3-ubyte.gz"))
    
        {TrnImgs = trnImgs; TrnLbls = trnLbls
         TstImgs = tstImgs; TstLbls = tstLbls}

    /// Loads the MNIST dataset and returns it as type MnistRawT.
    /// Use only if you need raw access to the MNIST data.
    let loadRaw directory =
        let testStr = if TestDataset then "-Test" else ""
        let hdfPath = Path.Combine (directory, sprintf "MNIST%s.h5" testStr)
        if File.Exists hdfPath then
            use hdf = new HDF5 (hdfPath, HDF5Read)
            {TrnImgs = ArrayNDHDF.read hdf "TrnImgs"; 
             TrnLbls = ArrayNDHDF.read hdf "TrnLbls";
             TstImgs = ArrayNDHDF.read hdf "TstImgs"; 
             TstLbls = ArrayNDHDF.read hdf "TstLbls";}
        else
            printf "Converting MNIST to HDF5..."
            let mnist = doLoadRaw directory
            use hdf = new HDF5 (hdfPath, HDF5Overwrite)
            ArrayNDHDF.write hdf "TrnImgs" (mnist.TrnImgs :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TrnLbls" (mnist.TrnLbls :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TstImgs" (mnist.TstImgs :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TstLbls" (mnist.TstLbls :?> ArrayNDHostT<single>)
            printfn "Done."
            mnist

    /// Loads the MNIST dataset and splits the original training set into
    /// a training and validation set using the ratio `valRatio` (between 0 and 1)
    /// for the validation set.
    let load directory valRatio =
        if not (0.0 <= valRatio && valRatio <= 1.0) then
            invalidArg "valRatio" "valRatio must be between 0.0 and 1.0"
        
        let raw = loadRaw directory
        let trnImgsFlat = raw.TrnImgs |> ArrayND.reshape [raw.TrnImgs.Shape.[0]; -1L]
        let tstImgsFlat = raw.TstImgs |> ArrayND.reshape [raw.TstImgs.Shape.[0]; -1L]
        let orgTrn = Dataset<InputTargetSampleT> [trnImgsFlat; raw.TrnLbls]

        let trn, vali =
            match orgTrn |> Dataset.partition [1. - valRatio; valRatio] with
            | [trn; vali] -> trn, vali
            | _ -> failwith "impossible"
        let tst = Dataset<InputTargetSampleT> [tstImgsFlat; raw.TstLbls]

        {Trn=trn; Val=vali; Tst=tst}

