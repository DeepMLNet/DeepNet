namespace Datasets

open System
open System.IO
open System.IO.Compression

open Basics
open ArrayNDNS


[<AutoOpen>]
module MnistTypes =

    /// MNIST dataset
    type MnistT = {
        /// 2d training images of shape [60000; 28; 28]
        TrnImgs:      ArrayNDT<single>
        /// flat training images of shape [60000; 784]
        TrnImgsFlat:  ArrayNDT<single>
        /// one-hot training labels of shape [60000; 10]
        TrnLbls:      ArrayNDT<single>

        /// 2d test images of shape [10000; 28; 28]
        TstImgs:      ArrayNDT<single>
        /// flat test images of shape [10000; 784]
        TstImgsFlat:  ArrayNDT<single>
        /// one-hot test labels of shape [10000; 10]
        TstLbls:      ArrayNDT<single>   
    }


module Mnist = 

    [<Literal>]
    let TestDataset = false

    let private assemble dataSeq =
        let data = List.ofSeq dataSeq
        let nSamples = List.length data

        let dataShape = ArrayND.shape data.[0]
        let ds = ArrayNDHost.zeros (nSamples :: dataShape)

        data |> List.iteri (fun smpl d -> ds.[smpl, Fill] <- d)
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
            let label = labelReader.ReadByte() |> int
            let labelHot : ArrayNDHostT<single> = ArrayNDHost.zeros [10];
            labelHot.[[label]] <- 1.0f

            let image = imageReader.ReadBytes (nRows * nCols)           
            let imageSingle = Array.map (fun p -> single p / 255.0f) image
            let imageMat = ArrayNDHost.ofArray imageSingle |> ArrayND.reshape [nRows; nCols]

            yield labelHot, imageMat
    }
        
    let private dataset labelPath imagePath =
        let labelSeq, imageSeq = sampleSeq labelPath imagePath |> Seq.toList |> List.unzip
        assemble labelSeq, assemble imageSeq    

    let private loadRaw directory =
        let trnLbls, trnImgs = 
            dataset (Path.Combine (directory, "train-labels-idx1-ubyte.gz")) 
                    (Path.Combine (directory, "train-images-idx3-ubyte.gz"))
        let tstLbls, tstImgs = 
            dataset (Path.Combine (directory, "t10k-labels-idx1-ubyte.gz")) 
                    (Path.Combine (directory, "t10k-images-idx3-ubyte.gz"))
    
        let trnImgsFlat = trnImgs |> ArrayND.reshape [trnImgs.Shape.[0]; -1]
        let tstImgsFlat = tstImgs |> ArrayND.reshape [tstImgs.Shape.[0]; -1]

        {TrnImgs = trnImgs; TrnImgsFlat = trnImgsFlat; TrnLbls = trnLbls;
         TstImgs = tstImgs; TstImgsFlat = tstImgsFlat; TstLbls = tstLbls;}

    let load directory =
        let testStr = if TestDataset then "-Test" else ""
        let hdfPath = Path.Combine (directory, sprintf "MNIST%s.h5" testStr)
        if File.Exists hdfPath then
            use hdf = new HDF5 (hdfPath, HDF5Read)
            {TrnImgs = ArrayNDHDF.read hdf "TrnImgs"; 
             TrnImgsFlat = ArrayNDHDF.read hdf "TrnImgsFlat"; 
             TrnLbls = ArrayNDHDF.read hdf "TrnLbls";
             TstImgs = ArrayNDHDF.read hdf "TstImgs"; 
             TstImgsFlat = ArrayNDHDF.read hdf "TstImgsFlat"; 
             TstLbls = ArrayNDHDF.read hdf "TstLbls";}
        else
            printf "Converting MNIST to HDF5..."
            let mnist = loadRaw directory
            use hdf = new HDF5 (hdfPath, HDF5Overwrite)
            ArrayNDHDF.write hdf "TrnImgs" (mnist.TrnImgs :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TrnImgsFlat" (mnist.TrnImgsFlat :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TrnLbls" (mnist.TrnLbls :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TstImgs" (mnist.TstImgs :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TstImgsFlat" (mnist.TstImgsFlat :?> ArrayNDHostT<single>)
            ArrayNDHDF.write hdf "TstLbls" (mnist.TstLbls :?> ArrayNDHostT<single>)
            printfn "Done."
            mnist

    let toCuda mnist =
        {TrnImgs = mnist.TrnImgs :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TrnImgsFlat = mnist.TrnImgsFlat :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TrnLbls = mnist.TrnLbls :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TstImgs = mnist.TstImgs :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TstImgsFlat = mnist.TstImgsFlat :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
         TstLbls = mnist.TstLbls :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev}
