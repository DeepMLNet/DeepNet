namespace Datasets

open System
open System.IO
open System.IO.Compression

open ArrayNDNS


[<AutoOpen>]
module MnistTypes =

    type MnistT = {
        TrnImgs:      ArrayNDT<single>;
        TrnLbls:      ArrayNDT<single>;
        TstImgs:      ArrayNDT<single>;
        TstLbls:      ArrayNDT<single>;   
    }


module Mnist = 

    let private assemble dataSeq =
        let data = List.ofSeq dataSeq
        let nSamples = List.length data

        let dataShape = ArrayND.shape data.[0]
        let ds = ArrayNDHost.zeros (dataShape @ [nSamples])

        data |> List.iteri (fun smpl d -> 
            if smpl % 1000 = 0 then printfn "%d" smpl
            ds.[Fill, smpl] <- d)
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

        //let nSamples = min nSamples 500

        for smpl in 0 .. nSamples - 1 do
            let label = labelReader.ReadByte() |> int
            let labelHot : ArrayNDT<single> = ArrayNDHost.zeros [10];
            labelHot.[[label]] <- 1.0f

            let image = imageReader.ReadBytes (nRows * nCols)           
            let imageSingle = Array.map (fun p -> single p / 255.0f) image
            let imageMat = ArrayNDHost.ofArray imageSingle [nRows; nCols]

            yield labelHot, imageMat
    }
        
    let private dataset labelPath imagePath =
        let labelSeq, imageSeq = sampleSeq labelPath imagePath |> Seq.toList |> List.unzip
        assemble labelSeq, assemble imageSeq    

    let load directory =
        let trnLbls, trnImgs = 
            dataset (Path.Combine (directory, "train-labels-idx1-ubyte.gz")) 
                    (Path.Combine (directory, "train-images-idx3-ubyte.gz"))
        let tstLbls, tstImgs = 
            dataset (Path.Combine (directory, "t10k-labels-idx1-ubyte.gz")) 
                    (Path.Combine (directory, "t10k-images-idx3-ubyte.gz"))
    
        {TrnImgs = trnImgs; TrnLbls = trnLbls;
         TstImgs = tstImgs; TstLbls = tstLbls;}
