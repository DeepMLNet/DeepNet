module MNIST

open System
open System.IO
open System.IO.Compression

open ArrayNDNS


let assemble dataSeq =
    let data = List.ofSeq dataSeq
    let nSamples = List.length data

    let dataShape = ArrayND.shape data.[0]
    let ds = ArrayNDHost.zeros (dataShape @ [nSamples])

//    let b = [1; "abc"; 2.5f]

    for smpl in 0 .. nSamples - 1 do
        ()



let readMNISTFile imagePath labelPath =
    use imageGzStream = File.OpenRead imagePath
    use labelGzStream = File.OpenRead labelPath
    use imageStream = new GZipStream (imageGzStream, CompressionMode.Decompress)
    use labelStream = new GZipStream (labelGzStream, CompressionMode.Decompress)
    use imageReader = new BinaryReader (imageStream)
    use labelReader = new BinaryReader (labelStream)

    if labelReader.ReadUInt32 () <> 2049u then failwith "invalid MNIST label file"
    if imageReader.ReadUInt32 () <> 2051u then failwith "invalid MNIST image file"

    let nSamples = labelReader.ReadInt32()
    if imageReader.ReadInt32() <> nSamples then failwith "number of samples mismatch in MNIST"

    let nRows = imageReader.ReadInt32()
    let nCols = imageReader.ReadInt32()

    seq {
        for smpl in 0 .. nSamples - 1 do
            let label = imageReader.ReadByte() |> int
            let labelHot : ArrayNDT<single> = ArrayNDHost.zeros [10];
            labelHot.[[label]] <- 1.0f

            let image = imageReader.ReadBytes (nRows * nCols)           
            let imageSingle = Array.map (fun p -> single p / 255.0f) image
            let imageMat = ArrayNDHost.ofArray imageSingle [nRows; nCols]

            yield labelHot, imageMat
    }



let readMNIST directory =
    ()
