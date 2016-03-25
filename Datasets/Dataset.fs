namespace Datasets

open System.Collections
open System.Collections.Generic
open Microsoft.FSharp.Reflection

open Basics
open ArrayNDNS


[<AutoOpen>]
module DatasetTypes =

    /// A dataset of a record type 'S containing ArrayNDHostT<_> data variables.
    /// The last dimension of each record field is the sample.
    /// All record fields must contain the same number of samples.
    type Dataset<'S> (fieldStorages: IArrayNDT list) =

        // verify that all fields have equal number of samples
        let nSamples = fieldStorages.[0] |> ArrayND.shape |> List.last
        do
            fieldStorages
            |> List.iter (fun fs ->
                if ArrayND.shape fs |> List.last <> nSamples then
                    invalidArg "fieldStorages" "unequal number of samples in fields")

        /// checks arguments for being in range
        let checkRange smpl =
            if not (smpl >= 0 && smpl < nSamples) then
                failwithf "sample index %d is out of range (have %d samples)" smpl nSamples

        /// Partitions this dataset using the given ratios.
        member this.Partition (ratios: float list) = 
            let ratioSum = List.sum ratios
            let partitionedFieldStorages = 
                fieldStorages
                |> List.map (fun fs ->
                    let fsPart, _ =
                        (0, List.indexed ratios)
                        ||> List.mapFold (fun pos (idx, ratio) ->
                            let isLast = (idx = List.length ratios - 1)
                            let smpls =
                                if isLast then nSamples - pos
                                else int (ratio / ratioSum * (float nSamples))
                            fs.[Fill, pos .. pos+smpls-1], pos+smpls)    
                    fsPart)
                |> List.transpose
            partitionedFieldStorages |> List.map Dataset<'S>

        /// Returns a record of type 'S containing the sample with the given index.
        member this.Item 
            with get (smpl: int) =
                checkRange smpl
                let smplData =
                    [| for fs in fieldStorages -> fs.[Fill, smpl] |> box |]
                FSharpValue.MakeRecord (typeof<'S>, smplData) :?> 'S

        /// Returns a record of type 'S containing a slice of samples.
        member this.GetSlice (start: int option, stop: int option) =
            match start with | Some smpl -> checkRange smpl | None -> ()
            match stop  with | Some smpl -> checkRange smpl | None -> ()  
            let sliceData =
                [| for fs in fieldStorages -> fs.[[RngAllFill; Rng (start, stop)]] |> box |]
            FSharpValue.MakeRecord (typeof<'S>, sliceData) :?> 'S            
                            
        /// Returns a record of type 'S containing all samples.
        member this.All = 
            let allData =
                [| for fs in fieldStorages -> fs |> box |]
            FSharpValue.MakeRecord (typeof<'S>, allData) :?> 'S            

        /// number of samples
        member this.NSamples = nSamples

        /// Generates a function that returns a sequence of batches with the given size of this dataset.
        /// If the number of samples in this dataset is not a multiple of the batch size,
        /// the last batch will still have the specified size but is padded with zeros.
        member this.Batches batchSize = 
            let lastBatchElems = nSamples % batchSize
            let lastBatchStart = nSamples - lastBatchElems

            // create padded last batch, if necessary
            let lastBatch =
                if lastBatchElems = 0 then None
                else                   
                    fieldStorages
                    |> List.map (fun fsAll ->
                        let shpAll = ArrayND.shape fsAll
                        let shpBatch = shpAll |> List.set (List.length shpAll - 1) batchSize                    
                        let fsBatch = fsAll |> ArrayND.newContiguousOfSameType shpBatch 
                        fsBatch.[Fill, 0 .. lastBatchElems-1] <- fsAll.[Fill, lastBatchStart .. nSamples-1]
                        fsBatch)
                    |> Some

            fun () ->                    
                seq {
                    // all batches except last batch if padding was necessary
                    for start in 0 .. batchSize .. lastBatchStart-1 do
                        let stop = start + batchSize - 1
                        yield this.[start .. stop]  
                    
                    // padded last batch if necessary
                    match lastBatch with
                    | Some lastBatch ->
                        let data =
                            [| for fs in lastBatch -> fs |> box |]
                        yield FSharpValue.MakeRecord (typeof<'S>, data) :?> 'S     
                    | None -> ()        
                }           

        interface IEnumerable<'S> with
            member this.GetEnumerator() =
                (seq { for idx in 0 .. nSamples - 1 -> this.[idx] }).GetEnumerator()
        interface IEnumerable with
            member this.GetEnumerator() =
                (this :> IEnumerable<'S>).GetEnumerator() :> IEnumerator


        /// Constructs a dataset from a sequence of samples of record type 'S.
        /// Each field in 'S must be of type ArrayNDT<_> and the dimensionality of each field
        /// must be constant over all samples.
        /// If the shape of a field varies over the samples it is padded (with zeros) to the largest 
        /// shape in the sample sequence.
        /// The given sequence is enumerated only one time and the data is copied once.
        static member FromSamples (samples: 'S seq) =          
            let samples = Seq.cache samples 
            if Seq.isEmpty samples then
                invalidArg "samples" "need at least one sample to create a Dataset"

            // ary.[smpl,field] : IArrayNDT[,]
            let nFields = Array.length (FSharpValue.GetRecordFields (Seq.head samples))
            let nSamples = Seq.length samples
            let ary = Array2D.zeroCreate nSamples nFields
            for smpl, value in Seq.indexed samples do
                ary.[smpl, *] <-
                    FSharpValue.GetRecordFields value
                    |> Array.map (fun v -> v :?> IArrayNDT)

            // find largest shape of each field over all samples
            let maxShape (fieldSmpls: IArrayNDT seq) =
                let mutable maxShape = Seq.head fieldSmpls |> ArrayND.shape
                for smpl in fieldSmpls do
                    let smplShape = ArrayND.shape smpl
                    if List.length smplShape <> List.length maxShape then
                        failwith "dimensionality of a field must be equal over all samples"
                    maxShape <- (maxShape, smplShape) ||> List.map2 max
                maxShape

            // build data storage
            let fieldStorage (fieldSmpls: IArrayNDT seq) =
                let maxSmplShp = maxShape fieldSmpls
                let storShp = maxSmplShp @ [nSamples]
                let fieldTyp = (Seq.head fieldSmpls).DataType
                let stor = ArrayNDHost.newContiguousOfType fieldTyp storShp 
                for smpl, smplVal in Seq.indexed fieldSmpls do
                    stor.[Fill, smpl] <- smplVal
                stor :> IArrayNDT            

            let fieldStorages = [for fld=0 to nFields-1 do yield fieldStorage ary.[*, fld]]
            Dataset<'S> fieldStorages

 

    /// A training/validation/test partioning of a dataset.
    type TrnValTst<'S> = 
        { Trn:    Dataset<'S>
          Val:    Dataset<'S>
          Tst:    Dataset<'S> }
        with 
            /// Creates the partitioning from the specified dataset.
            static member Of (dataset: Dataset<'S>, ?trnRatio: float, ?valRatio: float, ?tstRatio: float) =
                let trnRatio = defaultArg trnRatio 0.75
                let valRatio = defaultArg valRatio 0.15
                let tstRatio = defaultArg tstRatio 0.15
                match dataset.Partition [trnRatio; valRatio; tstRatio] with
                | [trn; vali; tst] -> {Trn=trn; Val=vali; Tst=tst}
                | _ -> failwith "impossible"


            
