namespace Datasets

open System.Collections
open System.Collections.Generic
open Microsoft.FSharp.Reflection

open Basics
open ArrayNDNS
open Util

/// A dataset of a record type 'S containing ArrayNDT<_> data variables.
/// The first dimension of each record field is the sample.
/// All record fields must contain the same number of samples.
/// The default constructor expects a list of ArrayNDTs corresponding to the fields
/// in 'S. 
/// To construct a Dataset<_> from a sequence of samples use the
/// Dataset<_>.FromSamples method.
[<StructuredFormatDisplay("{Pretty}")>]
type Dataset<'S> (fieldStorages: IArrayNDT list,
                  isSeq:         bool) =

    do if not (FSharpType.IsRecord typeof<'S>) then
        failwith "Dataset sample type must be a record containing ArrayNDTs"

    // make sure that all field storages are in C-order
    do for fs in fieldStorages do
        if not (ArrayND.isC fs) then
            failwith "all field storages in must be in C-order"

    /// number of samples
    let nSamples = fieldStorages.[0].Shape.[0]
    do if fieldStorages |> List.exists (fun fs -> fs.Shape.[0] <> nSamples) then
        invalidArg "fieldStorages" "unequal number of samples in fields"

    /// number of steps in a sequence dataset
    let nSteps () = 
        if isSeq then
            if fieldStorages.[0].NDims < 2 then
                failwith "fields must be at least two-dimensional for a sequence dataset"
            let nSteps = fieldStorages.[0].Shape.[1]
            if fieldStorages |> List.exists (fun fs -> fs.NDims < 2 || fs.Shape.[1] <> nSteps) then
                failwith "unequal number of steps in fields"        
            nSteps
        else failwith "not a sequence dataset"
    do if isSeq then nSteps () |> ignore

    /// checks arguments for being in range
    let checkRange smpl =
        if not (0L <= smpl && smpl < nSamples) then
            failwithf "sample index %d is out of range (have %d samples)" smpl nSamples

    /// checks step argumetn to be in range
    let checkStepRange step =
        if not (0L <= step && step < nSteps()) then
            failwithf "step index %d is out of range (have %d steps)" step (nSteps())

    /// Creates a non-sequence dataset using the specified field storages.
    new (fieldStorages: IArrayNDT list) = Dataset<'S> (fieldStorages, false)

    /// Constructs a dataset from samples.
    static member New (samples: 'S seq, isSeq: bool) =          
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
            let storShp = (int64 nSamples) :: maxSmplShp
            let fieldTyp = (Seq.head fieldSmpls).DataType
            let stor = ArrayNDHost.newCOfType fieldTyp storShp 
            for smpl, smplVal in Seq.indexed fieldSmpls do
                if stor.[int64 smpl, Fill].Shape = smplVal.Shape then
                    stor.[int64 smpl, Fill] <- smplVal
                else
                    failwithf "the sample with index %d has shape %A but shape %A was expected"
                              smpl smplVal.Shape stor.[smpl, Fill].Shape
            stor :> IArrayNDT            
        let fieldStorages = 
            [for fld=0 to nFields-1 do yield fieldStorage ary.[*, fld]]

        Dataset<'S> (fieldStorages, isSeq)

    /// Returns a record of type 'S containing the sample with the given index.
    member this.Item 
        with get (smpl: int64) =
            checkRange smpl
            let smplData =
                [| for fs in fieldStorages -> fs.[smpl, Fill] |> box |]
            FSharpValue.MakeRecord (typeof<'S>, smplData) :?> 'S

    /// For a sequence dataset, 
    /// returns a record of type 'S containing the sample and slot with the given indices.
    member this.Item
        with get (smpl: int64, step: int64) =
            checkRange smpl; checkStepRange step
            let smplData =
                [| for fs in fieldStorages -> fs.[smpl, step, Fill] |> box |]
            FSharpValue.MakeRecord (typeof<'S>, smplData) :?> 'S

    /// Returns a record of type 'S containing a slice of samples.
    member this.GetSlice (start: int64 option, stop: int64 option) =
        start |> Option.iter checkRange; stop |> Option.iter checkRange
        let sliceData =
            [| for fs in fieldStorages -> fs.[[Rng (start, stop); RngAllFill]] |> box |]
        FSharpValue.MakeRecord (typeof<'S>, sliceData) :?> 'S            

    /// For a sequence dataset,
    /// returns a record of type 'S containing a slice of samples and slots.
    member this.GetSlice (startSmpl: int64 option, stopSmpl: int64 option,
                          startStep: int64 option, stopStep: int64 option) =
        startSmpl |> Option.iter checkRange; stopSmpl |> Option.iter checkRange
        startStep |> Option.iter checkStepRange; stopStep |> Option.iter checkStepRange
        let sliceData =
            [| for fs in fieldStorages -> fs.[[Rng (startSmpl, stopSmpl); 
                                               Rng (startStep, stopStep); RngAllFill]] |> box |]
        FSharpValue.MakeRecord (typeof<'S>, sliceData) :?> 'S            
                            
    /// Returns a record of type 'S containing all samples.
    member this.All = 
        let allData =
            [| for fs in fieldStorages -> fs |> box |]
        FSharpValue.MakeRecord (typeof<'S>, allData) :?> 'S            

    /// Returns a new dataset containing the samples from start to stop.
    member this.Part (start: int64, stop: int64) =        
        let partData =
            fieldStorages |> List.map (fun fs -> fs.[[Rng (Some start, Some stop); RngAllFill]])
        Dataset<'S> (partData, isSeq)

    /// number of samples
    member this.NSamples = nSamples

    /// number of steps for sequence dataset
    member this.NSteps = nSteps()

    /// true if this is a sequence dataset
    member this.IsSeq = isSeq

    /// list of arrays corresponding to the data of each field in the sample record
    member this.FieldStorages = fieldStorages

    /// data type of samples
    member this.SampleType = typeof<'S>

    /// storage location
    member this.Location = fieldStorages.[0].Location

    /// Generates a function that returns a sequence of batches with the given size of this dataset.
    /// If the number of samples in this dataset is not a multiple of the batch size,
    /// the last batch will still have the specified size but is padded with zeros.
    member this.PaddedBatches batchSize = 
        let lastBatchElems = nSamples % batchSize
        let lastBatchStart = nSamples - lastBatchElems

        // create padded last batch, if necessary
        let lastBatch =
            if lastBatchElems = 0L then None
            else                   
                fieldStorages
                |> List.map (fun fsAll ->
                    let shpAll = ArrayND.shape fsAll
                    let shpBatch = shpAll |> List.set 0 batchSize                    
                    let fsBatch = fsAll |> ArrayND.newCOfSameType shpBatch 
                    fsBatch.[0L .. lastBatchElems-1L, Fill] <- fsAll.[lastBatchStart .. nSamples-1L, Fill]
                    fsBatch)
                |> Some

        fun () ->                    
            seq {
                // all batches except last batch if padding was necessary
                for start in 0L .. batchSize .. lastBatchStart-1L do
                    let stop = start + batchSize - 1L
                    yield this.[start .. stop]  
                    
                // padded last batch if necessary
                match lastBatch with
                | Some lastBatch ->
                    let data = [|for fs in lastBatch -> fs |> box|]
                    yield FSharpValue.MakeRecord (typeof<'S>, data) :?> 'S     
                | None -> ()        
            }           

    /// template batch
    member this.TmplBatch batchSize = 
        this.PaddedBatches batchSize () |> Seq.head

    /// calls the given function for each batch
    member private this.BatchIter batchSize batchFn = 
        let lastBatchElems = nSamples % batchSize
        let lastBatchStart = nSamples - lastBatchElems
        seq {
            // all batches except last batch 
            for start in 0L .. batchSize .. lastBatchStart-1L do
                let stop = start + batchSize - 1L
                yield! batchFn (start, stop)                    
            // last batch 
            if lastBatchStart < nSamples then
                yield! batchFn (lastBatchStart, nSamples-1L)
        }          

    /// returns a sequence of all slots for the given batch
    member private this.Slots slotSize (batchStart, batchStop) = 
        let nSteps = nSteps()
        let lastSlotElems = nSteps % slotSize
        let lastSlotStart = nSteps - lastSlotElems
        seq { 
            // all time slots except last slot
            for slotStart in 0L .. slotSize .. lastSlotStart-1L do
                let slotStop = slotStart + slotSize - 1L
                yield this.[batchStart .. batchStop, slotStart .. slotStop]
            // last time slot
            if lastSlotStart < nSteps then
                yield this.[batchStart .. batchStop, lastSlotStart .. nSteps-1L]
        }

    /// Returns a sequence of batches with the given size of this dataset.
    /// If the number of samples in this dataset is not a multiple of the batch size,
    /// the last batch will be smaller.
    member this.Batches batchSize = 
        this.BatchIter batchSize (fun (start, stop) -> Seq.singleton this.[start .. stop])

    /// Returns a sequence of batches of time slots with size `batchSize` and `slotSize`
    /// respectively of the dataset.
    member this.SlotBatches batchSize slotSize = 
        this.BatchIter batchSize (fun (start, stop) -> this.Slots slotSize (start, stop))

    // enumerator interfaces
    interface IEnumerable<'S> with
        member this.GetEnumerator() =
            (seq { for idx in 0L .. nSamples-1L -> this.[idx] }).GetEnumerator()
    interface IEnumerable with
        member this.GetEnumerator() =
            (this :> IEnumerable<'S>).GetEnumerator() :> IEnumerator

    /// pretty string
    member this.Pretty =
        if isSeq then
            sprintf "Sequence dataset containing %d samples of %s with %d steps per sample"
                    this.NSamples typeof<'S>.Name this.NSteps
        else
            sprintf "Dataset containing %d samples of %s"
                    this.NSamples typeof<'S>.Name 

    /// Saves this dataset into the specified HDF5 file.
    /// `hdfPrefixPath` optionally specifies a prefix path within the HDF5 file for the dataset.
    member this.Save (hdf, ?hdfPrefixPath) =
        let prefixPath = defaultArg hdfPrefixPath ""
        let fldInfos = FSharpType.GetRecordFields this.SampleType
        for fldInfo, fs in Seq.zip fldInfos this.FieldStorages do
            match fs with
            | :? IArrayNDHostT as fs -> ArrayNDHDF.writeUntyped hdf (prefixPath + "/" + fldInfo.Name) fs
            | _ -> failwith "can only save a dataset stored on the host"

    /// Saves this dataset into the specified HDF5 file.
    /// The file is overwritten.
    member this.Save (filename) =
        use hdf = HDF5.OpenWrite filename
        this.Save (hdf)

    /// Loads a dataset from the specified HDF5 file.
    /// `hdfPrefixPath` optionally specifies a prefix path within the HDF5 file for the dataset.
    static member Load<'S> (hdf, ?hdfPrefixPath) =
        let prefixPath = defaultArg hdfPrefixPath ""
        if not (FSharpType.IsRecord typeof<'S>) then
            failwith "Dataset sample type must be a record containing ArrayNDHostTs"
        FSharpType.GetRecordFields typeof<'S>
        |> Seq.map (fun fldInfo ->
            if not (typeof<IArrayNDT>.IsAssignableFrom fldInfo.PropertyType) then 
                failwith "Dataset sample type must be a record containing ArrayNDHostTs"
            let dataType = fldInfo.PropertyType.GenericTypeArguments.[0]
            ArrayNDHDF.readUntyped hdf (prefixPath + "/" + fldInfo.Name) dataType :> IArrayNDT)
        |> Seq.toList
        |> Dataset<'S>

    /// Loads a dataset from the specified HDF5 file.
    static member Load<'S> (filename) =
        use hdf = HDF5.OpenRead filename
        Dataset<'S>.Load (hdf)        

/// Dataset functions.
module Dataset =

    /// Constructs a dataset from a sequence of samples of record type 'S.
    /// Each field in 'S must be of type ArrayNDT<_> and the dimensionality of each field
    /// must be constant over all samples.
    /// If the shape of a field varies over the samples it is padded (with zeros) to the largest 
    /// shape in the sample sequence.
    /// The given sequence is enumerated only one time and the data is copied once.
    let ofSamples samples =
        Dataset.New (samples, false)

    /// Constructs a sequence dataset from a sequence of samples of record type 'S.
    let ofSeqSamples seqSamples =
        Dataset.New (seqSamples, true)

    /// Partitions this dataset using the given ratios.
    let partition ratios (ds: Dataset<'S>) = 
        let ratioSum = List.sum ratios
        let partitionedFieldStorages = 
            ds.FieldStorages
            |> List.map (fun fs ->
                let fsPart, _ =
                    (0L, List.indexed ratios)
                    ||> List.mapFold (fun pos (idx, ratio) ->
                        let isLast = (idx = List.length ratios - 1)
                        let smpls =
                            if isLast then ds.NSamples - pos
                            else int64 (ratio / ratioSum * (float ds.NSamples))
                        fs.[pos .. pos+smpls-1L, Fill], pos+smpls)    
                fsPart)
            |> List.transpose
        partitionedFieldStorages |> List.map (fun fs -> Dataset<'S> (fs, ds.IsSeq))

    /// Returns a sequence of batches with the given size of this dataset.         
    let batches batchSize (ds: Dataset<'S>) =
        ds.Batches batchSize

    /// Returns a sequence of batches of time slots with size `batchSize` and `slotSize`
    /// respectively of the dataset.
    let slotBatches batchSize slotSize (ds: Dataset<'S>) =
        ds.SlotBatches batchSize slotSize

    /// Number of samples.
    let nSamples (ds: Dataset<_>) =
        ds.NSamples

    /// Number of steps per sample in a sequence dataset.
    let nSteps (ds: Dataset<_>) =
        ds.NSteps

    /// Cuts each sequence in the dataset into multiple chunks of length `stepsPerCut`.
    let cutSequences stepsPerCut (ds: Dataset<_>) =
        let nSteps = ds.NSteps
        let nCuts = (nSteps + stepsPerCut - 1L) / stepsPerCut
        let padSteps = nCuts * stepsPerCut
        let cutFs =         
            ds.FieldStorages |> List.map (fun fs ->
                let rShp = fs.Shape.[2..]
                let fs =
                    // pad if necessary
                    if padSteps > nSteps then
                        let z = fs |> ArrayND.zerosOfSameType ([ds.NSamples; padSteps] @ rShp)
                        z.[*, 0L .. nSteps-1L, Fill] <- fs.[Fill]; z
                    else fs
                fs |> ArrayND.reshapeView ([ds.NSamples * nCuts; stepsPerCut] @ rShp))
        Dataset (cutFs, true)

    /// Cuts each sequence in the dataset into multiple chunks so that at least `minSamples` are 
    /// in the resulting dataset. If the dataset already contains at least `minSamples`,
    /// then the function returns the dataset unaltered.
    let cutToMinSamples minSamples (ds: Dataset<_>) =
        if ds.NSamples < minSamples then
            let nCuts = (minSamples + ds.NSamples - 1L) / ds.NSamples
            let stepsPerCut = max 1L (ds.NSteps / nCuts)
            ds |> cutSequences stepsPerCut
        else ds

    /// maps the field storages using the given function creating a new dataset
    let map (f: IArrayNDT -> #IArrayNDT) (ds: Dataset<'S>) : Dataset<'S> =
        ds.FieldStorages
        |> List.map (f >> (fun fs -> fs :> IArrayNDT))
        |> fun fs -> Dataset<'S> (fs, ds.IsSeq)

    /// copies this dataset to a CUDA GPU
    let toCuda (ds: Dataset<'S>) : Dataset<'S> =
        ds |> map (fun fs ->
            ArrayNDCuda.toDevUntyped (fs :?> IArrayNDHostT))

    /// copies this dataset to the host
    let toHost (ds: Dataset<'S>) : Dataset<'S> =
        ds |> map (fun fs ->
            ArrayNDCuda.toHostUntyped (fs :?> IArrayNDCudaT))

/// A training/validation/test partitioning of a dataset.
[<StructuredFormatDisplay("{Pretty}")>]
type TrnValTst<'S> = { 
    /// training partition
    Trn:    Dataset<'S>
    /// validation partition
    Val:    Dataset<'S>
    /// test partition
    Tst:    Dataset<'S> 
} with 
    member internal this.Pretty = 
        if this.Trn.IsSeq then
            sprintf "sequence dataset (%d training, %d validation, %d test %ss with \
                     %d steps per sample)" 
                this.Trn.NSamples this.Val.NSamples this.Tst.NSamples this.Trn.SampleType.Name
                this.Trn.NSteps
        else
            sprintf "dataset (%d training, %d validation, %d test %ss)"
                this.Trn.NSamples this.Val.NSamples this.Tst.NSamples this.Trn.SampleType.Name
        

module TrnValTst =

    /// Applies the given function to the training, validation and test partions
    /// and return the results as a tuple (trn, val, tst).
    let apply f trnValTst = 
        f trnValTst.Trn, f trnValTst.Val, f trnValTst.Tst

    /// Creates the partitioning from the specified dataset using the specified
    /// training, validation, test splits.
    let ofDatasetWithRatios (trnRatio, valRatio, tstRatio) dataset =
        match dataset |> Dataset.partition [trnRatio; valRatio; tstRatio] with
        | [trn; vali; tst] -> {Trn=trn; Val=vali; Tst=tst}
        | _ -> failwith "impossible"

    /// Creates the partitioning from the specified dataset using a split of
    /// 80%, 10% and 10% for the training, validation and test partitions respectively.
    let ofDataset dataset =
        dataset |> ofDatasetWithRatios (0.8, 0.1, 0.1)

    /// Copies the given dataset to a CUDA GPU.
    let toCuda (this: TrnValTst<'S>) = {
        Trn = this.Trn |> Dataset.toCuda
        Val = this.Val |> Dataset.toCuda
        Tst = this.Tst |> Dataset.toCuda
    }

    /// Copies the given dataset to the host.
    let toHost (this: TrnValTst<'S>) = {
        Trn = this.Trn |> Dataset.toHost
        Val = this.Val |> Dataset.toHost
        Tst = this.Tst |> Dataset.toHost
    }

    /// Saves this dataset to disk in an HDF5 file.
    /// HDF5 folders called 'Trn', 'Val' and 'Tst' are used to store the dataset parations.
    let save filename (this: TrnValTst<'S>) =
        use hdf = HDF5.OpenWrite filename
        this.Trn.Save (hdf, "Trn")
        this.Val.Save (hdf, "Val")
        this.Tst.Save (hdf, "Tst")

    /// Loads a dataset from an HDF5 file.
    /// HDF5 folders called 'Trn', 'Val' and 'Tst' are used to store the dataset parations.
    let load filename : TrnValTst<'S> = 
        use hdf = HDF5.OpenRead filename
        {
            Trn = Dataset.Load (hdf, "Trn")
            Val = Dataset.Load (hdf, "Val")
            Tst = Dataset.Load (hdf, "Tst")
        }
