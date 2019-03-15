namespace Tensor.Expr.ML

open System.Collections
open System.Collections.Generic
open Microsoft.FSharp.Reflection

open DeepNet.Utils
open Tensor
open Tensor.Backend



/// A dataset of a record type 'S containing Tensor<_> data variables.
/// The first dimension of each record field is the sample.
/// All record fields must contain the same number of samples.
/// The default constructor expects a list of ITensors corresponding to the fields in 'S. 
/// To construct a Dataset<_> from a sequence of samples use the
/// Dataset<_>.fromSamples function.
[<StructuredFormatDisplay("{Pretty}")>]
type Dataset<'S> (fieldStorages: ITensor list,
                  isSeq:         bool) =

    do if not (FSharpType.IsRecord typeof<'S>) then
        failwith "Dataset sample type must be a record containing Tensors."
    do if List.isEmpty fieldStorages then
        invalidArg "fieldStorages" "fieldStorages must not be empty."

    // make sure that all field storages are in row-major order and stored on the same device
    let _dev = fieldStorages.[0].Dev
    do for fs in fieldStorages do
        if not (TensorLayout.isRowMajor fs.Layout) then
            failwith "All field storages in must be in row-major order."
        if fs.Dev <> _dev then
            failwith "All field storages must be stored on same device."

    /// number of samples
    let _nSamples = fieldStorages.[0].Shape.[0]
    do if fieldStorages |> List.exists (fun fs -> fs.Shape.[0] <> _nSamples) then
        invalidArg "fieldStorages" "unequal number of samples in fields"

    /// number of steps in a sequence dataset
    let _nSteps () = 
        if isSeq then
            if fieldStorages.[0].NDims < 2 then
                failwith "fields must be at least two-dimensional for a sequence dataset"
            let nSteps = fieldStorages.[0].Shape.[1]
            if fieldStorages |> List.exists (fun fs -> fs.NDims < 2 || fs.Shape.[1] <> nSteps) then
                failwith "unequal number of steps in fields"        
            nSteps
        else failwith "not a sequence dataset"
    do if isSeq then _nSteps () |> ignore

    /// checks arguments for being in range
    let checkRange smpl =
        if not (0L <= smpl && smpl < _nSamples) then
            failwithf "sample index %d is out of range (have %d samples)" smpl _nSamples

    /// checks step argumetn to be in range
    let checkStepRange step =
        if not (0L <= step && step < _nSteps()) then
            failwithf "step index %d is out of range (have %d steps)" step (_nSteps())

    /// Creates a non-sequence dataset using the specified field storages.
    new (fieldStorages: ITensor list) = Dataset<'S> (fieldStorages, false)

    /// Constructs a dataset from samples.
    static member private Create (samples: 'S seq, isSeq: bool) =          
        let samples = Seq.cache samples 
        if Seq.isEmpty samples then
            invalidArg "samples" "need at least one sample to create a Dataset"

        // ary.[smpl,field] : IArray[,]
        let nFields = Array.length (FSharpValue.GetRecordFields (Seq.head samples))
        let nSamples = Seq.length samples 
        let ary = Array2D.zeroCreate nSamples nFields
        for smpl, value in Seq.indexed samples do
            ary.[smpl, *] <-
                FSharpValue.GetRecordFields value
                |> Array.map (fun v -> v :?> ITensor)

        // find largest shape of each field over all samples
        let maxShape (fieldSmpls: ITensor seq) =
            let mutable maxShape = Seq.head fieldSmpls |> ITensor.shape
            for smpl in fieldSmpls do
                let smplShape = ITensor.shape smpl
                if List.length smplShape <> List.length maxShape then
                    failwith "dimensionality of a field must be equal over all samples"
                maxShape <- (maxShape, smplShape) ||> List.map2 max
            maxShape

        // build data storage
        let fieldStorage (fieldSmpls: ITensor seq) =
            let maxSmplShp = maxShape fieldSmpls
            let storShp = (int64 nSamples) :: maxSmplShp
            let fieldTyp = (Seq.head fieldSmpls).DataType
            let stor = Tensor.NewOfType (storShp, fieldTyp, HostTensor.Dev, order=RowMajor)
            for smpl, smplVal in Seq.indexed fieldSmpls do
                if stor.[int64 smpl, Fill].Shape = smplVal.Shape then
                    stor.[int64 smpl, Fill] <- smplVal
                else
                    failwithf "the sample with index %d has shape %A but shape %A was expected"
                              smpl smplVal.Shape stor.[0L, Fill].Shape
            stor
        let fieldStorages = 
            [for fld=0 to nFields-1 do yield fieldStorage ary.[*, fld]]

        Dataset<'S> (fieldStorages, isSeq)

    /// Constructs a dataset from a sequence of samples of record type 'S.
    /// Each field in 'S must be of type Tensor<_> and the dimensionality of each field
    /// must be constant over all samples.
    /// If the shape of a field varies over the samples it is padded (with zeros) to the largest 
    /// shape in the sample sequence.
    /// The given sequence is enumerated only one time and the data is copied once.
    static member ofSamples samples =
        Dataset.Create (samples, false)

    /// Constructs a sequence dataset from a sequence of samples of record type 'S.
    static member ofSeqSamples seqSamples =
        Dataset.Create (seqSamples, true)

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
            [| for fs in fieldStorages -> fs.[[Rng.Rng (start, stop); Rng.AllFill]] |> box |]
        FSharpValue.MakeRecord (typeof<'S>, sliceData) :?> 'S            

    /// For a sequence dataset,
    /// returns a record of type 'S containing a slice of samples and slots.
    member this.GetSlice (startSmpl: int64 option, stopSmpl: int64 option,
                          startStep: int64 option, stopStep: int64 option) =
        startSmpl |> Option.iter checkRange; stopSmpl |> Option.iter checkRange
        startStep |> Option.iter checkStepRange; stopStep |> Option.iter checkStepRange
        let sliceData =
            [| for fs in fieldStorages -> fs.[[Rng.Rng (startSmpl, stopSmpl); 
                                               Rng.Rng (startStep, stopStep); Rng.AllFill]] |> box |]
        FSharpValue.MakeRecord (typeof<'S>, sliceData) :?> 'S            
                            
    /// Returns a record of type 'S containing all samples.
    member this.All = 
        let allData =
            [| for fs in fieldStorages -> fs |> box |]
        FSharpValue.MakeRecord (typeof<'S>, allData) :?> 'S            

    /// Returns a new dataset containing the samples from start to stop.
    member this.Part (start: int64, stop: int64) =        
        let partData =
            fieldStorages |> List.map (fun fs -> fs.[[Rng.Rng (Some start, Some stop); Rng.AllFill]])
        Dataset<'S> (partData, isSeq)

    /// number of samples
    member this.NSamples = _nSamples
    /// Number of samples.
    static member nSamples (ds: Dataset<'S>) = ds.NSamples

    /// Number of steps per sample in a sequence dataset.
    member this.NSteps = _nSteps()
    /// Number of steps per sample in a sequence dataset.
    static member nSteps (ds: Dataset<'S>) = ds.NSteps

    /// true if this is a sequence dataset
    member this.IsSeq = isSeq

    /// list of arrays corresponding to the data of each field in the sample record
    member this.FieldStorages = fieldStorages

    /// data type of samples
    member this.SampleType = typeof<'S>

    /// storage location
    member this.Dev = _dev

    /// Generates a function that returns a sequence of batches with the given size of this dataset.
    /// If the number of samples in this dataset is not a multiple of the batch size,
    /// the last batch will still have the specified size but is padded with zeros.
    member this.PaddedBatches batchSize = 
        let lastBatchElems = _nSamples % batchSize
        let lastBatchStart = _nSamples - lastBatchElems

        // create padded last batch, if necessary
        let lastBatch =
            if lastBatchElems = 0L then None
            else                   
                fieldStorages
                |> List.map (fun fsAll ->
                    let shpAll = ITensor.shape fsAll
                    let shpBatch = shpAll |> List.set 0 batchSize                    
                    let fsBatch = Tensor.NewOfType (shpBatch, fsAll.DataType, fsAll.Dev, order=RowMajor)
                    fsBatch.[0L .. lastBatchElems-1L, Fill] <- fsAll.[lastBatchStart .. _nSamples-1L, Fill]
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
        let lastBatchElems = _nSamples % batchSize
        let lastBatchStart = _nSamples - lastBatchElems
        seq {
            // all batches except last batch 
            for start in 0L .. batchSize .. lastBatchStart-1L do
                let stop = start + batchSize - 1L
                yield! batchFn (start, stop)                    
            // last batch 
            if lastBatchStart < _nSamples then
                yield! batchFn (lastBatchStart, _nSamples-1L)
        }          

    /// returns a sequence of all slots for the given batch
    member private this.Slots slotSize (batchStart, batchStop) = 
        let nSteps = _nSteps()
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
            (seq { for idx in 0L .. _nSamples-1L -> this.[idx] }).GetEnumerator()
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
            if fs.Dev <> HostTensor.Dev then 
                failwith "can only save a dataset stored on the host"
            HostTensor.write hdf (prefixPath + "/" + fldInfo.Name) fs

    /// Saves this dataset into the specified HDF5 file.
    /// The file is overwritten.
    member this.Save (filename) =
        use hdf = HDF5.OpenWrite filename
        this.Save (hdf)

    /// Loads a dataset from the specified HDF5 file.
    /// `hdfPrefixPath` optionally specifies a prefix path within the HDF5 file for the dataset.
    static member load (hdf, ?hdfPrefixPath) =
        let prefixPath = defaultArg hdfPrefixPath ""
        if not (FSharpType.IsRecord typeof<'S>) then
            failwith "Dataset sample type must be a record containing Tensors"
        FSharpType.GetRecordFields typeof<'S>
        |> Seq.map (fun fldInfo ->
            if not (typeof<ITensor>.IsAssignableFrom fldInfo.PropertyType) then 
                failwith "Dataset sample type must be a record containing Tensors"
            let dataType = fldInfo.PropertyType.GenericTypeArguments.[0]
            HostTensor.readUntyped hdf (prefixPath + "/" + fldInfo.Name))
        |> Seq.toList
        |> Dataset<'S>

    /// Loads a dataset from the specified HDF5 file.
    static member load (filename) =
        use hdf = HDF5.OpenRead filename
        Dataset<'S>.load (hdf)        

    /// Partitions this dataset using the given ratios.
    static member partition ratios (ds: Dataset<'S>) = 
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
    static member batches batchSize (ds: Dataset<'S>) =
        ds.Batches batchSize

    /// Returns a sequence of batches of time slots with size `batchSize` and `slotSize`
    /// respectively of the dataset.
    static member slotBatches batchSize slotSize (ds: Dataset<'S>) =
        ds.SlotBatches batchSize slotSize

    /// Cuts each sequence in the dataset into multiple chunks of length `stepsPerCut`.
    static member cutSequences stepsPerCut (ds: Dataset<'S>) =
        let nSteps = ds.NSteps
        let nCuts = (nSteps + stepsPerCut - 1L) / stepsPerCut
        let padSteps = nCuts * stepsPerCut
        let cutFs =         
            ds.FieldStorages |> List.map (fun fs ->
                let rShp = fs.Shape.[2..]
                let fs =
                    // pad if necessary
                    if padSteps > nSteps then
                        let z = Tensor.NewOfType ([ds.NSamples; padSteps] @ rShp, fs.DataType, fs.Dev)   
                        z.FillZero()
                        z.[*, 0L .. nSteps-1L, Fill] <- fs.[Fill]
                        z
                    else fs
                fs |> ITensor.reshapeView ([ds.NSamples * nCuts; stepsPerCut] @ rShp))
        Dataset (cutFs, true)

    /// Cuts each sequence in the dataset into multiple chunks so that at least `minSamples` are 
    /// in the resulting dataset. If the dataset already contains at least `minSamples`,
    /// then the function returns the dataset unaltered.
    static member cutToMinSamples minSamples (ds: Dataset<'S>) =
        if ds.NSamples < minSamples then
            let nCuts = (minSamples + ds.NSamples - 1L) / ds.NSamples
            let stepsPerCut = max 1L (ds.NSteps / nCuts)
            ds |> Dataset.cutSequences stepsPerCut
        else ds

    /// maps the field storages using the given function creating a new dataset
    static member map (f: ITensor -> #ITensor) (ds: Dataset<'S>) : Dataset<'S> =
        ds.FieldStorages
        |> List.map (f >> (fun fs -> fs :> ITensor))
        |> fun fs -> Dataset<'S> (fs, ds.IsSeq)

    /// copies this dataset to the specified device
    static member transfer dev (ds: Dataset<'S>) : Dataset<'S> =
        ds |> Dataset.map (ITensor.transfer dev)



