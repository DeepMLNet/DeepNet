namespace Tensor.Expr.ML

open DeepNet.Utils
open Tensor
open Tensor.Backend



/// Partition of the dataset.
[<RequireQualifiedAccess>]
type Partition =
    /// training parition of the dataset
    | Trn
    /// validation partition of the dataset
    | Val
    /// test partition of the dataset
    | Tst


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
            sprintf "sequence dataset (%d training, %d validation, %d test %ss with %d steps per sample)" 
                this.Trn.NSamples this.Val.NSamples this.Tst.NSamples this.Trn.SampleType.Name
                this.Trn.NSteps
        else
            sprintf "dataset (%d training, %d validation, %d test %ss)"
                this.Trn.NSamples this.Val.NSamples this.Tst.NSamples this.Trn.SampleType.Name
        
    /// Returns the respective partition.
    member this.Item 
        with get (part: Partition) =
            match part with
            | Partition.Trn -> this.Trn
            | Partition.Val -> this.Val
            | Partition.Tst -> this.Tst

    /// Returns the respective partition.
    static member part (part: Partition) (trnValTst: TrnValTst<'S>) =
        trnValTst.[part]

    /// Applies the given function to the training, validation and test partions
    /// and return the results as a tuple (trn, val, tst).
    static member apply f trnValTst = 
        f trnValTst.Trn, f trnValTst.Val, f trnValTst.Tst

    /// Creates the partitioning from the specified dataset using the specified
    /// training, validation, test splits.
    static member ofDatasetWithRatios (trnRatio, valRatio, tstRatio) (dataset: Dataset<'S>) =
        match dataset |> Dataset.partition [trnRatio; valRatio; tstRatio] with
        | [trn; vali; tst] -> {Trn=trn; Val=vali; Tst=tst}
        | _ -> failwith "impossible"

    /// Creates the partitioning from the specified dataset using a split of
    /// 80%, 10% and 10% for the training, validation and test partitions respectively.
    static member ofDataset (dataset: Dataset<'S>) =
        dataset |> TrnValTst.ofDatasetWithRatios (0.8, 0.1, 0.1)

    /// Copies the given dataset to the specified device.
    static member transfer dev (this: TrnValTst<'S>) = {
        Trn = this.Trn |> Dataset.transfer dev
        Val = this.Val |> Dataset.transfer dev
        Tst = this.Tst |> Dataset.transfer dev
    }

    /// Saves this dataset to disk in an HDF5 file.
    /// HDF5 folders called 'Trn', 'Val' and 'Tst' are used to store the dataset parations.
    static member save filename (this: TrnValTst<'S>) =
        use hdf = HDF5.OpenWrite filename
        this.Trn.Save (hdf, "Trn")
        this.Val.Save (hdf, "Val")
        this.Tst.Save (hdf, "Tst")

    /// Loads a dataset from an HDF5 file.
    /// HDF5 folders called 'Trn', 'Val' and 'Tst' are used to store the dataset parations.
    static member load filename : TrnValTst<'S> = 
        use hdf = HDF5.OpenRead filename
        {
            Trn = Dataset.load (hdf, "Trn")
            Val = Dataset.load (hdf, "Val")
            Tst = Dataset.load (hdf, "Tst")
        }


