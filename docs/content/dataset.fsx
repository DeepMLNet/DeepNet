(*** hide ***)
#load "../../DeepNet.fsx"

(**

Dataset Handling
================

Deep.Net provides a generic type for handling datasets used in machine learning.
It can handle samples that are of a user-defined record type containing fields of type ArrayNDT.
The following features are provided:

  * data storage on host and CUDA GPU
  * indexed sample access
  * sample range access
  * mini-batch sequencing (with optional padding of last batch)
  * partitioning into training, validation and test set
  * loading from and saving to disk

We are going to introduce it using a simple, synthetic dataset.


Creating a dataset
------------------
In most cases you are going to load a dataset by parsing some text or binary files.
However, since this is quite application-specific we do not want to concern ourselves with it here and will create a synthetic dataset using trigonometric functions on the fly.

### Defining the sample type
Our sample type consists of two fields: a scalar $x$ and a vector $\mathbf{v}$.
This corresponds to the following record type
*)

open ArrayNDNS

type MySampleType = {
    X:      ArrayNDT<single>
    V:      ArrayNDT<single>
}

(**
We use the data type `single` for fast arithmetic operations on the GPU.

### Generating some samples
Next, let us generate some samples.
The scalar $x$ shall be sampled randomly from a uniform distribution on the interval $[-2, 2]$.
The values of vector $v$ shall be given by the relation

$$$
\mathbf{v}(x) = \left( \begin{matrix} \mathrm{sinh} \, x \\ \mathrm{cosh} \, x \end{matrix} \right)

We can implement that using the following code.
*)

let generateSamples cnt = seq {
    let rng = System.Random (100)
    for n = 0 to cnt - 1 do
        let x = 2. * (rng.NextDouble () - 0.5) * 2. |> single
        yield {
            X = ArrayNDHost.scalar x
            V = ArrayNDHost.ofList [sinh x; cosh x]
        }    
}

(**
The generateSamples function produces the specified number of samples.
We can test it as follows.
*)

let smpls = generateSamples 100 |> List.ofSeq

for idx, smpl in List.indexed smpls do
    printfn "Sample %3d: X=%A    V=%A" idx smpl.X smpl.V

(**
This prints

    Sample   0: X=   1.8751    V=[   3.1841    3.3374]
    Sample   1: X=  -1.3633    V=[  -1.8265    2.0824]
    Sample   2: X=   0.6673    V=[   0.7179    1.2310]
    Sample   3: X=   1.6098    V=[   2.4010    2.6009]
    ...
    Sample  99: X=  -0.1610    V=[  -0.1617    1.0130]

Now that we have some data, we can create a dataset.

### Instantiating the dataset type
There are two ways to construct a dataset.

  1. The `Dataset<'S>.FromSamples` takes a sequence of samples (of type 'S) and constructs a dataset from them.
  2. The `Dataset<'S>` constructor takes a list of ArrayNDTs corresponding to the fields of the record type 'S. The first dimension of each passed array must correspond to the sample index.

Since we already have a sequence of sample, we use the first method.
*)

open Datasets

let ds = smpls |> Dataset.FromSamples

(**

Accessing single and multiple elements 
--------------------------------------

The dataset type supports the indexing and [slicing](https://blogs.msdn.microsoft.com/chrsmith/2008/12/09/f-zen-array-slices/) operations to access samples.

When accessing a single sample using the indexing operator we obtain a record from the sequence of samples we passed into the `Dataset.FromSamples` methods.
For example to print the third sample we write
*)

let smpl2 = ds.[2]
printfn "Sample 3: X=%A    V=%A" smpl2.X smpl2.V

(**
and get the output

    Sample 3: X=   0.6673    V=[   0.7179    1.2310]

When accessing multiple elements using the slicing operator, the returned value is of the same sample record type but the contained tensors have one additional dimension on the left corresponding to the sample index.
For example we can get a record containing the first three sample using the following code.
*)

let smpl0to2 = ds.[0..2]
printfn "Samples 0,1,2:\nX=%A\nV=\n%A" smpl0to2.X smpl0to2.V

(**
This prints

    Samples 0,1,2:
    X=[   1.8751   -1.3633    0.6673]
    V=
    [[   3.1841    3.3374]
     [  -1.8265    2.0824]
     [   0.7179    1.2310]]

Hence all tensors in the sample record raise in rank by one dimension, i.e. the scalar `X` became a vector and the vector `V` became a matrix with each row corresponding to a sample.


Iterating over the dataset
--------------------------

You can also iterate over the samples of the dataset directly.
*)

for smpl in ds do
    printfn "Sample: %A" smpl

(**
This prints

    Sample: {X =    1.8751;
     V = [   3.1841    3.3374];}
    Sample: {X =   -1.3633;
     V = [  -1.8265    2.0824];}
    Sample: {X =    0.6673;
     V = [   0.7179    1.2310];}
    Sample: {X =    1.6098;
     V = [   2.4010    2.6009];}
    ...
    Sample: {X =   -0.1610;
     V = [  -0.1617    1.0130];}



Mini-batches
------------

The `ds.Batches` function returns a sequence of mini-batches from the dataset.
It takes one argument specifying the number of samples in each batch.
If the total number of samples in the dataset is not a multiple of the batch size, the last batch will have less samples.

The following code prints the sizes of the obtained mini-batches.
*)

for idx, batch in Seq.indexed (ds.Batches 30) do
    printfn "Batch %d: shape of X: %A    shape of V: %A" 
        idx batch.X.Shape batch.V.Shape

(**
This outputs

    Batch 0: shape of X: [30]    shape of V: [30; 2]
    Batch 1: shape of X: [30]    shape of V: [30; 2]
    Batch 2: shape of X: [30]    shape of V: [30; 2]
    Batch 3: shape of X: [10]    shape of V: [10; 2]

If you need the last batch to be padded to the specified batch size, use the `ds.PaddedBatches` method instead.


Partitioning
------------

It is often necessary to split a dataset into partitions.

The `ds.Partition` methods takes a list of ratios and returns a list of new datasets obtained by splitting the dataset according to the specified ratios.
Partitioning is done by sequentially taking samples from the beginning, until the first partition has the requested number of samples.
Then the samples for the second partition are taken and so on.

The following example splits our dataset into three partitions of ratios $1/2$, $1/4$ and $1/4$.
*)

let partitions = ds.Partition [0.5; 0.25; 0.25]

for idx, p in List.indexed partitions do
    printfn "Partition %d has %d samples." idx p.NSamples

(**
This prints

    Partition 0 has 50 samples.
    Partition 1 has 25 samples.
    Partition 2 has 25 samples.


### Training, validation and test splits

In machine learning it is common practice to split the dataset into a training, validation and test dataset.
Deep.Net provides the `TrnValTst<'S>` type for that purpose.
It is a record type with the fields `Trn`, `Val` and `Tst` of type `Dataset<'S>`.
It can be constructed from an existing dataset using the `TrnValTst.Of` function.

The following code demonstrates its use using the ratios $0.7$, $0.15$ and $0.15$ for the train, validation and test set respectively.
The ratio specification is optional; if it is omitted ratios of $0.8$, $0.1$ and $0.1$ are used.
*)

let dsp = TrnValTst.Of (ds, 0.7, 0.15, 0.15)

printfn "Training set size:    %d" dsp.Trn.NSamples
printfn "Validation set size:  %d" dsp.Val.NSamples
printfn "Test set size:        %d" dsp.Tst.NSamples

(**
This prints

    Training set size:    70
    Validation set size:  15
    Test set size:        15


Data transfer
-------------

The `ds.ToCuda` and `ds.ToHost` methods copy the dataset to the CUDA GPU or to the host respectively.
The TrnValTst type provides the same methods.


Disk storage
------------

Use the `ds.Save` method to save a dataset to disk using the HDF5 format.
The `Dataset<'S>.Load` function loads a saved dataset.
The TrnValTst type provides the same methods.


Summary
=======

The `Dataset<'S>` type provides a convenient way to work with datasets.
Type-safety is provided by preserving the user-specified sample type `'S` when accessing individual or multiple samples.

*)
