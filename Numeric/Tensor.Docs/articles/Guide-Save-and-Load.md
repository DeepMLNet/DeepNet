# Save and load

## Disk storage in HDF5 format

Tensors can be stored in industry-standard [HDF5 files](https://en.wikipedia.org/wiki/Hierarchical_Data_Format).
Multiple tensors can be stored in a single HDF5 file and accessed by assigning names to them.

### Writing tensors to disk
The following code creates two tensors `k` and `l` and writes them into the HDF5 file `tensors.h5` in the current directory.

```fsharp
open Basics // TODO: remove after refactoring
let k = ArrayNDHost.initIndexed [5; 3] (fun [i; j] -> 3. * float i + float j)
let l = ArrayNDHost.initIndexed [5] (fun [i] -> 2. * float i)
let hdfFile = HDF5.OpenWrite "tensors.h5"
ArrayNDHDF.write hdfFile "k" k
ArrayNDHDF.write hdfFile "l" l
hdfFile.Dispose ()
```

The resulting file can be viewed using any HDF5 viewer, for example using the free, cross-platform [HDFView](https://www.hdfgroup.org/products/java/hdfview/) application as shown below.

![HDFView screenshot](img/hdfview.png)

### Loading tensors from disk
The following code loads the tensors `k` and `l` from the previously created HDF5 file `tensors.h5` and stores them in the variables `k2` and `l2`.

```fsharp
let hdfFile2 = HDF5.OpenRead "tensors.h5"
let k2 : ArrayNDHostT<float> = ArrayNDHDF.read hdfFile2 "k"
let l2 : ArrayNDHostT<float> = ArrayNDHDF.read hdfFile2 "l" 
hdfFile2.Dispose ()
```

The data types of `k2` and `l2` must be declared explicitly, since they must be known at compile-time.
If the declared data type does not match the data type encountered in the HDF5, an error will be raised.


## Reading .npy and .npz files produced by Numpy

For compatibility, it is possible to read `.npy` and `.npz` files produced by Numpy.
Not all features of the format are supported.
Writing `.npy` and `.npz` files is not possible; use the HDF5 format instead.

Use the `NPYFile.load` function to read an `.npy` file and return its contents as an `ArrayNDHostT`.
Use the `NPZFile.Open` function to open an `.npz` file and the `Get` method of the resulting object to obtain individual entries as `ArrayNDHostT`.
