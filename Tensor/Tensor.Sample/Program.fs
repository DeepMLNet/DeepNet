// Basic tutorial of the F# tensor library.
// Read the full documentation at http://www.deepml.net/Tensor

#nowarn "25"

open Tensor

let header title =
    printfn ""
    printfn "============================================================================="
    printfn "%s" title
    printfn "============================================================================="

[<EntryPoint>]
let main argv =

    // ==================================================================================
    header "Tensor creation and transfer"
    // ==================================================================================

    // Create a matrix of shape 3x2 and type int filled with zeros in host memory.
    let z1 = Tensor<int>.zeros HostTensor.Dev [3L; 2L]
    printfn "z1=\n%A" z1
    printfn "z1.Dev=%A" z1.Dev

    // Alternate (shorter) form for the above operation.
    let z1 = HostTensor.zeros<int> [3L; 2L]

    // Create a vector of length 3 of type single filled with ones in host memory.
    let o1 = Tensor<single>.ones HostTensor.Dev [3L]
    printfn "o1=\n%A" o1

    // Create an identity 3x3 matrix of type float in host memory.
    let id1 = Tensor<float>.identity HostTensor.Dev 3L
    printfn "id1=\n%A" id1

    // Create a scalar tensor with value 33.2.
    let s1 = Tensor.scalar HostTensor.Dev 33.2
    printfn "s1=%A" s1
    printfn "The numeric value of s1 is %f." s1.Value

    // Initialize a 7x5 matrix using a function.
    let a = HostTensor.init [7L; 5L] (fun [|i; j|] -> 5.0 * float i + float j)
    printfn "a=\n%A" a

    // Create a vector from an F# sequence.
    let seq1 = seq { for i=0 to 20 do if i % 3 = 0 then yield i } |> HostTensor.ofSeq
    printfn "The tensor seq1 is\n%A" seq1

    // Create a vector with a specified increment between elements.
    let arange1 = HostTensor.arange 5.0 0.1 6.0
    printfn "arange1=%A" arange1

    // Create a vector with a fixed increment and specified number of elements.
    let linspace1 = HostTensor.linspace 2.0 4.0 30L
    printfn "linspace1=%A" linspace1

    // Transfer a tensor to the GPU.
    // This works only if you have a CUDA-capable GPU and CUDA SDK 8.0 is installed.
    try
        let m = seq {1 .. 10} |> HostTensor.ofSeq
        let mGpu = CudaTensor.transfer m
        printfn "mGpu=\n%A" mGpu
        printfn "mGpu.Dev=%A" mGpu.Dev
    with e ->
        printfn "CUDA error: %A" e.Message
        printfn "This is normal if no CUDA GPU is present."

    // ==================================================================================
    header "Elements and slicing"
    // ==================================================================================

    // Read element 1,1 (zero-based) of tensor a.
    let v = a.[[1L; 1L]]
    printfn "v=%f" v

    // Set element 2,2 (zero-based) of tensor a to 55.0.
    a.[[2L; 2L]] <- 55.0
    printfn "a (after assignment)=\n%A" a

    // Get a view of the first row of tensor a.
    let a1 = a.[0L, *]
    printfn "a1=%A" a1

    // Changing an element in the view a1 also effects the tensor a.
    a1.[[1L]] <- 99.0
    printfn "a1 (after assignment)=%A" a1
    printfn "a (after assignment to a1)=\n%A" a
    
    // Setting multiple elements at once by assigning a tensor.
    let a2 = HostTensor.ones<float> [5L]
    printfn "a2=%A" a2
    a.[0L, *] <- a2
    printfn "a (after assignment of a2)=\n%A" a

    // ==================================================================================
    header "Shape operations"
    // ==================================================================================

    // Getting the shape of tensor a.
    printfn "a has shape %A, rank %d and %d elements." a.Shape a.NDims a.NElems

    // Reshaping a 4x4 matrix into a vector of length 16.
    let b = HostTensor.init [4L; 4L] (fun [|y; x|] -> 4 * int y + int x)
    printfn "b=\n%A" b
    let b1 = Tensor.reshape [16L] b
    printfn "b1=%A" b1

    // Reshaping a 4x4 matrix into a 4x2x2 tensor.
    let b2 = Tensor.reshape [4L; 2L; Remainder] b
    printfn "b2=\n%A" b2

    // Transposing matrix b by explicitly swapping its axes.
    let b3 = Tensor.swapDim 0 1 b
    printfn "b3=\n%A" b3

    // Using the .T property for transposition.
    printfn "b.T=\n%A" b.T

    // Permuting the axes of tensor b2.
    let b4 = Tensor.permuteAxes [2; 0; 1] b2
    printfn "b4.Shape=%A" b4.Shape

    // Inserting an dimension of size one into tensor b4.
    let b5 = b4.[*, *, NewAxis, *]
    printfn "b5.Shape=%A" b5.Shape

    // Broadcasting a size one dimension to size 3.
    let c = HostTensor.init [1L; 4L] (fun [|_; i|] -> int i)
    printfn "c=\n%A" c
    let c1 = Tensor.broadcastTo[3L; 4L] c
    printfn "c1=\n%A" c1

    // Multiple elements in a broadcasted tensor share the same memory.
    // Thus assigning to one element will change all elements along a 
    // broadcasted axis as well as the orinal tensor that was broadcasted.
    c1.[[1L; 1L]] <- 11
    printfn "c1 (after assignment)=\n%A" c1
    printfn "c (after assignment to c1)=\n%A" c

    // ==================================================================================
    header "Tensor operations"
    // ==================================================================================

    // Arithmetic operators (+, -, *, /, %, **) work element-wise on all elements 
    // of the two tensors involved.
    let d = HostTensor.init [4L] (fun [|i|] -> float i)
    let e = HostTensor.init [4L] (fun [|i|] -> 10.0 * float i)
    let f = d + e
    printfn "f=\n%A" f

    // It is also possible to perform arithmetic operations with a tensor and a
    // primitive data type, for example a float.
    let d1 = d * 100.0
    printfn "d1=\n%A" d1

    // If the tensors of a binary operation have different shapes, automatic broadcasting rules apply.
    //
    // 1. The tensor with less dimensions is padded from the left with size 1 dimensions,
    //    until both tensors have the same number of dimensions.
    // 2. If one tensor has size 1 in a dimensions and the other tensor has a different size
    //    in that dimension, the size 1 tensor is broadcasted to the size of the the other tensor.
    let a = HostTensor.init [4L; 3L; 1L] (fun [|i; j; k|] -> float i + 0.1 * float j)
    let b = HostTensor.init [2L] (fun [|i|] -> 10.0 * float i)
    let apb = a + b
    printfn "a=\n%A" a
    printfn "b=\n%A" b
    printfn "apb=\n%A" apb
    printfn "apb.Shape=%A" apb.Shape

    // Standard F# arithmetic functions (sin, cos, exp, round, floor, etc.) also work element-wise.
    let f2 = sin f
    printfn "f2=%A" f2
    let f2b = round f
    printfn "f2b=%A" f2b

    // You can explicitly specify the destination tensor for the result of all tensor operations.
    // Use the Fill* methods for that, which can be found under "see also" in the reference.
    let f3 = HostTensor.ofList [-1.0; -1.0; 29.0; 40.0]    
    f3.FillMultiply d e
    printfn "f3 (after FillMultiply)=%A" f3

    // This can also be used to perform operations in-place to save memory.
    // You might find this useful when working with very large tensors 
    // and limited memory or want to avoid unnecessary copies.
    f3.FillMultiply f3 e
    printfn "f3 (after 2nd FillMultiply)=%A" f3

    // Use the .* operator to calculate the dot product of two tensors.
    // This is also called the matrix product.
    // If applied to a matrix and a vector, the matrix-vector product will be computed.
    // If applied to two vectors, the scalar product will be computed.
    let h = HostTensor.init [5L; 3L] (fun [|i; j|] -> 3.0f * single i + single j)
    let i = 0.1f + HostTensor.identity 3L
    let hi = h .* i
    printfn "hi=\n%A" hi

    // If both tensors are stored on the GPU, then the dot product is
    // evaluated directly on the GPU using CUBLAS.
    try
        let hGpu = CudaTensor.transfer h
        let iGpu = CudaTensor.transfer i
        let hiGpu = hGpu .* iGpu
        printfn "hiGpu=%A" hiGpu
        printfn "hiGpu.Dev=%A" hiGpu.Dev
    with e ->
        printfn "CUDA error: %A" e.Message
        printfn "This is normal if no CUDA GPU is present."    

    // We can also use double-precision data types for dot products.
    let fh = Tensor<float>.convert h
    let fi = Tensor<float>.convert i
    let fhi = fh .* fi
    printfn "fhi=\n%A" fhi

    // Of course, this also works on GPUs.
    // However, must GPUs are significantly slower when using double-precision arithmetic.
    // Thus you might want to stick to single-precision floats when you plan to use GPU acceleration.
    try
        let fhGpu = CudaTensor.transfer fh
        let fiGpu = CudaTensor.transfer fi
        let fhiGpu = fhGpu .* fiGpu
        printfn "fhiGpu=%A" fhiGpu
        printfn "fhiGpu.Dev=%A" fhiGpu.Dev  
    with e ->
        printfn "CUDA error: %A" e.Message
        printfn "This is normal if no CUDA GPU is present."    

    // Matrix inversion
    let iinv = Tensor.invert i
    printfn "iinv=\n%A" iinv

    // The F# functions map, mapi, map2, mapi2, fold are also directly available, but only
    // for tensors stored in host memory.
    let f3 = HostTensor.map (fun x -> if x > 15.0 then 7.0 + x else -1.0) f
    printfn "f3=%A" f3

    // ==================================================================================
    header "Reduction operations"
    // ==================================================================================

    // The sum functions sums all elements of a tensor.
    let s1 = Tensor.sum f
    printfn "s1=%f" s1

    // The sumAxis function sums all elements along an axis.
    // Here it is used to calculate the sums of each row of the matrix g.
    let g = HostTensor.init [4L; 4L] (fun [|y; x|] -> 4 * int y + int x)
    let s2 = Tensor.sumAxis 0 g
    printfn "s2=%A" s2

    // Likewise, we can use maxAxis to find the maximum element of each row of matrix g.
    let m2 = Tensor.maxAxis 0 g
    printfn "m2=%A" m2

    // ==================================================================================
    header "Comparison and logic operations"
    // ==================================================================================

    // The operators ====, <<<<, <<==, >>>>, >>==, <<>> perform element-wise comparison
    // of two tensors and return a tensor of data type bool.
    let d = HostTensor.ofList [0;  1; 2;  3]
    let e = HostTensor.ofList [0; 10; 2; 30]
    let j = d ==== e
    printfn "j=%A" j

    // The operators ~~~~, &&&&, ||||, ^^^^ work on boolean tensors only and perform
    // the logical operations negation, and, or, xor respectively.
    let nj = ~~~~j
    printfn "nj=%A" nj
    let jnj = j &&&& nj
    printfn "jnj=%A" jnj
    let joj = j |||| nj
    printfn "joj=%A" joj

    // You can get the indices of all true entries in a boolean tensor by using the
    // trueIdx function. It returns a tensor of type int64 filled with indices.    
    let a = HostTensor.ofList2D [[true; false; true; false]
                                 [false; true; true; false]]
    let b = Tensor.trueIdx a
    printfn "b=\n%A" b

    // Boolean tensors can be used to element-wise switch between the elements
    // of two tensors. If cond is true, the corresponding element is taken from
    // ifTrue. Otherwise, it is taken from ifFalse.
    let cond = HostTensor.ofList [true; false; false]
    let ifTrue = HostTensor.ofList [2.0; 3.0; 4.0]
    let ifFalse = HostTensor.ofList [5.0; 6.0; 7.0]
    let t = Tensor.ifThenElse cond ifTrue ifFalse
    printfn "t=%A" t

    // A boolean mask can be used to select only the elements from a tensor
    // for which the corresponding entry in the boolean tensor is true.
    let a = HostTensor.ofList2D [[1.0; 2.0; 3.0]
                                 [4.0; 5.0; 6.0]]
    let m = HostTensor.ofList2D [[true;  true;  false]
                                 [false; false; true ]]
    let b = a.M(m)
    printfn "b=%A" b 

    // It is also possible to selectively mask a subset of the dimensions.
    let m0 = HostTensor.ofList [true; false]
    let e = a.M(m0, NoMask)
    printfn "e=\n%A" e

    // Masking can also be used to replace elements within a tensor.
    // This is done by assigning to the .M property.
    let m = HostTensor.ofList2D [[true;  true;  false]
                                 [false; false; true ]]
    a.M(m) <- HostTensor.ofList [8.0; 9.0; 0.0]
    printfn "a (after masked assignment)=\n%A" a

    // The all functions checks if all elements within a boolean tensor are true.
    let aj = Tensor.all j
    printfn "aj=%b" aj

    // The allAxis function checks if all elements along an axis are true.
    // It returns a tensor with one dimension less than the input tensor.
    let g = HostTensor.ofList2D [[true; false; false;  true]
                                 [true; false;  true; false]]
    let ga = Tensor.allAxis 0 g    
    printfn "ga=%A" ga

    // The any functions checks if any element within a boolean tensor is true.
    // There exists a corresponding anyAxis function.
    let anyj = Tensor.any j
    printfn "anyj=%b" anyj

    // ==================================================================================
    header "Index functions"
    // ==================================================================================

    // Use the allIdx function to get a sequence of indices that sequentially enumerates
    // all elements of a tensor.
    let a = HostTensor.zeros<int> [2L; 3L]
    let s = Tensor.allIdx a
    printfn "s=%A" s

    // The argMax function returns the index of the largest element of a tensor.
    let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
                                 [5.0; 6.0; 7.0; 8.0]]
    let bArgMax = Tensor.argMax a    
    printfn "bArgMax=%A" bArgMax

    // The argMaxAxis finds the largest element along the specified axis within a tensor.
    // It returns a tensor of indices.
    let bArgMaxAxis = Tensor.argMaxAxis 1 a
    printfn "bArgMaxAxis=%A" bArgMaxAxis

    // The tryFind functions finds the first occurence of an element.
    let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
                                 [5.0; 6.0; 7.0; 3.0]]
    let bFind = Tensor.tryFind 3.0 a    
    printfn "bFind=%A" bFind

    // The complimentary findAxis functions searches for the element along the
    // specified axis. Here it is used to find the index of the element 3.0
    // within each row of matrix a.
    let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
                                 [5.0; 6.0; 7.0; 3.0]]
    let bFindAxis = Tensor.findAxis 3.0 1 a
    printfn "bFindAxis=%A" bFindAxis

    // Use the gather function to collect elements from a source tensor.
    // The indices are specified by index tensors of type int64 with
    // one tensor per dimension of the source tensor.
    let src = HostTensor.ofList2D [[0.0; 0.1; 0.2; 0.3]
                                   [1.0; 1.1; 1.2; 1.3]
                                   [2.0; 2.1; 2.2; 2.3]]
    let i0 = HostTensor.ofList [1L; 2L; 0L; 0L]
    let i1 = HostTensor.ofList [3L; 1L; 0L; 3L]
    let g = Tensor.gather [Some i0; Some i1] src    
    printfn "g=%A" g

    // If None is specified instead of an index tensor, the source index
    // in that dimensions matches the target index.
    let j1 = HostTensor.ofList [3L; 1L; 0L]
    let g2 = Tensor.gather [None; Some j1] src
    printfn "g2=%A" g2

    // The scatter functions work like the gather function, however here
    // the index tensor does not specify where an element should be read
    // from but where it should be written to.
    // If a destination index occurs multiple times, all elements that are
    // written to it are summed implicitly.
    let src = HostTensor.ofList2D [[0.0; 0.1; 0.2; 0.3]
                                   [1.0; 1.1; 1.2; 1.3]
                                   [2.0; 2.1; 2.2; 2.3]]
    let i0 = HostTensor.ofList2D [[0L; 0L; 0L; 0L]
                                  [2L; 2L; 2L; 2L]
                                  [1L; 1L; 1L; 1L]]
    let i1 = HostTensor.ofList2D [[3L; 3L; 3L; 3L]
                                  [0L; 1L; 2L; 3L]
                                  [0L; 1L; 2L; 3L]]
    let s = Tensor.scatter [Some i0; Some i1] [4L; 4L] src
    printfn "s=\n%A" s

    // ==================================================================================
    header "Save and load"
    // ==================================================================================

    // HDF5 is an industry-standard format for exchange of numerical data.
    // Multiple tensors can be stored in a single HDF5 file.
    // Here, we write tensors k and l to the file tensors.h5.
    let k = HostTensor.init [5L; 3L] (fun [|i; j|] -> 3.0 * float i + float j)
    let l = HostTensor.init [5L] (fun [|i|] -> 2.0 * float i)
    using (HDF5.OpenWrite "tensors.h5") (fun hdfFile ->
        HostTensor.write hdfFile "k" k
        HostTensor.write hdfFile "l" l)

    // Now, we reopen the file for reading and read the two tensors back.
    use hdfFile2 = HDF5.OpenRead "tensors.h5"
    let k2 = HostTensor.read<float> hdfFile2 "k"
    let l2 = HostTensor.read<float> hdfFile2 "l"    
    printfn "k2=\n%A" k2
    printfn "l2=\n%A" l2 

    // Get HDFView from https://support.hdfgroup.org/products/java/hdfview/
    // to explore the HDF5 file interactively.

    0 
