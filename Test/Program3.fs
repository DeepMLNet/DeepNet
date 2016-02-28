module Program3

open Basics
open ArrayNDNS

open SymTensor
open Models
open Datasets


let ``Test slice`` () =
    let ary : ArrayNDT<single> = ArrayNDHost.ones [5; 7; 4]
    //printfn "ary=\n%A" ary

    let slc1 = ary.[0..1, 1..3, 2..4]
    printfn "slc1=\n%A" slc1

    let slc1b = ary.[0..1, 1..3, *]
    printfn "slc1b=\n%A" slc1b

    let slc2 = ary.[1, 1..3, 2..4]
    printfn "slc2=\n%A" slc2

    let ary2 : ArrayNDT<single> = ArrayNDHost.ones [5; 4]
    //printfn "ary2=\n%A" ary2

    let slc3 = ary2.[NewAxis, 1..3, 2..4]
    printfn "slc3=\n%A" slc3

    let slc4 = ary2.[Fill, 1..3, 2..4]
    printfn "slc4=\n%A" slc4

    ary2.[NewAxis, 1..3, 2..4] <- slc3



let ``Convert MNIST to HDF5`` () =
    let mnist = Mnist.load @"C:\Local\surban\dev\fexpr\Data\MNIST"
    use hdf = new HDF5 (@"C:\Local\surban\dev\fexpr\Data\MNIST\MNIST.h5", HDF5Overwrite)

    ArrayNDHDF.write hdf "TrnImgs" mnist.TrnImgs
    ArrayNDHDF.write hdf "TrnLbls" mnist.TrnLbls
    ArrayNDHDF.write hdf "TstImgs" mnist.TstImgs
    ArrayNDHDF.write hdf "TstLbls" mnist.TstLbls



[<EntryPoint>]
let main argv = 
    
    ``Test slice`` ()

    ``Convert MNIST to HDF5`` ()

    0

