module ArrayNDTests

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open Datasets


[<Fact>]
let ``Basic arithmetic works`` () =
    let v1val : ArrayNDHostT<float> = ArrayNDHost.ones [3]
    let v2val = ArrayNDHost.scalar 3.0 |> ArrayND.padLeft |> ArrayND.broadcastToShape [3]
    let v3val = -v2val
    let v4val = v1val + v3val

    printfn "v1val: %A" v1val.[[0]]
    printfn "v2val: %A" v2val.[[0]]
    printfn "v3val: %A" v3val.[[0]]
    printfn "v4val: %A" v4val.[[0]]
    

[<Fact>]
let ``Slicing works`` () =
    let ary : ArrayNDHostT<single> = ArrayNDHost.ones [5; 7; 4]
    //printfn "ary=\n%A" ary

    let slc1 = ary.[0..1, 1..3, 2..4]
    printfn "slc1=\n%A" slc1

    let slc1b = ary.[0..1, 1..3, *]
    printfn "slc1b=\n%A" slc1b

    let slc2 = ary.[1, 1..3, 2..4]
    printfn "slc2=\n%A" slc2

    let ary2 : ArrayNDHostT<single> = ArrayNDHost.ones [5; 4]
    //printfn "ary2=\n%A" ary2

    let slc3 = ary2.[NewAxis, 1..3, 2..4]
    printfn "slc3=\n%A" slc3

    let slc4 = ary2.[Fill, 1..3, 2..4]
    printfn "slc4=\n%A" slc4

    ary2.[NewAxis, 1..3, 2..4] <- slc3

[<Fact>]
let ``Pretty printing works`` () =
    printfn "3x4 one matrix:       \n%A" (ArrayNDHost.ones [3; 4] :> ArrayNDT<float>)
    printfn "6 zero vector:        \n%A" (ArrayNDHost.zeros [6] :> ArrayNDT<float>)
    printfn "5x5 identity matrix:  \n%A" (ArrayNDHost.identity 5 :> ArrayNDT<float>)

[<Fact>]
let ``Batched matrix-matrix dot product`` () =
    let N, M = 2, 3
    let rng = System.Random(123)
    let a = rng.UniformArrayND (-3., 3.) [N; M; 4; 3]
    let b = rng.UniformArrayND (-1., 1.) [N; M; 3; 2]
    let c = a .* b

    let cr = ArrayNDHost.zeros<float> [N; M; 4; 2]
    for n = 0 to N - 1 do
        for m = 0 to M - 1 do
            cr.[n, m, Fill] <- a.[n, m, Fill] .* b.[n, m, Fill]
    ArrayND.almostEqual c cr |> ArrayND.value |> Assert.True
    
[<Fact>]
let ``Batched matrix-vector dot product`` () =
    let N, M = 2, 3
    let rng = System.Random(123)
    let a = rng.UniformArrayND (-3., 3.) [N; M; 4; 3]
    let b = rng.UniformArrayND (-1., 1.) [N; M; 3]
    let c = a .* b

    let cr = ArrayNDHost.zeros<float> [N; M; 4]
    for n = 0 to N - 1 do
        for m = 0 to M - 1 do
            cr.[n, m, Fill] <- a.[n, m, Fill] .* b.[n, m, Fill]
    ArrayND.almostEqual c cr |> ArrayND.value |> Assert.True
    
