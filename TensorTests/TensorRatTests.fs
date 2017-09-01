module TensorRatTests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor


[<Fact>]
let ``Basic Rat operations`` () =
    let a = Rat (2, 3)
    let b = Rat (7, 8)
    let c = Rat (10)
    let d = Rat (0, 5)
    let n = Unchecked.defaultof<Rat>

    printfn "zero n=%A" n
    printfn "d=%A" d
    printfn "a=%A b=%A c=%A" a b c
    printfn "a+b=%A" (a+b)
    printfn "a-b=%A" (a-b)
    printfn "a*b=%A" (a*b)
    printfn "a/b=%A" (a/b)
    printfn "b mod a=%A" (b%a)

    n |> should equal Rat.Zero
    a + b - b |> should equal a
    a * b / b |> should equal a
    a / a |> should equal Rat.One
    b - b |> should equal Rat.Zero

    match a with
    | RatFrac(p, q) ->
        printfn "a= %A / %A" p q
        p |> should equal (bigint 2)
        q |> should equal (bigint 3)
    | RatInteger _ -> failwith "a is not an integer"

    match c with
    | RatFrac _ -> failwith "c is not a fraciton"
    | RatInteger p ->
        p |> should equal (bigint 10)


[<Fact>]
let ``Rat Tensors`` () =
    let A = HostTensor.zeros<Rat> [3L; 3L]
    let B = HostTensor.ones<Rat> [3L; 3L]
    let C = Rat 3 * B
    let D = HostTensor.filled [3L; 3L] (Rat (2,3))

    printfn "A=\n%A" A
    printfn "B=\n%A" B
    printfn "C=3*B=\n%A" C
    printfn "D=\n%A" D
    printfn "C+D=\n%A" (C+D)

    printfn "C ==== C=\n%A" (C ==== C)
    printfn "A >>== B=\n%A" (A >>== B)
    printfn "sum C= %A" (Tensor.sum C)


[<Fact>]
let ``Rat Conversions`` () =
    let v = 3
    
    let rv = conv<Rat> v
    let bv = conv<int> rv

    printfn "v=%A rv=%A bv=%A" v rv bv

    bv |> should equal v











    