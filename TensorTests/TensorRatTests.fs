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
    | RatInt _ -> failwith "a is not an integer"

    match c with
    | RatFrac _ -> failwith "c is not a fraciton"
    | RatInt p ->
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
let ``Rat Dot`` () =
    let A = HostTensor.identity<Rat> 5L
    A.[2L, *] <- (Rat 2) * A.[2L, *]
    let B = HostTensor.linspace (Rat 0) (Rat 5) 5L
    let B = Tensor.replicate 0 5L B.[NewAxis, *]

    printfn "A=\n%A" A
    printfn "B=\n%A" B
    printfn "A.*B=\n%A" (A .* B)
    printfn "A.*B (float)=\n%A" (Tensor.convert<float> A .* Tensor.convert<float> B)

    let C1 = Tensor.convert<float> (A .* B)
    let C2 = Tensor.convert<float> A .* Tensor.convert<float> B
    Tensor.almostEqual C1 C2 |> should equal true

    let C1 = Tensor.convert<float> (A .* B.[0L, *])
    let C2 = Tensor.convert<float> A .* Tensor.convert<float> B.[0L, *]
    Tensor.almostEqual C1 C2 |> should equal true

    let C1 = Tensor.convert<float> (Tensor.padLeft A .* Tensor.padLeft B)
    let C2 = Tensor.convert<float> (Tensor.padLeft A) .* Tensor.convert<float> (Tensor.padLeft B)
    Tensor.almostEqual C1 C2 |> should equal true

[<Fact>]
let ``Rat Conversions`` () =
    let v = 3   
    let rv = conv<Rat> v
    let bv = conv<int> rv
    printfn "v=%A rv=%A bv=%A" v rv bv
    bv |> should equal v

[<Fact>]
let ``Rat Infinites`` () =
    Rat.PosInf + Rat.PosInf |> Rat.isPosInf |> should equal true
    Rat.PosInf + Rat.NegInf |> Rat.isNaN |> should equal true
    Rat.NegInf + Rat.PosInf |> Rat.isNaN |> should equal true
    Rat.NegInf + Rat.NegInf |> Rat.isNegInf |> should equal true
    Rat.One + Rat.NaN |> Rat.isNaN |> should equal true
    Rat.PosInf * Rat.Zero |> Rat.isNaN |> should equal true
    Rat.PosInf * Rat.NegInf |> Rat.isNegInf |> should equal true

[<Fact>]
let ``Rat Ordering`` () =
    let l = [Rat.MinusOne; Rat.One; Rat.Zero; Rat.NaN
             Rat.PosInf; Rat.NegInf; Rat.PosInf; Rat.NegInf; Rat.NaN]
    let ls = List.sort l
    printfn "sorted: %A" ls
    (ls.[0].IsNaN && ls.[1].IsNaN && ls.[2].IsNegInf && ls.[3].IsNegInf &&
     ls.[4] = Rat.MinusOne && ls.[5] = Rat.Zero && ls.[6] = Rat.One &&
     ls.[7].IsPosInf && ls.[8].IsPosInf) |> should equal true


[<Fact>]
let ``Rat Tensor Conversions`` () =
    let A = HostTensor.arange 1 2 10
    let B = A |> Tensor.convert<Rat>
    let C = B |> Tensor.convert<int32>
    printfn "A=\n%A" A
    printfn "B=\n%A" B
    printfn "C=\n%A" C
    C ==== A |> Tensor.all |> should equal true
     

[<Fact>]
let ``Rat Tensor Arange`` () =
    let A = HostTensor.arange (Rat(1)) (Rat(1,3)) (Rat(3))
    printfn "arange:\n%A" A


[<Fact>]
let ``Rat Tensor Linspace`` () =
    let A = HostTensor.linspace (Rat 1) (Rat 3) 10L
    printfn "linspace:\n%A" A
     








    