namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor
open Tensor.Algorithm
open Tensor.Utils
open DeepNet.Utils


type RatTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``a + b - b = a`` () =
        let a = Rat (2, 3)
        let b = Rat (7, 8)
        printfn "a=%A b=%A" a b 

        let c = a + b
        printfn "c=%A" c

        let d = c - b
        printfn "d=%A" d

        printfn "d.Num=%A  d.Dnm=%A" d.Num d.Dnm
        printfn "a.Num=%A  a.Dnm=%A" a.Num a.Dnm
        printfn "a.Num=d.Num: %A   a.Dnm=d.Dnm: %A" (a.Num = d.Num) (a.Dnm = d.Dnm)
        printfn "a.IsNaN: %A    d.IsNaN: %A" a.IsNaN d.IsNaN

        printfn "d=a: %A" (d = a)
        d |> should equal a


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
        printfn ""
        printfn "a+b-b=%A" (a+b-b)
        printfn "a+b-b = a = %A" ((a+b-b) = a)

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
        | _ -> failwith "a is not infinite"

        match c with
        | RatFrac _ -> failwith "c is not a fraciton"
        | RatInt p -> p |> should equal (bigint 10)
        | _ -> failwith "c is not infinite"


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
        printfn "A.*B (float)=\n%A" (Tensor<float>.convert A .* Tensor<float>.convert B)

        let C1 = Tensor<float>.convert (A .* B)
        let C2 = Tensor<float>.convert A .* Tensor<float>.convert B
        Tensor.almostEqual (C1, C2) |> should equal true

        let C1 = Tensor<float>.convert (A .* B.[0L, *])
        let C2 = Tensor<float>.convert A .* Tensor<float>.convert B.[0L, *]
        Tensor.almostEqual (C1, C2) |> should equal true

        let C1 = Tensor<float>.convert (Tensor.padLeft A .* Tensor.padLeft B)
        let C2 = Tensor<float>.convert (Tensor.padLeft A) .* Tensor<float>.convert (Tensor.padLeft B)
        Tensor.almostEqual (C1, C2) |> should equal true

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
        let B = A |> Tensor<Rat>.convert
        let C = B |> Tensor<int32>.convert
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
         

    [<Fact>]
    let ``Floor and Ceiling`` () =
        let doTest a =
            let af = float a
            
            printfn "floor:"
            printfn "Rat:     a=%A   floor a=%A   floor (-a)=%A" a (floor a) (floor -a)
            printfn "float:   a=%A   floor a=%A   floor (-a)=%A" af (floor af) (floor -af)
            floor af |> should equal (float (floor a))
            floor -af |> should equal (float (floor -a))

            printfn "ceil:"
            printfn "Rat:     a=%A   ceil a=%A   ceil (-a)=%A" a (ceil a) (ceil -a)
            printfn "float:   a=%A   ceil a=%A   ceil (-a)=%A" a (ceil af) (ceil -af)
            ceil af |> should equal (float (ceil a))
            ceil -af |> should equal (float (ceil -a))

            printfn "truncate:"
            printfn "Rat:     a=%A   truncate a=%A   truncate (-a)=%A" a (truncate a) (truncate -a)
            printfn "float:   a=%A   truncate a=%A   truncate (-a)=%A" a (truncate af) (truncate -af)
            truncate af |> should equal (float (truncate a))
            truncate -af |> should equal (float (truncate -a))            

        doTest (Rat(1, 10))
        doTest (Rat(0, 10))
        doTest (Rat(10, 10))
        doTest (Rat(9, 10))
        doTest (Rat(12, 10))









        