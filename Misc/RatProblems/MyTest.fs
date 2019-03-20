module MyTest

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor.Algorithm

type TestType (output: ITestOutputHelper) =

    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let test () =
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
        assert (d = a)


//[<EntryPoint>]
//let main argv =
//    let a = Rat (2, 3)
//    let b = Rat (7, 8)
//    printfn "a=%A b=%A" a b 

//    let c = a + b
//    printfn "c=%A" c

//    let d = c - b
//    printfn "d=%A" d

//    printfn "d.Num=%A  d.Dnm=%A" d.Num d.Dnm
//    printfn "a.Num=%A  a.Dnm=%A" a.Num a.Dnm
//    printfn "a.Num=d.Num: %A   a.Dnm=d.Dnm: %A" (a.Num = d.Num) (a.Dnm = d.Dnm)
//    printfn "a.IsNaN: %A    d.IsNaN: %A" a.IsNaN d.IsNaN

//    printfn "d=a: %A" (d = a)
//    assert (d = a)

//    0 // return an integer exit code
