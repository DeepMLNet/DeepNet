namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Expr
open Utils



type DerivTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Deriv: a + b`` () =
        printfn "==== Deriv a+b:"
        let expr = Expr Vars.a + Expr Vars.b
        let derivs = Deriv.compute expr
        printfn "wrt a: %A" derivs.[Vars.a]  
        printfn "wrt b: %A" derivs.[Vars.b]

    [<Fact>]
    let ``Deriv: sin a * exp b`` () =
        printfn "==== Deriv sin a * exp b:"
        let expr = sin (Expr Vars.a) * exp (Expr Vars.b)
        let derivs = Deriv.compute expr
        printfn "wrt a: %s" (derivs.[Vars.a].ToString())
        printfn "wrt b: %s" (derivs.[Vars.b].ToString())

