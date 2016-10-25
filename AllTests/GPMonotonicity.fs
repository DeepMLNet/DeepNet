module GPTest

open Xunit
open FsUnit.Xunit

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System
open Basics
open TestUtils
open Models
open MLPlots

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Monotonicity on Artificial Examples`` () =
    let nSmpls = 10
    let nInp = 20
    let rand =  Random(1234)
    let sampleX num = rand.SortedUniformArrayND (-3.0,3.0) [num] |> ArrayND.single |> ArrayNDHost.fetch
    let sampleEpsilon num = rand.NormalArrayND (0.0,sqrt(0.1)) [num] |> ArrayND.single |> ArrayNDHost.fetch
    let fctA x = if x < 0.5f then 0.0f else 2.0f
    let fctB x = 2.0f*x
    let fctC x = exp(1.5f*x)
    let fctD x = 2.0f/(1.0f+exp(-8.0f*x+ 4.0f))
    let mb = ModelBuilder<single> "GP"
    let nTrnSmpls = mb.Size "nTrnSmpls"
    let nInput = mb.Size "nInput"
    let sigNs = mb.Var<single> "sigNs" [nTrnSmpls]
    let x = mb.Var<single>  "x" [nTrnSmpls]
    let t = mb.Var<single>  "t" [nTrnSmpls]
    let inp = mb.Var<single>  "inp" [nInput]
    let fInp = mb.Var<single>  "Finp" [nInput] 
    let zeroMean x = Expr.zerosLike x
    let hyperPars1 = {GaussianProcess.Kernel =  GaussianProcess.SquaredExponential (1.0f,1.0f)
                                                GaussianProcess.MeanFunction = zeroMean
                                                GaussianProcess.Monotonicity = None
                                                GaussianProcess.CutOutsideRange = false}
    let pars1 = GaussianProcess.pars (mb.Module "GaussianProcess1") hyperPars1
    let hyperPars2 = {hyperPars1 with GaussianProcess.Monotonicity = Some (1e-6f,10,-3.0f,3.0f)}
    let pars2 = GaussianProcess.pars (mb.Module "GaussianProcess2") hyperPars2
    let mi = mb.Instantiate (DevCuda,
                            Map[nTrnSmpls, nSmpls
                                nInput,    nInp])
//    let x = ExpectationPropagation.normalize x
//    let t = ExpectationPropagation.normalize t
    let mean1, _ = GaussianProcess.predict pars1 x t sigNs inp
    let mean2, _ = GaussianProcess.predict pars2 x t sigNs inp
    let two = Expr.twoOfSameType mean1
    let RMSE1 = sqrt((mean1 - fInp) ** two|> Expr.mean)
    let RMSE2 = sqrt((mean2 - fInp) ** two|> Expr.mean)
    let rmse1Fun = mi.Func RMSE1 |>arg5 x t sigNs inp fInp
    let rmse2Fun = mi.Func RMSE2 |>arg5 x t sigNs inp fInp
    let mean1Fun = mi.Func mean1 |>arg5 x t sigNs inp fInp
    let mean2Fun = mi.Func mean2 |>arg5 x t sigNs inp fInp
    let runTest (func: single-> single) =
        let x = sampleX nSmpls
        let y = (ArrayND.map func x) + sampleEpsilon nSmpls
        let sigmas = (ArrayND.onesLike x) * sqrt(0.1f)
        let input =ArrayNDHost.linSpaced -3.0f 3.0f nInp |> ArrayNDCuda.toDev
        let fInput = (ArrayND.map func input) 
        let rmse1 = rmse1Fun x y sigmas input fInput|> ArrayNDHost.fetch
        let rmse2 = rmse2Fun x y sigmas input fInput|> ArrayNDHost.fetch
        printfn "x = \n %A \ny = \n %A \ninput = \n %A \nFinput = \n %A \n" x y input fInput
        printfn "mean1 =\n%A" (mean1Fun x y sigmas input fInput|> ArrayNDHost.fetch)
        printfn "mean2 =\n%A" (mean2Fun x y sigmas input fInput|> ArrayNDHost.fetch)
        printfn "rmse1  = %A\n rmse2  = %A" rmse1 rmse2
        rmse1.Data.[0] ,rmse2.Data.[0] 
        
    let rmsesFA1,rmsesFA2 = [0..9] |> List.map (fun _ -> (runTest fctA)) |> List.unzip 
    let rmsesFB1,rmsesFB2 = [0..9] |> List.map (fun _ -> (runTest fctB)) |> List.unzip 
    let rmsesFC1,rmsesFC2 = [0..9] |> List.map (fun _ -> (runTest fctC)) |> List.unzip 
    let rmsesFD1,rmsesFD2 = [0..9] |> List.map (fun _ -> (runTest fctD)) |> List.unzip 
    printfn "rmse function A: %7.4f  %7.4f" (List.average rmsesFA1) (List.average rmsesFA2)
    printfn "rmse function B: %7.4f  %7.4f" (List.average rmsesFB1) (List.average rmsesFB2)
    printfn "rmse function C: %7.4f  %7.4f" (List.average rmsesFC1) (List.average rmsesFC2)
    printfn "rmse function D: %7.4f  %7.4f" (List.average rmsesFD1) (List.average rmsesFD2)