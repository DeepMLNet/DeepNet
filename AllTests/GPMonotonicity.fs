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

[<Fact>]
let ``On Artificial Examples`` () =
    let nSmpls = 100
    let nInp = 500
    let rand =  Random()
    let sampleX num = rand.UniformArrayND (0.0,1.0) [num] |> ArrayND.single |> ArrayNDHost.fetch
    let sampleEpsilon num = rand.NormalArrayND (0.0,1.0) [num] |> ArrayND.single |> ArrayNDHost.fetch
    let normalize (mean:single) std (ary:ArrayNDHostT<single>) = ArrayND.map (fun x -> (x-mean)/std) ary
    let fctA x = if x < 0.5f then 0.0f else 2.0f
    let fctB x = 2.0f*x
    let fctC x = exp(1.5f*x)
    let fctD x = 2.0f/(1.0f+exp(-8.0f*x+ 4.0f))
    let mb = ModelBuilder<single> "GP"
    let nTrnSmpls = mb.Size "nTrnSmpls"
    let nInput = mb.Size "nInput"
    let sigNs = mb.Var "sigs" [nTrnSmpls]
    let x = mb.Var  "x" [nTrnSmpls]
    let t = mb.Var  "t" [nTrnSmpls]
    let inp = mb.Var  "inp" [nInput]
    let zeroMean x = Expr.zerosLike x
    let hyperPars1 = {GaussianProcess.Kernel =  GaussianProcess.SquaredExponential (1.0f,1.0f);
                                                GaussianProcess.MeanFunction = zeroMean;
                                                GaussianProcess.Monotonicity = None;
                                                GaussianProcess.CutOutsideRange = false}
    let pars1 = GaussianProcess.pars (mb.Module "GaussianProcess") hyperPars1
    let hyperPars2 = {hyperPars1 with GaussianProcess.Monotonicity = Some 1e-6f}
    let pars2 = GaussianProcess.pars (mb.Module "GaussianProcess") hyperPars2
    let mi = mb.Instantiate (DevCuda,
                            Map[nTrnSmpls, nSmpls
                                nInput,    nInp])
    let mean1, cov1 = GaussianProcess.predict pars1 x t sigNs inp
    let mean2, cov2 = GaussianProcess.predict pars2 x t sigNs inp
    let RMSE1 = sqrt(LossLayer.loss LossLayer.MSE mean1 inp)
    let RMSE2 = sqrt(LossLayer.loss LossLayer.MSE mean2 inp)
    let rmse1Fun = mi.Func RMSE1 |>arg4 x t sigNs inp
    let rmse2Fun = mi.Func RMSE1 |>arg4 x t sigNs inp
    let runTest (func: single-> single) =
        let x = sampleX nSmpls
        let y = (ArrayND.map func x) + sampleEpsilon nSmpls
        let x = normalize 0.0f 0.5f x |> ArrayNDCuda.toDev
        let y = normalize 0.0f 0.5f y |> ArrayNDCuda.toDev
        let sigmas = ArrayND.zerosLike x 
        let input = ArrayNDHost.linSpaced -3.0f 3.0f nInp |> ArrayNDCuda.toDev
        let rmse1 = rmse1Fun x y sigmas input |> ArrayNDHost.fetch
        let rmse2 = rmse1Fun x y sigmas input |> ArrayNDHost.fetch
        ArrayNDHost.toList rmse1 |> List.head, ArrayNDHost.toList rmse2 |> List.head
    let rmsesFA1,rmsesFA2 = [0..49] |> List.map (fun _ -> (runTest fctA)) |> List.unzip 
    let rmsesFB1,rmsesFB2 = [0..49] |> List.map (fun _ -> (runTest fctB)) |> List.unzip 
    let rmsesFC1,rmsesFC2 = [0..49] |> List.map (fun _ -> (runTest fctC)) |> List.unzip 
    let rmsesFD1,rmsesFD2 = [0..49] |> List.map (fun _ -> (runTest fctD)) |> List.unzip 
    printfn "rmse function A: %7.4f  %7.4f" (List.average rmsesFA1) (List.average rmsesFA2)
    printfn "rmse function B: %7.4f  %7.4f" (List.average rmsesFB1) (List.average rmsesFB2)
    printfn "rmse function C: %7.4f  %7.4f" (List.average rmsesFC1) (List.average rmsesFC2)
    printfn "rmse function D: %7.4f  %7.4f" (List.average rmsesFD1) (List.average rmsesFD2)