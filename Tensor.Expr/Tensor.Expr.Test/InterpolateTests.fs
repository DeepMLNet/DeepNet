namespace global
#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.Ops
open Tensor.Cuda
open TestUtils



type InterpolateTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Simple 1D`` () =
        runOnAllDevs output (fun ctx ->
            let tbl = [1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f]
                        |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let minVal = 1.0
            let maxVal = 6.0

            let ip = 
                Interpolator.create tbl [minVal] [maxVal] 
                    [OutsideInterpolatorRange.Nearest] InterpolationMode.Linear None

            let nSmpls = SizeSym "nSmpls"
            let inp = Var<single> (ctx / "inp", [Size.sym nSmpls])
            let expr = Expr inp |> Expr.interpolate1D ip 

            let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                            |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let expVal = [ 1.0f; 1.0f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.0f; 6.0f]
                            |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let varEnv = VarEnv.ofSeq [inp, inpVal]
            let resVal = expr |> Expr.eval varEnv

            printfn "tbl=\n%A" tbl
            printfn "inp=\n%A" inpVal
            printfn "res=\n%A" resVal

            let resVal = HostTensor.transfer resVal
            let expVal = HostTensor.transfer expVal
            Tensor.almostEqual (resVal, expVal, absTol=0.005f, relTol=1e-5f) |> should equal true
        )


    [<Fact>]
    let ``Simple 2D`` () =
        runOnAllDevs output (fun ctx ->
            let tbl = [[1.0f; 2.0f; 3.0f]
                       [4.0f; 5.0f; 6.0f]
                       [7.0f; 8.0f; 9.0f]]
                      |> HostTensor.ofList2D |> Tensor.transfer ctx.Dev
            let minVal = [0.0; 0.0]
            let maxVal = [2.0; 2.0]

            let ip = 
                Interpolator.create tbl minVal maxVal 
                    [OutsideInterpolatorRange.Nearest; OutsideInterpolatorRange.Nearest] 
                    InterpolationMode.Linear None

            let nSmpls = SizeSym "nSmpls"
            let inp1 = Var<single> (ctx / "inp1", [Size.sym nSmpls])
            let inp2 = Var<single> (ctx / "inp2", [Size.sym nSmpls])
            let expr = Expr.interpolate2D ip (Expr inp1) (Expr inp2)

            let inpVal1 = [-0.1f; 0.0f; 0.5f; 1.5f; 2.0f; 2.3f;] |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let inpVal2 = [-0.1f; 0.0f; 0.8f; 4.5f; 2.0f; 2.3f;] |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let expVal =  [ 1.0f; 1.0f; 3.3f; 7.5f; 9.0f; 9.0f;] |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let varEnv = VarEnv.ofSeq [inp1, inpVal1; inp2, inpVal2]
            let resVal = expr |> Expr.eval varEnv

            printfn "tbl=\n%A" tbl
            printfn "inp1=\n%A" inpVal1
            printfn "inp2=\n%A" inpVal2
            printfn "res=\n%A" resVal

            let resVal = HostTensor.transfer resVal
            let expVal = HostTensor.transfer expVal
            Tensor.almostEqual (resVal, expVal, absTol=0.005f, relTol=1e-5f) |> should equal true
        )


    [<Fact>]
    let ``Derivative 1D`` () =
        runOnAllDevs output (fun ctx ->
            let tbl = [1.0f; 2.0f; 4.0f; 7.0f; 11.0f; 16.0f]
                        |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let minVal = 1.0
            let maxVal = 6.0

            let ip = 
                Interpolator.create tbl [minVal] [maxVal] 
                    [OutsideInterpolatorRange.Nearest] InterpolationMode.Linear None

            let nSmpls = SizeSym "nSmpls"
            let inp = Var<single> (ctx / "inp", [Size.sym nSmpls])
            let expr = Expr.interpolate1D ip (Expr inp)
            let dexpr = Deriv.compute expr
            let dinp = dexpr.Wrt inp

            let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                            |> HostTensor.ofList |> Tensor.transfer ctx.Dev
            let expVal = [ 0.0f; 0.0f; 1.0f; 1.0f; 2.0f; 5.0f; 0.0f; 0.0f; 0.0f]
                            |> HostTensor.ofList |> Tensor.diagMat |> Tensor.transfer ctx.Dev
            let varEnv = VarEnv.ofSeq [inp, inpVal]
            let resVal = dinp |> Expr.eval varEnv

            printfn "derivative:"
            printfn "tbl=\n%A" tbl
            printfn "inp=\n%A" inpVal
            printfn "res=\n%A" resVal

            let resVal = HostTensor.transfer resVal
            let expVal = HostTensor.transfer expVal
            Tensor.almostEqual (resVal, expVal, absTol=0.005f, relTol=1e-5f) |> should equal true
        )


