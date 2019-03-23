namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr
open Tensor.Host
open Tensor.Cuda
open TestUtils


type TracerTestVals (dev: ITensorDevice) =
    let ctx = Context.root dev

    let a = Var<float> (ctx / "a", [Size.fix 2L; Size.fix 3L])
    let b = Var<float> (ctx / "b", [Size.fix 2L; Size.fix 3L])

    let aVal = HostTensor.counting 6L |> Tensor<float>.convert |> Tensor.reshape [2L; 3L] |> Tensor.transfer dev
    let bVal = 10L + HostTensor.counting 6L |> Tensor<float>.convert |> Tensor.reshape [2L; 3L] |> Tensor.transfer dev

    let data = HostTensor.counting 10L |> Tensor<float>.convert |> Tensor.transfer dev

    let varEnv =
        VarEnv.empty
        |> VarEnv.add a aVal
        |> VarEnv.add b bVal

    member this.A = a
    member this.B = b
    member this.Data = data
    member this.VarEnv = varEnv


type HDF5TracerTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let getTracer (dev: ITensorDevice) name =
        let devStr =
            match dev with
            | :? TensorHostDevice -> "host"
            | :? TensorCudaDevice -> "cuda"
            | _ -> dev.Id
        let hdf = HDF5.OpenWrite (sprintf "%s_%s.h5" name devStr)
        HDF5Tracer (hdf)

    let simpleExpr dev =
        let v = TracerTestVals dev
        let expr = Expr v.A + Expr v.B
        let evalEnv: Ops.EvalEnv = {VarEnv=v.VarEnv; Tracer=getTracer dev "simpleExpr"}
        expr.Untyped |> UExpr.evalWithEnv evalEnv

    let exprWithData dev =
        let v = TracerTestVals dev
        let expr = 10.0 * Expr v.Data + 5.0
        let evalEnv: Ops.EvalEnv = {VarEnv=v.VarEnv; Tracer=getTracer dev "exprWithData"}
        expr.Untyped |> UExpr.evalWithEnv evalEnv        

    [<Fact>]
    let ``CPU: simpleExpr`` () =
        simpleExpr HostTensor.Dev
    [<CudaFact>]
    let ``Cuda: simpleExpr`` () =
        simpleExpr CudaTensor.Dev

    [<Fact>]
    let ``CPU: exprWithData`` () =
        exprWithData HostTensor.Dev
    [<CudaFact>]
    let ``Cuda: exprWithData`` () =
        exprWithData CudaTensor.Dev

        