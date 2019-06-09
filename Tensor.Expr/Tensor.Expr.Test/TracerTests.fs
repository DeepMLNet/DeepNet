namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr
open Tensor.Expr.Base
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

    let performTraceTest (dev: ITensorDevice) name expr varEnv =
        let devStr =
            match dev with
            | :? TensorHostDevice -> "host"
            | :? TensorCudaDevice -> "cuda"
            | _ -> dev.Id
        let hdfName = sprintf "%s_%s.h5" name devStr

        // write trace
        printfn "Writing trace %s on %A" name dev
        (
            use hdf = HDF5.OpenWrite hdfName
            let tracer = HDF5Tracer hdf    
            let evalEnv: Base.EvalEnv = {VarEnv=varEnv; Tracer=tracer}
            expr |> UExpr.evalWithEnv evalEnv |> ignore
        )

        // read trace
        printfn "Reading trace"
        (
            use hdf = HDF5.OpenRead hdfName
            let trace = HDF5Trace hdf
            printfn "Root expression: %A" trace.Root
            let data = trace.[trace.Root]
            printfn "Trace data for root:\n%A" data
            printfn "Channel values:\n%A" data.ChVals
        )

    let simpleExpr dev =
        let v = TracerTestVals dev
        let expr = Expr v.A + Expr v.B
        performTraceTest dev "simpleExpr" expr.Untyped v.VarEnv

    let exprWithData dev =
        let v = TracerTestVals dev
        let expr = 10.0 * Expr v.Data + 5.0
        performTraceTest dev "exprWithData" expr.Untyped v.VarEnv  

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

        