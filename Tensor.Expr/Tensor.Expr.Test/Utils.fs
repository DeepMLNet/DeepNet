module Utils

open System.IO
open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Expr
//open Models
//open Datasets
//open Optimizers


let dumpExpr (output: ITestOutputHelper) (expr: Expr) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 
    printfn "Expr: %s" (expr.ToString())
    printfn "==== DataType:           %A" expr.DataType
    printfn "==== Device:             %A" expr.Dev
    printfn "==== Shape:              %A" expr.Shape
    printfn "==== CanEvalAllSymSizes: %A" expr.CanEvalAllSymSizes
    printfn "==== Vars:               %A" expr.Vars
    printfn ""



//let post device (x: Tensor<'T>) =
//    if device = DevCuda then CudaTensor.transfer x
//    else x 
    
//let compareTracesLock = obj ()

//let dumpTrace filename trace = 
//    let txtPath = Path.GetFullPath (filename + ".txt")
//    let hdfPath = Path.GetFullPath (filename + ".h5")
//    trace |> Trace.dumpToFile txtPath hdfPath
//    printfn "Dumped trace to %s and %s" txtPath hdfPath

//let compareTraces func dump =
//    printfn "Evaluating on CUDA device..."
//    use traceHndl = Trace.startSession "CUDA"
//    func DevCuda
//    let cudaTrace = traceHndl.End()
//    if dump then cudaTrace |> dumpTrace "CUDA"

//    printfn "Evaluating on host..."
//    use traceHndl = Trace.startSession "Host"
//    func DevHost
//    let hostTrace = traceHndl.End ()
//    if dump then hostTrace |> dumpTrace "Host"

//    let diffs = Trace.compare hostTrace cudaTrace
//    if diffs > 0 then
//        printfn "Traces differ. Dumping to UneqalCUDA.{txt,h5} and UneqalHost.{txt,h5}."
//        cudaTrace |> dumpTrace "UnequalCUDA" 
//        hostTrace |> dumpTrace "UnequalHost"
//    diffs

//let evalHostCuda func =
//    printfn "Evaluating on host..."
//    func DevHost
//    printfn "Evaluating on CUDA device..."
//    func DevCuda
//    printfn "Done."

//[<RequiresExplicitTypeArguments>]
//let buildVars<'T> shps = 
//    [for idx, shp in List.indexed shps do
//        let name = sprintf "v%d" idx
//        let sshp = 
//            shp 
//            |> List.map (function | -1L -> SizeSpec.broadcastable
//                                  | s -> SizeSpec.fix s)
//        yield Expr.var<'T> name sshp]

//[<RequiresExplicitTypeArguments>]
//let buildVarEnv<'T> (vars: ExprT list) shps (rng: System.Random) (dev: IDevice) =
//    (VarEnv.empty, List.zip vars shps)
//    ||> List.fold (fun varEnv (var, shp) ->
//        let shp = shp |> List.map (function | -1L -> 1L | s -> s)
//        let value = rng.UniformTensor (conv<'T> -1.0, conv<'T> 1.0) shp |> dev.ToDev
//        varEnv |> VarEnv.add var value
//    )

//[<RequiresExplicitTypeArguments>]
//let randomEval<'T, 'R> shps exprFn (dev: IDevice) =
//    let rng = System.Random(123)
//    let vars = buildVars<'T> shps
//    let expr = exprFn vars
//    let fn = Func.make<'R> dev.DefaultFactory expr
//    let varEnv = buildVarEnv<'T> vars shps rng dev
//    fn varEnv |> ignore

//let requireEqualTraces evalFn =
//    compareTraces evalFn false
//    |> should equal 0

//let requireEqualTracesWithRandomData shps (exprFn: ExprT list -> ExprT) =
//    compareTraces (randomEval<single, single> shps exprFn) false
//    |> should equal 0

//let requireEqualTracesWithRandomDataLogic shps (exprFn: ExprT list -> ExprT) =
//    compareTraces (randomEval<single, bool> shps exprFn) false
//    |> should equal 0

//let requireEqualTracesWithRandomDataIdx shps (exprFn: ExprT list -> ExprT) =
//    compareTraces (randomEval<single, int64> shps exprFn) false
//    |> should equal 0

//let randomDerivativeCheckTree device tolerance shps (exprFn: ExprT list -> ExprT) =
//    let rng = System.Random(123)
//    let vars = buildVars<float> shps
//    let expr = exprFn vars
//    let varEnv = buildVarEnv<float> vars shps rng device
//    DerivCheck.checkExprTree device tolerance 1e-7 varEnv expr

//let randomDerivativeCheckTreeOnHost = randomDerivativeCheckTree DevHost
//let randomDerivativeCheckTreeOnCuda = randomDerivativeCheckTree DevCuda

//let randomDerivativeCheck device tolerance shps (exprFn: ExprT list -> ExprT) =
//    let rng = System.Random(123)
//    let vars = buildVars<float> shps
//    let expr = exprFn vars
//    let varEnv = buildVarEnv<float> vars shps rng device
//    DerivCheck.checkExpr device tolerance 1e-7 varEnv expr

//let randomDerivativeCheckOnHost = randomDerivativeCheck DevHost
//let randomDerivativeCheckOnCuda = randomDerivativeCheck DevCuda

