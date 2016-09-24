module TestUtils

open System.IO
open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers


let post device (x: ArrayNDT<'T>) =
    if device = DevCuda then ArrayNDCuda.toDev (x :?> ArrayNDHostT<'T>) :> ArrayNDT<'T>
    else x 
    
let compareTracesLock = obj ()

let compareTraces func dump =
    printfn "Evaluating on CUDA device..."
    use traceHndl = Trace.startSession "CUDA"
    func DevCuda
    let cudaTrace = traceHndl.End()
    if dump then
        use tw = File.CreateText("CUDA.txt")
        Trace.dump tw cudaTrace

    printfn "Evaluating on host..."
    use traceHndl = Trace.startSession "Host"
    func DevHost
    let hostTrace = traceHndl.End ()
    if dump then
        use tw = File.CreateText("Host.txt")
        Trace.dump tw hostTrace

    Trace.compare hostTrace cudaTrace

let evalHostCuda func =
    printfn "Evaluating on host..."
    func DevHost
    printfn "Evaluating on CUDA device..."
    func DevCuda
    printfn "Done."

[<RequiresExplicitTypeArguments>]
let buildVars<'T> shps = 
    [for idx, shp in List.indexed shps do
        let name = sprintf "v%d" idx
        let sshp = 
            shp 
            |> List.map (function | -1 -> SizeSpec.broadcastable
                                  | s -> SizeSpec.fix s)
        yield Expr.var<'T> name sshp]

[<RequiresExplicitTypeArguments>]
let buildVarEnv<'T> (vars: ExprT list) shps (rng: System.Random) (dev: IDevice) =
    (VarEnv.empty, List.zip vars shps)
    ||> List.fold (fun varEnv (var, shp) ->
        let shp = shp |> List.map (function | -1 -> 1  | s -> s)
        let value = rng.UniformArrayND (conv<'T> -1.0, conv<'T> 1.0) shp |> dev.ToDev
        varEnv |> VarEnv.add var value
    )

[<RequiresExplicitTypeArguments>]
let randomEval<'T> shps exprFn (dev: IDevice) =
    let rng = System.Random(123)
    let vars = buildVars<'T> shps
    let expr = exprFn vars
    let fn = Func.make<'T> dev.DefaultFactory expr
    let varEnv = buildVarEnv<'T> vars shps rng dev
    fn varEnv |> ignore

let requireEqualTracesWithRandomData shps (exprFn: ExprT list -> ExprT) =
    compareTraces (randomEval<single> shps exprFn) false
    |> should equal 0

let randomDerivativeCheck tolerance shps (exprFn: ExprT list -> ExprT) =
    let rng = System.Random(123)
    let vars = buildVars<float> shps
    let expr = exprFn vars
    let varEnv = buildVarEnv<float> vars shps rng DevHost
    DerivCheck.checkExprTree tolerance 1e-7 varEnv expr

