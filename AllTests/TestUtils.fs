module TestUtils

open System.IO
open Xunit
open FsUnit.Xunit

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

let randomEval shps exprFn (dev: IDevice) =
    let rng = System.Random(123)

    let vars = 
        [for idx, shp in List.indexed shps do
            let name = sprintf "v%d" idx
            let sshp = shp |> List.map SizeSpec.fix
            yield Expr.var name sshp]
    let expr = exprFn vars
    let fn = Func.make dev.DefaultFactory expr

    let varEnv = 
        (VarEnv.empty, List.zip vars shps)
        ||> List.fold (fun varEnv (var, shp) ->
            let value = rng.UniformArrayND (-1.0f, 1.0f) shp |> dev.ToDev
            varEnv |> VarEnv.add var value
        )
    fn varEnv |> ignore

let requireEqualTracesWithRandomData shps exprFn =
    compareTraces (randomEval shps exprFn) false
    |> should equal 0
