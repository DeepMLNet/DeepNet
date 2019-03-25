module TestUtils

open System.IO
open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Expr
open Tensor.Backend
open Tensor.Expr.Ops
open Tensor.Cuda


let dumpExpr (output: ITestOutputHelper) (expr: UExpr) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 
    printfn "Expr: %s" (expr.ToString())
    printfn "==== DataType:           %A" expr.DataType
    printfn "==== Device:             %A" expr.Dev
    printfn "==== Shape:              %A" expr.Shape
    printfn "==== CanEvalAllSymSizes: %A" expr.CanEvalAllSymSizes
    printfn "==== Vars:               %A" expr.Vars
    printfn ""



//let dumpTrace filename trace = 
//    let txtPath = Path.GetFullPath (filename + ".txt")
//    let hdfPath = Path.GetFullPath (filename + ".h5")
//    trace |> Trace.dumpToFile txtPath hdfPath
//    printfn "Dumped trace to %s and %s" txtPath hdfPath

let rec exprDepth (root: BaseExpr) (part: BaseExpr) =
    if root = part then Some 0
    else
        root.Args
        |> Map.toSeq
        |> Seq.tryPick (fun (_, exprCh) -> 
            exprDepth exprCh.Expr part |> Option.map ((+) 1))
        

let compareTraces (output: ITestOutputHelper) (fn: ITensorDevice -> ITracer -> unit) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let devs = [
        yield "Host", HostTensor.Dev
        if TensorCudaDevice.count > 0 then yield "Cuda", CudaTensor.Dev
    ]    
    if List.length devs < 2 then
        failwith "At least two tensor devices must be available for trace comparison."

    // perform traces
    let tracePath = Path.GetTempFileName()
    (
        use traceHdf = HDF5.OpenWrite tracePath
        for devName, dev in devs do
            printfn "Evaluating on %s..." devName
            let tracer = HDF5Tracer (traceHdf, devName)
            fn dev tracer
    )

    // compare traces
    let mutable diffsExist = false
    (
        use traceHdf = HDF5.OpenRead tracePath
        let masterDevName, _ = List.head devs
        let masterTrace = HDF5Trace (traceHdf, masterDevName)
        let masterExpr = masterTrace.Root
        
        printfn ""
        printfn "Expression:\n%A" masterExpr

        for devName, _ in List.tail devs do
            let devTrace = HDF5Trace (traceHdf, devName)
            let diffs = 
                HDF5Trace.diff (masterTrace, devTrace) |> List.ofSeq
                |> List.sortByDescending (fun d -> exprDepth masterExpr d.AExpr)
            if not (List.isEmpty diffs) then
                diffsExist <- true
                printfn ""
                printfn "Trace differs between devices %s and %s:" masterDevName devName
                for d in diffs do
                    printfn ""
                    printfn "Subexpression %A:" d.AExpr
                    printfn "Channel: %A" d.Ch
                    printfn "Value on %A:\n%A" masterDevName d.AValue
                    printfn "Value on %A:\n%A" devName d.BValue
                    printfn ""
            else
                printfn ""
                printfn "Trace between devices %s and %s match." masterDevName devName
    )

    File.Delete tracePath
    diffsExist

//let evalHostCuda func =
//    printfn "Evaluating on host..."
//    func DevHost
//    printfn "Evaluating on CUDA device..."
//    func DevCuda
//    printfn "Done."

let buildVars (ctx: Context) typShps = [
    for idx, (typ, shp) in List.indexed typShps do
        let name = ctx / sprintf "v%d" idx
        let sshp = 
            shp 
            |> List.map (function | -1L -> Size.broadcastable
                                  | s -> Size.fix s)
        let var = Var.make (name, typ, sshp)
        yield var
]

let buildVarEnv (rng: System.Random) (vars: Var list)  =
    (VarEnv.empty, vars)
    ||> List.fold (fun varEnv var ->
        let shp = Shape.eval var.Shape
        let value =
            if var.DataType = typeof<int> || var.DataType = typeof<int64> then
                HostTensor.randomInt rng (0, 100) shp
                |> ITensor.convertToType var.DataType
            elif var.DataType = typeof<bool> then
                HostTensor.randomInt rng (0, 1) shp ==== 1 :> ITensor
            else
                HostTensor.randomUniform rng (-10.0, 10.0) shp 
                |> ITensor.convertToType var.DataType
        varEnv |> VarEnv.addBaseVar var (value |> ITensor.transfer var.Dev))

let randomTraceEval typShps exprFn (dev: ITensorDevice) (tracer: ITracer)  =
    let ctx = Context.root dev
    let rng = System.Random 123
    let vars = buildVars ctx typShps
    let varEnv = buildVarEnv rng vars
    let varExprs = vars |> List.map UExpr
    let expr = exprFn varExprs
    let evalEnv = {VarEnv=varEnv; Tracer=tracer}
    expr |> UExpr.evalWithEnv evalEnv |> ignore
   

let requireEqualTraces output evalFn =
    compareTraces output evalFn |> should equal false

let requireEqualTracesWithRandomData output typShps (exprFn: UExpr list -> UExpr) =
    compareTraces output (randomTraceEval typShps exprFn) |> should equal false
 

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

