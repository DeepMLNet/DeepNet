open System
open System.Reflection

open ManagedCuda
open ManagedCuda.BasicTypes

open BenchmarkDotNet.Jobs
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Attributes.Jobs
open BenchmarkDotNet.Running

open Tensor
open Tensor.Utils


type FastRunConfig () as this =
    inherit ManualConfig ()

    let job = Job()
    //let job = job.WithIterationTime 100
    let job = job.WithLaunchCount 1
    let job = job.WithWarmupCount 1
    let job = job.WithTargetCount 2

    do this.Add job


type IBinaryWork =

    abstract Nothing: unit -> unit
    abstract Add: unit -> unit
    abstract Subtract: unit -> unit


type BinaryWork<'T> (dev, shape) =

    let nElems = shape |> List.fold (*) 1L

    let a = 
        Tensor<'T>.arange dev (conv<'T> 0L) (conv<'T> 1L) (conv<'T> nElems)
        |> Tensor.reshape shape
    let b = Tensor.copy -a

    interface IBinaryWork with
        member __.Nothing () = a |> ignore
        member __.Add () = 
            let c = a + b
            ()
        member __.Subtract () = a - b |> ignore


//[<Config(typeof<FastRunConfig>)>]
[<ShortRunJob>]
type BinaryOp () =

    let mutable worker = BinaryWork<int> (HostTensor.Dev, [1L]) :> IBinaryWork

    let mutable cudaContext : CudaContext = null

    let sync () =
        if not (isNull cudaContext) then
            cudaContext.Synchronize()

    //[<Params("int32", "int64", "single", "double")>]
    [<Params("single", "double")>]
    //[<Params("single")>]
    member val Type = "" with get, set

    //[<Params("Host", "Cuda")>]
    [<Params("Host")>]
    member val Dev = "" with get, set

    //[<Params("100x100", "1000x1000")>]
    //[<Params("1000x1000")>]
    [<Params("1000x1000", "2000x2000")>]
    member val Shape = "" with get, set

    [<GlobalSetup>]
    member this.Setup () =
        if this.Dev = "Cuda" then
            cudaContext <- new CudaContext(createNew=false)

        let typ =
            match this.Type with
            | "int32" -> typeof<int32>
            | "int64" -> typeof<int64>
            | "single" -> typeof<single>
            | "double" -> typeof<double>
            | _ -> failwithf "unknown data type: %s" this.Type
        let dev = 
            match this.Dev with
            | "Host" -> HostTensor.Dev
            | "Cuda" -> CudaTensor.Dev
            | _ -> failwithf "unknown device: %s" this.Dev
        let shape =
            this.Shape.Split('x') |> Seq.map Int64.Parse |> List.ofSeq

        let workerType = typedefof<BinaryWork<_>>.MakeGenericType typ
        worker <- Activator.CreateInstance (workerType, dev, shape) :?> IBinaryWork
        sync()

    //[<Benchmark>]
    //member __.Nothing () = worker.Nothing () ; sync ()

    [<Benchmark>]
    member __.Add () = worker.Add () ; sync ()

    //[<Benchmark>]
    //member __.Subtract () = worker.Subtract () ; sync ()
    

[<EntryPoint>]
let main argv = 
    let switcher = BenchmarkSwitcher (Assembly.GetExecutingAssembly())
    switcher.RunAll() |> ignore
    0

