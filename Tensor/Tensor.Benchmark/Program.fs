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


// type FastRunConfig () as this =
//     inherit ManualConfig ()
//     let job = Job()
//     //let job = job.WithIterationTime 100
//     let job = job.WithLaunchCount 1
//     let job = job.WithWarmupCount 1
//     let job = job.WithTargetCount 2
//     do this.Add job


type IWorker =
    abstract Nothing: unit -> unit
    abstract Zeros: unit -> unit
    abstract Ones: unit -> unit
    abstract Arange: unit -> unit
    abstract Copy: unit -> unit
    abstract Negate: unit -> unit
    abstract Add: unit -> unit
    abstract Subtract: unit -> unit
    abstract Multiply: unit -> unit
    abstract Divide: unit -> unit
    abstract Power: unit -> unit
    abstract Dot: unit -> unit
    abstract Sgn: unit -> unit
    abstract Log: unit -> unit
    abstract Log10: unit -> unit
    abstract Exp: unit -> unit
    abstract Sin: unit -> unit
    abstract Cos: unit -> unit
    abstract Tan: unit -> unit
    abstract Asin: unit -> unit
    abstract Acos: unit -> unit
    abstract Atan: unit -> unit
    abstract Sinh: unit -> unit
    abstract Cosh: unit -> unit
    abstract Tanh: unit -> unit
    abstract Sqrt: unit -> unit
    abstract Ceiling: unit -> unit
    abstract Floor: unit -> unit
    abstract Round: unit -> unit
    abstract Truncate: unit -> unit
    abstract MaxElemwise: unit -> unit
    abstract MinElemwise: unit -> unit
    abstract Equal: unit -> unit
    abstract NotEqual: unit -> unit
    abstract Less: unit -> unit
    abstract LessOrEqual: unit -> unit
    abstract Greater: unit -> unit
    abstract GreaterOrEqual: unit -> unit
    abstract Not: unit -> unit
    abstract And: unit -> unit
    abstract Or: unit -> unit
    abstract Xor: unit -> unit
    abstract CountTrueAxis: unit -> unit
    abstract IfThenElse: unit -> unit
    abstract SumAxis: unit -> unit
    abstract ProductAxis: unit -> unit
    abstract MaxAxis: unit -> unit
    abstract MinAxis: unit -> unit
    abstract AllAxis: unit -> unit
    abstract AnyAxis: unit -> unit
    abstract ArgMaxAxis: unit -> unit
    abstract ArgMinAxis: unit -> unit
    abstract FindAxis: unit -> unit
    abstract MaskedGet: unit -> unit
    abstract MaskedSet: unit -> unit
    abstract TrueIndices: unit -> unit
    abstract Invert: unit -> unit
    abstract PseudoInvert: unit -> unit
    abstract SVD: unit -> unit
    abstract SymEigDec: unit -> unit
    

type Worker<'T> (dev, shape) =
    let lastAxis = List.length shape - 1
    let nElems = shape |> List.fold (*) 1L

    let rng = Random 123
    let rndNumbers = Seq.initInfinite (fun _ -> rng.NextDouble() * 100. - 50. |> conv<'T>)
    let rndBools = Seq.initInfinite (fun _ -> rng.NextDouble() >= 0.5)

    let a = rndNumbers |> HostTensor.ofSeqWithShape shape |> Tensor.transfer dev
    let b = rndNumbers |> HostTensor.ofSeqWithShape shape |> Tensor.transfer dev
    let p = rndBools |> HostTensor.ofSeqWithShape shape |> Tensor.transfer dev
    let q = rndBools |> HostTensor.ofSeqWithShape shape |> Tensor.transfer dev
    
    let maskedSetTarget = Tensor<'T>.zeros dev shape
    let maskedSetElems = 
        if dev = HostTensor.Dev then a.M(p)
        else Tensor<'T>.zeros dev shape // masking currently unsupported on CUDA

    interface IWorker with
        member __.Nothing () = let c = a in ()
        member __.Zeros () = let c = Tensor<'T>.zeros dev shape in ()
        member __.Ones () = let c = Tensor<'T>.ones dev shape in ()
        member __.Arange () = let c = Tensor.arange dev (conv<'T> 0L) (conv<'T> 1L) (conv<'T> nElems) in ()
        member __.Copy () = let c = Tensor.copy a in ()
        member __.Negate () = let c = -a in ()
        member __.Add () = let c = a + b in ()
        member __.Subtract () = let c = a - b in ()
        member __.Multiply () = let c = a * b in ()
        member __.Divide () = let c = a / b in ()
        member __.Power () = let c = a ** b in ()
        member __.Dot () = let c = a .* b in ()
        member __.Sgn () = let c = sgn a in ()
        member __.Log () = let c = log a in ()
        member __.Log10 () = let c = log10 a in ()
        member __.Exp () = let c = exp a in ()
        member __.Sin () = let c = sin a in ()
        member __.Cos () = let c = cos a in ()
        member __.Tan () = let c = tan a in ()
        member __.Asin () = let c = asin a in ()
        member __.Acos () = let c = acos a in ()
        member __.Atan () = let c = atan a in ()
        member __.Sinh () = let c = sinh a in ()
        member __.Cosh () = let c = cosh a in ()
        member __.Tanh () = let c = tanh a in ()
        member __.Sqrt () = let c = sqrt a in ()
        member __.Ceiling () = let c = ceil a in ()
        member __.Floor () = let c = floor a in ()
        member __.Round () = let c = round a in ()
        member __.Truncate () = let c = truncate a in ()
        member __.MaxElemwise () = let c = Tensor.maxElemwise a b in ()
        member __.MinElemwise () = let c = Tensor.minElemwise a b in ()
        member __.Equal () = let c = a ==== b in ()
        member __.NotEqual () = let c = a <<>> b in ()
        member __.Less () = let c = a <<<< b in ()
        member __.LessOrEqual () = let c = a <<== b in ()
        member __.Greater () = let c = a >>>> b in ()
        member __.GreaterOrEqual () = let c = a >>== b in ()
        member __.Not () = let c = ~~~~p  in ()
        member __.And () = let c = p &&&& q in ()
        member __.Or () = let c = p |||| q in ()
        member __.Xor () = let c = p ^^^^ q in ()
        member __.CountTrueAxis () = let c = Tensor.countTrueAxis lastAxis p in ()
        member __.IfThenElse () = let c = Tensor.ifThenElse p a b in ()
        member __.SumAxis () = let c = Tensor.sumAxis lastAxis a in ()
        member __.ProductAxis () = let c = Tensor.productAxis lastAxis a in ()
        member __.MaxAxis () = let c = Tensor.maxAxis lastAxis a in ()
        member __.MinAxis () = let c = Tensor.minAxis lastAxis a in ()
        member __.AllAxis () = let c = Tensor.allAxis lastAxis p in ()
        member __.AnyAxis () = let c = Tensor.anyAxis lastAxis p in ()
        member __.ArgMaxAxis () = let c = Tensor.argMaxAxis lastAxis a in ()
        member __.ArgMinAxis () = let c = Tensor.argMinAxis lastAxis a in ()
        member __.FindAxis () = let c = Tensor.findAxis (conv<'T> 0) lastAxis a in ()
        member __.MaskedGet () = let c = a.M(p) in ()
        member __.MaskedSet () = maskedSetTarget.M(p) <- maskedSetElems
        member __.TrueIndices () = let c = Tensor.trueIdx p in ()
        member __.Invert () = let c = Tensor.invert a in ()
        member __.PseudoInvert () = let c = Tensor.pseudoInvert a in ()
        member __.SVD () = let u, s, v = Tensor.SVD a in ()
        member __.SymEigDec () = let vals, vecs = Tensor.symmetricEigenDecomposition MatrixPart.Upper a in ()


//[<Config(typeof<FastRunConfig>)>]
[<ShortRunJob>]
type Bench () =

    let mutable worker = Worker<int> (HostTensor.Dev, [1L]) :> IWorker

    let mutable cudaContext : CudaContext = null

    let sync () =
        if not (isNull cudaContext) then
            cudaContext.Synchronize()

    //[<Params("int32", "int64", "single", "double")>]
    //[<Params("single")>]
    [<Params("single", "double")>]
    member val Type = "" with get, set

    //[<Params("Host")>]
    [<Params("Host", "Cuda")>]
    member val Dev = "" with get, set

    //[<Params("100x100", "1000x1000")>]
    //[<Params("1000x1000")>]
    //[<Params("1000x1000", "2000x2000")>]
    [<Params("1000x1000", "2000x2000", "8000x8000")>]
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

        let workerType = typedefof<Worker<_>>.MakeGenericType typ
        worker <- Activator.CreateInstance (workerType, dev, shape) :?> IWorker
        sync()

    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Nothing () = worker.Nothing () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Zeros () = worker.Zeros () ; sync ()
    [<Benchmark>] member __.Ones () = worker.Ones () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Arange () = worker.Arange () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Copy () = worker.Copy () ; sync ()
    [<Benchmark>] member __.Negate () = worker.Negate () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Add () = worker.Add () ; sync ()
    [<Benchmark>] member __.Subtract () = worker.Subtract () ; sync ()
    [<Benchmark>] member __.Multiply () = worker.Multiply () ; sync ()
    [<Benchmark>] member __.Divide () = worker.Divide () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Power () = worker.Power () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Dot () = worker.Dot () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Sgn () = worker.Sgn () ; sync ()
    [<Benchmark>] member __.Log () = worker.Log () ; sync ()
    [<Benchmark>] member __.Log10 () = worker.Log10 () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Exp () = worker.Exp () ; sync ()
    [<Benchmark>] member __.Sin () = worker.Sin () ; sync ()
    [<Benchmark>] member __.Cos () = worker.Cos () ; sync ()
    [<Benchmark>] member __.Tan () = worker.Tan () ; sync ()
    [<Benchmark>] member __.Asin () = worker.Asin () ; sync ()
    [<Benchmark>] member __.Acos () = worker.Acos () ; sync ()
    [<Benchmark>] member __.Atan () = worker.Atan () ; sync ()
    [<Benchmark>] member __.Sinh () = worker.Sinh () ; sync ()
    [<Benchmark>] member __.Cosh () = worker.Cosh () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Tanh () = worker.Tanh () ; sync ()
    [<Benchmark>] member __.Sqrt () = worker.Sqrt () ; sync ()
    [<Benchmark>] member __.Ceiling () = worker.Ceiling () ; sync ()
    [<Benchmark>] member __.Floor () = worker.Floor () ; sync ()
    [<Benchmark>] member __.Round () = worker.Round () ; sync ()
    [<Benchmark>] member __.Truncate () = worker.Truncate () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.MaxElemwise () = worker.MaxElemwise () ; sync ()
    [<Benchmark>] member __.MinElemwise () = worker.MinElemwise () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Equal () = worker.Equal () ; sync ()
    [<Benchmark>] member __.NotEqual () = worker.NotEqual () ; sync ()
    [<Benchmark>] member __.Less () = worker.Less () ; sync ()
    [<Benchmark>] member __.LessOrEqual () = worker.LessOrEqual () ; sync ()
    [<Benchmark>] member __.Greater () = worker.Greater () ; sync ()
    [<Benchmark>] member __.GreaterOrEqual () = worker.GreaterOrEqual () ; sync ()
    [<Benchmark>] member __.Not () = worker.Not () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.And () = worker.And () ; sync ()
    [<Benchmark>] member __.Or () = worker.Or () ; sync ()
    [<Benchmark>] member __.Xor () = worker.Xor () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.CountTrueAxis () = worker.CountTrueAxis () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.IfThenElse () = worker.IfThenElse () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.SumAxis () = worker.SumAxis () ; sync ()
    [<Benchmark>] member __.ProductAxis () = worker.ProductAxis () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.MaxAxis () = worker.MaxAxis () ; sync ()
    [<Benchmark>] member __.MinAxis () = worker.MinAxis () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.AllAxis () = worker.AllAxis () ; sync ()
    [<Benchmark>] member __.AnyAxis () = worker.AnyAxis () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.ArgMaxAxis () = worker.ArgMaxAxis () ; sync ()
    [<Benchmark>] member __.ArgMinAxis () = worker.ArgMinAxis () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.FindAxis () = worker.FindAxis () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.MaskedGet () = worker.MaskedGet () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.MaskedSet () = worker.MaskedSet () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.TrueIndices () = worker.TrueIndices () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.Invert () = worker.Invert () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.PseudoInvert () = worker.PseudoInvert () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.SVD () = worker.SVD () ; sync ()
    [<Benchmark>] [<BenchmarkCategory("Overview")>] member __.SymEigDec () = worker.SymEigDec () ; sync ()

[<EntryPoint>]
let main argv = 
    let switcher = BenchmarkSwitcher (Assembly.GetExecutingAssembly())
    switcher.Run (args=argv) |> ignore
    0

