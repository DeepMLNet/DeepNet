namespace Tensor.Benchmark

open System
open System.Reflection
open System.IO
open System.Collections.Generic

open BenchmarkDotNet.Jobs
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Attributes.Jobs
open BenchmarkDotNet.Running
open BenchmarkDotNet.Order
open BenchmarkDotNet.Columns
open BenchmarkDotNet.Engines
open BenchmarkDotNet.Exporters
open BenchmarkDotNet.Exporters.Csv
open BenchmarkDotNet.Horology
open BenchmarkDotNet.Reports



type MyOrder () =  
    interface IOrderProvider with
        member __.GetExecutionOrder benchmarks = Seq.ofArray benchmarks
        member __.GetSummaryOrder (benchmarks, summary) = 
            let names = benchmarks |> Seq.groupBy (fun b -> b.Target.Method.Name) |> Seq.map fst |> Array.ofSeq
            benchmarks |> Seq.sortBy (fun b -> names |> Array.findIndex ((=) b.Target.Method.Name))
        member __.GetHighlightGroupKey _ = null
        member __.GetLogicalGroupKey (config, allBenchmarks, benchmark) = benchmark.Target.Method.Name
        member __.GetLogicalGroupOrder logicalGroups = logicalGroups 
        member __.SeparateLogicalGroups = true

type MyConfig () as this =
    inherit ManualConfig ()
    do this.Add (CsvExporter (CsvSeparator.CurrentCulture, SummaryStyle(PrintUnitsInHeader=true, PrintUnitsInContent=false, TimeUnit=TimeUnit.Millisecond)))
    do this.Add (Job.Default.WithWarmupCount(0).WithLaunchCount(1).WithIterationTime(TimeInterval.FromMilliseconds 250.0).WithTargetCount(4).WithUnrollFactor(1).WithMinInvokeCount(1))
    do this.Set (MyOrder())
    do this.Set (SummaryStyle (PrintUnitsInHeader=true, PrintUnitsInContent=false, TimeUnit=TimeUnit.Millisecond))
