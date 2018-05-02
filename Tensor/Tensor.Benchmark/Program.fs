// To run the benchmark execute the following command:
// dotnet run -c Release -- TensorBenchmark

namespace Tensor.Benchmark

open System
open System.Reflection
open System.IO
open System.Collections.Generic
open System.Runtime.InteropServices
open System.Diagnostics

open ManagedCuda
open ManagedCuda.BasicTypes
open Newtonsoft.Json

open BenchmarkDotNet.Jobs
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Attributes.Jobs
open BenchmarkDotNet.Running
open BenchmarkDotNet.Order
open BenchmarkDotNet.Columns
open BenchmarkDotNet.Engines
open BenchmarkDotNet.Horology
open BenchmarkDotNet.Reports
open BenchmarkDotNet.Exporters
open BenchmarkDotNet.Exporters.Csv

open Tensor


module Main =

    let getProcessOutput name args =
        use p = new Process()
        p.StartInfo.FileName <- name
        p.StartInfo.Arguments <- args
        p.StartInfo.RedirectStandardOutput <- true
        p.Start() |> ignore
        let out = p.StandardOutput.ReadToEnd()
        p.WaitForExit()
        out       

    let getCpu () =
        if RuntimeInformation.IsOSPlatform (OSPlatform.Linux) then
            let cpuLine = "model name      : "
            File.ReadAllLines "/proc/cpuinfo"
            |> Seq.pick (fun l -> 
                if l.StartsWith cpuLine then Some (l.[cpuLine.Length..].Trim())
                else None)
        elif RuntimeInformation.IsOSPlatform (OSPlatform.Windows) then
            (getProcessOutput "wmic" "cpu get name").Split("\n").[1].Trim()
        else
            raise (PlatformNotSupportedException ())

    let getGpu () =
        use context = new CudaContext(createNew=false)
        "nVidia " + context.GetDeviceInfo().DeviceName

    let getDotNetVersion () =
        (getProcessOutput "dotnet" "--version").Trim()

    let writeInfo () =
        let resultsPath = Path.Combine("BenchmarkDotNet.Artifacts", "results") |> Path.GetFullPath
        Directory.CreateDirectory resultsPath |> ignore
            
        let infoPath = Path.Combine(resultsPath, "Info.json") |> Path.GetFullPath        
        let tv = typedefof<Tensor<_>>.Assembly.GetName().Version
        let info = {
            Name = Environment.MachineName
            CPU = getCpu ()
            GPU = getGpu ()
            OS = RuntimeInformation.OSDescription.Trim()
            Library = sprintf "F# Tensor %d.%d.%d" tv.Major tv.Minor tv.Build
            Runtime = sprintf ".NET SDK %s" (getDotNetVersion())
            Date = DateTime.UtcNow
        }
        File.WriteAllText (infoPath, JsonConvert.SerializeObject(info, Formatting.Indented))

    [<EntryPoint>]
    let main argv = 
        Tensor.Cuda.Cfg.FastKernelMath <- true
        //Tensor.Cuda.Cfg.RestrictKernels <- true

        writeInfo ()
        let switcher = BenchmarkSwitcher (Assembly.GetExecutingAssembly())
        switcher.Run (args=argv) |> ignore
        0

