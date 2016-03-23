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


let post device x =
    if device = DevCuda then ArrayNDCuda.toDev x :> ArrayNDT<'T>
    else x :> ArrayNDT<'T>
    
let compareTraces func dump =
    printfn "Evaluating on CUDA device..."
    Trace.startSession "CUDA"
    func DevCuda
    let cudaTrace = Trace.endSession ()
    if dump then
        use tw = File.CreateText("CUDA.txt")
        Trace.dump tw cudaTrace

    printfn "Evaluating on host..."
    Trace.startSession "Host"
    func DevHost
    let hostTrace = Trace.endSession ()
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
