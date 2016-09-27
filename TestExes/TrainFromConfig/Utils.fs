namespace TrainFromConfig

open FSharp.Configuration
open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System.Text.RegularExpressions
open GPTransfer

[<AutoOpen>]
module Utils =

    ///active patterns
    let (|Cuda|Host|) input = if input = "DevCuda" then Cuda else Host

    let inStringToStringOption string =
        let m = Regex.Match ("Some (\S+)", string)
        if m.Success then
            Some m.Groups.[0].Value
        else None
    
    let inStringToIntOption string =
        let m = Regex.Match ("Some (\d+)", string)
        if m.Success then
            let mutable intvalue = 0
            if System.Int32.TryParse(m.Groups.[0].Value, &intvalue) then Some(intvalue)
            else None
        else None
    
    let inStringToFloatOption string =
        let m = Regex.Match ("Some (\d+)", string)
        if m.Success then
            let mutable floatvalue = 0.0
            if System.Double.TryParse(m.Groups.[0].Value, &floatvalue) then Some(floatvalue)
            else None
        else None 

