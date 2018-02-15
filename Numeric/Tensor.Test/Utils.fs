namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor.Utils
open Tensor


module Util = 

    /// directory of our assembly
    let assemblyDirectory = 
        // http://stackoverflow.com/questions/52797/how-do-i-get-the-path-of-the-assembly-the-code-is-in
        let codeBase = System.Reflection.Assembly.GetExecutingAssembly().CodeBase
        let uri = new System.UriBuilder(codeBase)
        let path = System.Uri.UnescapeDataString(uri.Path)
        System.IO.Path.GetDirectoryName(path)


module Seq =

    /// sequence counting from given value to infinity
    let countingFrom from = seq {
        let mutable i = from
        while true do
            yield i
            i <- i + 1
    }

    /// sequence counting from zero to infinity
    let counting = countingFrom 0
    

/// Extensions to System.Random.
[<AutoOpen>]
module internal RandomExtensions = 

    type System.Random with

        /// Generates an infinite sequence of non-negative random integers.
        member this.Seq () =
            Seq.initInfinite (fun _ -> this.Next())    

        /// Generates an infinite sequence of non-negative random integers that is less than the specified maximum.
        member this.Seq (maxValue) =
            Seq.initInfinite (fun _ -> this.Next(maxValue))    

        /// Generates an infinite sequence of random integers within the given range.
        member this.Seq (minValue, maxValue) =
            Seq.initInfinite (fun _ -> this.Next(minValue, maxValue))
            
                