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
    