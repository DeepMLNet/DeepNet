namespace Basics

open System.IO
open Nessos.FsPickler
open Nessos.FsPickler.Json

/// JSON serialization.
module Json =
    
    let private serialzier = 
        FsPickler.CreateJsonSerializer(indent=true, omitHeader=true)

    /// Loads JSON data from the specified file.
    let load path =
        use tr = File.OpenText path
        serialzier.Deserialize (tr)

    /// Saves an object as JSON to the specified file.
    let save path value =
        use tw = File.CreateText path
        serialzier.Serialize (tw, value)


/// Binary serialization.
module Pickle =
    
    let private serialzier = 
        FsPickler.CreateBinarySerializer()

    /// Loads binary data from the specified file.
    let load path =
        use tr = File.OpenRead path
        serialzier.Deserialize (tr)

    /// Saves an object to the specified file.
    let save path value =
        use tw = File.Create path
        serialzier.Serialize (tw, value)
