namespace Models

open System.IO
open MBrace.FsPickler
open MBrace.FsPickler.Json

/// JSON serialization.
module Json =
    
    let private serializer = 
        FsPickler.CreateJsonSerializer(indent=true, omitHeader=true)

    /// Serializes an object into a JSON string.
    let serialize value =
        serializer.PickleToString value

    /// Deserializes an object from a JSON string.
    let deserialize str =
        serializer.UnPickleOfString str

    /// Loads JSON data from the specified file.
    let load path =
        use tr = File.OpenText path
        serializer.Deserialize (tr)

    /// Saves an object as JSON to the specified file.
    let save path value =
        use tw = File.CreateText path
        serializer.Serialize (tw, value)
    
         
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
