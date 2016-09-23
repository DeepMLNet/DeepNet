namespace SymTensor

open Basics
open ArrayNDNS



module Dump =

    let mutable private target = None
    let mutable prefix = ""
    let mutable private dumpedFullnames : Set<string> = Set []

    let getTarget ()=
        target

    let isActive () =
        target.IsSome

    let start dumpPath =
        if isActive () then failwith "dump session already active"
        target <- Some (HDF5.OpenWrite dumpPath)
        dumpedFullnames <- Set.empty

    let stop () =
        if not (isActive ()) then failwith "no dump session active"
        target.Value.Dispose ()
        target <- None

    let dumpValue name value =
        match target with
        | Some t ->
            let fullname =
                if prefix.EndsWith "/" then prefix + name
                else prefix + "/" + name
            if not (dumpedFullnames |> Set.contains fullname) then
                ArrayNDHDF.writeUntyped t fullname value
                dumpedFullnames <- dumpedFullnames |> Set.add fullname
        | None -> ()
        
