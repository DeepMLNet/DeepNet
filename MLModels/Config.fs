namespace Models

open System
open System.IO
open Microsoft.FSharp.Compiler.Interactive.Shell

open Basics


/// Configuration loader.
module Config = 

    type private FsiEvaluator () =

        // Build command line arguments & start FSI session
        let inStream = new StringReader("")
        let argv = [| "C:\\fsi.exe" |]
        let allArgs = Array.append argv [|"--noninteractive"
                                          "--quiet"
                                          "--lib:" + Util.assemblyDirectory
                                          "--reference:MLModels.dll"
                                          "--reference:SymTensor.dll"
                                          "--define:CONFIG"
                                          |]        
        let fsiConfig = FsiEvaluationSession.GetDefaultConfiguration()
        let fsiSession = FsiEvaluationSession.Create(fsiConfig, allArgs, inStream, Console.Out, Console.Error, true)  

        /// Evaluate F# expression and return the result, strongly typed.
        member this.EvalExpression (text) : 'T = 
            match fsiSession.EvalExpression(text) with
            | Some value -> 
                if value.ReflectionType <> typeof<'T> then 
                    failwithf "evaluated expression is of type %A but type %A was expected"
                        value.ReflectionType typeof<'T>        
                value.ReflectionValue |> unbox
            | None -> failwith "evaluated expression provided no result"

        // Evaluate F# script.
        member this.EvalScript path = 
            fsiSession.EvalInteraction (File.ReadAllText path)
            
    /// Directory of the currently loading configuration file.
    let mutable cfgDir = ""

    /// First ancessor directory of configuration file directory that contains a "Cfg" subdirectory.
    let mutable baseDir = ""

    /// Load configuration F# script (.fsx file) and returns the variable named "cfg".
    let load path = 
        if not (File.Exists path) then
            failwithf "configuration file %s does not exist" (Path.GetFullPath path)
        printfn "Using configuration file %s" (Path.GetFullPath path)

        // set paths for configuration file consumption
        cfgDir <- path |> Path.GetFullPath |> Path.GetDirectoryName
        let rec findBaseDir dir =
            if Directory.Exists (Path.Combine (dir, "Cfg")) then Path.GetFullPath dir
            else findBaseDir (Path.Combine (dir, ".."))
        baseDir <- findBaseDir cfgDir

        // evaluate configuration script
        try 
            let eval = FsiEvaluator()
            eval.EvalScript path
            eval.EvalExpression "cfg"
        with
        | _ -> 
            fprintfn Console.Error "Loading configuration file %s failed." (Path.GetFullPath path)
            exit 1

    /// Loads configuration script and changes current directory to output directory for that configuration.
    let loadAndChdir path = 
        if not (File.Exists path) then
            failwithf "configuration file %s does not exist" (Path.GetFullPath path)
        Directory.SetCurrentDirectory (Path.GetDirectoryName path)
        load (Path.GetFileName path)

    

