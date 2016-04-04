namespace Models

open System
open System.IO
open System.Text
open Microsoft.FSharp.Compiler.Interactive.Shell

open Basics



module Config = 

    type FsiEvaluator () =

        // Build command line arguments & start FSI session
        let inStream = new StringReader("")
        let argv = [| "C:\\fsi.exe" |]
        let allArgs = Array.append argv [|"--noninteractive"
                                          "--quiet"
                                          "--lib:" + Util.assemblyDirectory
                                          "--reference:Models.dll"
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
            

    let load directory = 
        let filename = Path.Combine (directory, "cfg.fsx")
        if not (File.Exists filename) then
            failwithf "configuration file %s does not exist" (Path.GetFullPath filename)
        
        Directory.SetCurrentDirectory directory

        try 
            let eval = FsiEvaluator()
            eval.EvalScript filename
            eval.EvalExpression "cfg"
        with
        | _ -> 
            fprintfn Console.Error "Loading configuration file %s failed." (Path.GetFullPath filename)
            exit 1
    

