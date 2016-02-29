namespace SymTensor

[<AutoOpen>]
module ModelContextTypes =

    /// Module context.
    [<StructuredFormatDisplay("{PrettyString}")>]
    type MC (context: string) =

        let mutable subMCs = Map.empty
        let mutable parameters : Set<IVarSpec> = Set.empty

        let toSizeSpec (name: string) =
            let fullName = context + "." + name
            if name.StartsWith ">" then SizeSpec.flexSymbol fullName
            else SizeSpec.symbol fullName

        let toShapeSpec (shapeObj: obj list) =
            shapeObj |> List.map
                (function
                | :? string as symName -> toSizeSpec symName
                | :? int as f when f = -1 -> SizeSpec.broadcastable
                | :? int as f when f >= 0 -> SizeSpec.fix f
                | r -> failwithf "size must be either a size symbol name (string), \
                                  a fixed size (positive integer) or -1 for broadcast, but got %A" r)

        /// Creates and returns a model variables.
        member this.Var (name: string) (shape: obj list) =
            Expr.var (context + "." + name) (toShapeSpec shape)

        /// Creates and returns a model parameter.
        member this.Param (name: string) (shape: obj list) =
            let p = Expr.var (context + "." + name) (toShapeSpec shape)
            parameters <- parameters |> Set.add (Expr.extractVar p :> IVarSpec)
            p

        /// Creates and returns a symbolic size.
        /// If the name starts with ">" a dynamic size is created.
        member this.Size (name: string) =
            toSizeSpec name

        /// creates and returns a submodule
        member this.Module (name: string) =
            let subContext = context + "." + name
            match Map.tryFind subContext subMCs with
            | Some subMC -> subMC
            | None ->
                let subMC = MC (subContext)
                subMCs <- subMCs |> Map.add subContext subMC
                subMC
            
        /// context name
        member this.Context = context

        /// submodules
        member this.SubModules = subMCs

        /// parameters of this module and all submodules
        member this.Parameters = 
            seq {for KeyValue(_, smc) in subMCs -> smc.Parameters}
            |> Set.unionMany 
            |> Set.union parameters

        member this.PrettyString = 
            sprintf "%A {%s}"
                context (this.Parameters |> Set.toSeq |> Seq.map (sprintf "%A") |> String.concat ", ")
