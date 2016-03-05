namespace SymTensor

open ArrayNDNS

[<AutoOpen>]
module ModelContextTypes =
 
    /// A set of symbolic variables forming a set of parameters for a model.
    type ParameterSetT<'T> (name:           string, 
                            parameters:     VarSpecT<'T> seq) =

        let pars = parameters |> Seq.toList |> List.sort

        let shapes = pars |> List.map VarSpec.shape

        /// layout of data vector
        let startIdxs, totalElems =
            shapes
            |> List.mapFold (fun pos shp -> pos, pos + ShapeSpec.nElem shp)
                SizeSpec.zero

        /// variable containing all parameters
        let dataVar : ExprT<'T> = Expr.var ("PS_" + name) [totalElems]

        /// parameter variables
        let parameterVars =
            (startIdxs, shapes)
            ||> List.map2 (fun startIdx shp ->
                let elems = ShapeSpec.nElem shp
                let v = dataVar.[startIdx .. startIdx + elems - 1]
                Expr.reshape shp v)
            |> List.zip pars
            |> Map.ofList

        member this.Parameters = pars
        member this.Shapes = shapes
        member this.StartIdxs = startIdxs
        member this.TotalElems = totalElems

        /// flat variable containing all parameters
        member this.Flat = dataVar

        /// variable for a given parameter
        member this.Item
            with get (par: VarSpecT<'T>) = parameterVars.[par]


    /// Actual values for variables in a ParameterSet.
    type ParameterStorageT<'T> (parameterSet:    ParameterSetT<'T>,
                                symSizes:        SymSizeEnvT,
                                allocator:       NShapeSpecT -> ArrayNDT<'T>) =

        let evalSize = SymSizeEnv.subst symSizes >> SizeSpec.eval
        let evalShape = List.map evalSize

        let nShapes = parameterSet.Shapes |> List.map evalShape
        let nStartIdxs = parameterSet.StartIdxs |> List.map evalSize       
        let nTotalElems = parameterSet.TotalElems |> evalSize

        let dataVal = allocator [nTotalElems]

        let parameterVals = 
            (nStartIdxs, nShapes)
            ||> List.map2 (fun startIdx shp ->
                let elems = List.fold (*) 1 shp
                let v = dataVal.[startIdx .. startIdx + elems - 1]
                ArrayND.reshapeView shp v)
            |> List.zip parameterSet.Parameters
            |> Map.ofList

        /// flat array containing all parameters
        member this.Flat 
            with get () = dataVal
            and set value = dataVal.[Fill] <- value

        /// value for a given parameter
        member this.Item
            with get (par: VarSpecT<'T>) = parameterVals.[par]
            and set (par: VarSpecT<'T>) value = parameterVals.[par].[Fill] <- value


    /// Module context.
    [<StructuredFormatDisplay("{PrettyString}")>]
    type MC<'T> (context: string) =

        let mutable subMCs = Map.empty
        let mutable parameters : Set<VarSpecT<'T>> = Set.empty

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
            parameters <- parameters |> Set.add (Expr.extractVar p)
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

        /// ParameterSet of this module's (and all submodule's) variables
        member this.ParameterSet = 
            ParameterSetT (context, this.Parameters)

        member this.PrettyString = 
            sprintf "%A {%s}"
                context (this.Parameters |> Set.toSeq |> Seq.map (sprintf "%A") |> String.concat ", ")

        