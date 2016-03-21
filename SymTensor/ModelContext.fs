namespace SymTensor

open ArrayNDNS

[<AutoOpen>]
module ModelContextTypes =

    /// allocates an ArrayNDT given the shape
    type ArrayAllocatorT<'T> = NShapeSpecT -> ArrayNDT<'T>

    /// Model evaluation device specification.
    type IDevice =
        abstract member Allocator: NShapeSpecT -> ArrayNDT<'T> 
        abstract member Compiler: IUExprCompiler
        abstract member DefaultLoc: ArrayLocT

    /// Evaluates the model on the host.
    let DevHost = { 
        new IDevice with
            member this.Allocator shp = ArrayNDHost.newContiguous shp :> ArrayNDT<_>
            member this.Compiler = { new IUExprCompiler with 
                                        member this.Name = "Host"
                                        member this.Compile env exprs = onHost env exprs }
            member this.DefaultLoc = LocHost
    }

    /// A set of symbolic variables forming a set of parameters for a model.
    [<StructuredFormatDisplay("{Pretty}")>]
    type ParameterSetT<'T when 'T: equality> 
            (name:           string, 
             parameters:     VarSpecT<'T> seq) =

        let flatVarName = "PS_" + name
        let pars = parameters |> Seq.toList |> List.sort

        let shapes = pars |> List.map VarSpec.shape

        /// layout of data vector
        let startIdxs, totalElems =
            shapes
            |> List.mapFold (fun pos shp -> pos, pos + ShapeSpec.nElem shp)
                SizeSpec.zero

        /// variable containing all parameters
        let flatVar : ExprT<'T> = Expr.var (flatVarName) [totalElems]

        /// parameter variables
        let parameterVars =
            (startIdxs, shapes)
            ||> List.map2 (fun startIdx shp ->
                let elems = ShapeSpec.nElem shp
                let v = flatVar.[startIdx .. startIdx + elems - 1]
                Expr.reshape shp v)
            |> List.zip pars
            |> Map.ofList

        member this.Parameters = pars
        member this.Shapes = shapes
        member this.StartIdxs = startIdxs
        member this.TotalElems = totalElems

        /// name of flat parameter vector
        member this.FlatName = flatVarName

        /// flat parameter vector containing all parameters
        member this.Flat = flatVar

        /// derivative of "expr" w.r.t. flat parameter vector
        member this.WrtFlat expr = 
            let expr = this.Subst expr
            let derivs = Deriv.compute expr
            match derivs |> Map.tryFind (Expr.extractVar this.Flat) with
            | Some deriv -> deriv
            | None -> 
                printfn "Warning: ParameterSet %s does not occur in expression for differentiation." name
                Expr.zeroMatrix 
                    (Expr.shapeOf expr |> ShapeSpec.nElem) 
                    (Expr.shapeOf this.Flat |> ShapeSpec.nElem)
                

        /// variable for a given parameter
        member this.Item
            with get (par: VarSpecT<'T>) = parameterVars.[par]

        /// substitutes this ParameterSet into the given expression
        member this.Subst expr =
            (expr, pars)
            ||> List.fold (fun expr par -> Expr.subst (Expr.makeVar par) this.[par] expr)

        /// pretty string
        member this.Pretty =
            let parsStr =
                pars
                |> List.map (sprintf "%A")
                |> String.concat ", "
            sprintf "%s = {%s}" flatVarName parsStr


    /// Actual values for variables in a ParameterSet.
    type ParameterStorageT<'T when 'T: equality> 
            (parameterSet:    ParameterSetT<'T>,
             symSizes:        SymSizeEnvT,
             allocator:       ArrayAllocatorT<'T>) =

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

        /// Uses the values of this ParameterStorageT for the the corresponding
        /// ParameterSetT in the given variable environment.
        member this.Use varEnv =
            varEnv |> VarEnv.add parameterSet.Flat this.Flat


    /// A model builder.
    [<StructuredFormatDisplay("{PrettyString}")>]
    type ModelBuilder<'T when 'T: equality> (context: string) =

        let mutable subMBs = Map.empty
        let mutable parameters : Set<VarSpecT<'T>> = Set.empty
        let mutable vars : Set<IVarSpec> = Set.empty
        let mutable symSizes = []

        let toSizeSpec (name: string) =            
            SizeSpec.symbol name

        let toShapeSpec (shapeObj: obj list) =
            shapeObj |> List.map
                (function
                | :? string as symName -> toSizeSpec symName
                | :? int as f when f = -1 -> SizeSpec.broadcastable
                | :? int as f when f >= 0 -> SizeSpec.fix f
                | r -> failwithf "size must be either a size symbol name (string), \
                                  a fixed size (positive integer) or -1 for broadcast, but got %A" r)

        /// creates and returns a submodule
        member this.Module (name: string) =
            let subContext = context + "." + name
            match Map.tryFind subContext subMBs with
            | Some subMC -> subMC
            | None ->
                let subMC = ModelBuilder (subContext)
                subMBs <- subMBs |> Map.add subContext subMC
                subMC

        /// Creates and returns a model variable.
        member this.Var (name: string) (shape: ShapeSpecT) =
            let v = Expr.var (context + "." + name) shape
            vars <- vars |> Set.add (Expr.extractVar v :> IVarSpec)
            v

        /// Creates and returns a model parameter.
        member this.Param (name: string) (shape: ShapeSpecT) =
            let p = Expr.var (context + "." + name) shape
            parameters <- parameters |> Set.add (Expr.extractVar p)
            p

        /// Creates and returns a symbolic size.
        /// If the name starts with ">" a dynamic size is created.
        member this.Size (name: string) =
            let s = toSizeSpec name
            symSizes <- s :: symSizes
            s

        /// Returns a fixed size. If size is -1 then a broadcastable size one is created.
        member this.Fix size =
            if size = -1 then SizeSpec.broadcastable
            else SizeSpec.fix size
           
        /// context name
        member this.Context = context

        /// submodules
        member this.SubModels = subMBs

        /// parameters of this model and all submodels
        member this.Parameters = 
            seq {for KeyValue(_, smc) in subMBs -> smc.Parameters}
            |> Set.unionMany 
            |> Set.union parameters

        /// variables of this model and all submodels
        member this.Vars =
            seq {for KeyValue(_, smc) in subMBs -> smc.Vars}
            |> Set.unionMany 
            |> Set.union vars
            
        /// size symbols of this model and all submodels
        member this.SymSizes =
            subMBs
            |> Map.toSeq
            |> Seq.map (fun (_, m) -> m.SymSizes)
            |> List.concat
            |> List.append symSizes

        /// Completes the definition of model parameters and size symbols
        /// and returns a DefinedModel.
        member this.ParametersComplete () =
            DefinedModel (context, this.Parameters, this.Vars, this.SymSizes)
            
        member this.PrettyString = 
            sprintf "%A {%s}"
                context (this.Parameters |> Set.toSeq |> Seq.map (sprintf "%A") |> String.concat ", ")


    /// A model with all parameters and symbolic sizes defined.
    and DefinedModel<'T when 'T: equality> (context:     string,
                                            parameters:  Set<VarSpecT<'T>>,
                                            vars:        Set<IVarSpec>,
                                            symSizes:    SizeSpecT list) =

        let parameterSet = ParameterSetT (context, parameters)
        let mutable symSizeEnv = SymSizeEnv.empty
        let mutable varLocs : VarLocsT = Map.empty

        /// sets a symbolic size to a value
        member this.SetSize size value =
            match size with
            | Base (Sym sym) -> symSizeEnv <- SymSizeEnv.add sym (SizeSpec.fix value) symSizeEnv
            | _ -> failwith "need a size symbol to set size"

        /// sets the location of the given variable
        member this.SetLoc var loc =
            varLocs <- varLocs |> Map.add (Expr.extractVar var |> UVarSpec.ofVarSpec) loc

        /// Infers localtion symbolic size by matching a variables symbolic shape to the shape
        /// of the given variable value.
        member this.UseTmplVal var value =
            VarEnv.empty 
            |> VarEnv.add var value
            |> VarEnv.inferSymSizes
            |> SymSizeEnv.merge symSizeEnv
            |> fun ss -> symSizeEnv <- ss

            varLocs <- varLocs |> Map.add (Expr.extractVar var |> UVarSpec.ofVarSpec) (ArrayND.location value)

        /// ParameterSet of this module's (and all submodules') parameteres
        member this.ParameterSet = parameterSet

        /// Inferred size symbol values
        member this.SymSizeEnv = symSizeEnv

        /// Variable locations
        member this.VarLocs = varLocs
        
        /// Substitutes the ParameterSet into the given expression
        member this.Subst expr =
            this.ParameterSet.Subst expr

        /// Derivative of "expr" w.r.t. flat vector containing all model parameters
        member this.WrtParameters expr =
            this.ParameterSet.WrtFlat expr

        /// instantiates the model with numeric sizes for all size symbols
        member this.Instantiate (device: IDevice) =
            // check if instantiable
            for symSize in symSizes do
                if not (SymSizeEnv.subst symSizeEnv symSize |> SizeSpec.canEval) then
                    failwithf "Cannot instantiate model because size symbol %A has no value."
                        symSize

            // apply default variable location
            let mutable varLocs = varLocs
            for var in vars do
                if not (varLocs |> Map.containsKey (UVarSpec.ofVarSpec var)) then
                    varLocs <- varLocs |> Map.add (UVarSpec.ofVarSpec var) device.DefaultLoc

            // create compile environement
            let compileEnv =
                {SymSizes=symSizeEnv; VarLocs=varLocs; ResultLoc=device.DefaultLoc; CanDelay=false}

            ModelInstance (parameterSet, device, compileEnv)


    /// A model with numeric sizes for all size symbols and allocated parameter storage.
    and ModelInstance<'T when 'T: equality> (parameterSet:    ParameterSetT<'T>,
                                             device:          IDevice,                                             
                                             compileEnv:      CompileEnvT) =

        let parameterStorage = ParameterStorageT (parameterSet, compileEnv.SymSizes, device.Allocator)        

        let compileSpec resultLoc = 
            // add ParameterStorage to variable locations
            let psVar = Expr.extractVar parameterSet.Flat |> UVarSpec.ofVarSpec
            let psVal = parameterStorage.Flat
            let varLocs =
                compileEnv.VarLocs
                |> Map.add psVar (ArrayND.location psVal)

            device.Compiler, {compileEnv with ResultLoc = resultLoc;
                                              VarLocs   = varLocs}

        let subst = parameterSet.Subst
        let usePars = parameterStorage.Use

        /// Parameter values.
        member this.ParameterStorage = parameterStorage

        /// Compile environment.
        member this.CompileEnv = compileEnv

        /// Creates a function from the given expression using the model's ParameterSet and ParameterStorage
        /// using the specified result location.
        member this.Func (resultLoc: ArrayLocT, expr0: ExprT<'T>) =
            Func.make (compileSpec resultLoc) (subst expr0) << usePars

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (resultLoc: ArrayLocT, expr0: ExprT<'T>, expr1: ExprT<'T>) =
            Func.make2 (compileSpec resultLoc) (subst expr0) (subst expr1) << usePars

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (resultLoc: ArrayLocT, expr0: ExprT<'T>, expr1: ExprT<'T>, expr2: ExprT<'T>) =
            Func.make3 (compileSpec resultLoc) (subst expr0) (subst expr1) (subst expr2) << usePars

        /// Creates a function from the given expression using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (expr0: ExprT<'T>) =
            this.Func (device.DefaultLoc, expr0)

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (expr0: ExprT<'T>, expr1: ExprT<'T>) =
            this.Func (device.DefaultLoc, expr0, expr1)

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (expr0: ExprT<'T>, expr1: ExprT<'T>, expr2: ExprT<'T>) =
            this.Func (device.DefaultLoc, expr0, expr1, expr2)
        
