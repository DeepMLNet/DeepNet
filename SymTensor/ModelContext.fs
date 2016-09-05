namespace SymTensor

open System

open Basics
open ArrayNDNS

[<AutoOpen>]
module ModelContextTypes =

    /// function that returns an initialization value for a parameter
    type Initializer<'T> = int -> int list -> ArrayNDHostT<'T>

    /// model parameter
    type ParameterInfo<'T> = {
        ExprRef:        ExprT<'T> ref
        Initializer:    Initializer<'T>
    }

    /// Model evaluation device specification.
    type IDevice =
        abstract member Allocator:      NShapeSpecT -> ArrayNDT<'T> 
        abstract member ToDev:          ArrayNDHostT<'T> -> ArrayNDT<'T>
        abstract member ToHost:         ArrayNDT<'T> -> ArrayNDHostT<'T>
        abstract member Compiler:       IUExprCompiler
        abstract member DefaultLoc:     ArrayLocT
        abstract member DefaultFactory: IUExprCompiler * CompileEnvT

    /// Evaluates the model on the host.
    let DevHost = { 
        new IDevice with
            member this.Allocator shp = ArrayNDHost.newC shp :> ArrayNDT<_>
            member this.ToDev ary =     ary :> ArrayNDT<_>
            member this.ToHost ary =    ary :?> ArrayNDHostT<_>
            member this.Compiler =      { new IUExprCompiler with 
                                            member this.Name = "Host"
                                            member this.Compile env exprs = onHost env exprs }
            member this.DefaultLoc =    LocHost
            member this.DefaultFactory = this.Compiler, {CompileEnv.empty with ResultLoc=this.DefaultLoc}
    }

    /// A set of symbolic variables forming a set of parameters for a model.
    [<StructuredFormatDisplay("{Pretty}")>]
    type ParameterSetT<'T when 'T: equality and 'T: comparison> 
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

        /// mapping from parameter expression to parameter variable
        let parameterOfExpr =
            parameterVars
            |> Map.toSeq
            |> Seq.map (fun (var, expr) -> expr, var)
            |> Map.ofSeq

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

        /// returns the parameter for the given variable (in expression form) 
        member this.ParameterOf expr = 
            match parameterOfExpr |> Map.tryFind expr with
            | Some var -> var
            | None -> failwith "this ParameterSet does not contain a variable for the specified expression"

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
    type ParameterStorageT<'T when 'T: equality and 'T: comparison> 
            (parameterSet:    ParameterSetT<'T>,
             symSizes:        SymSizeEnvT,
             device:          IDevice) =

        let evalSize = SymSizeEnv.subst symSizes >> SizeSpec.eval
        let evalShape = List.map evalSize

        let nShapes = parameterSet.Shapes |> List.map evalShape
        let nStartIdxs = parameterSet.StartIdxs |> List.map evalSize       
        let nTotalElems = parameterSet.TotalElems |> evalSize

        let dataVal = device.Allocator [nTotalElems]

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

        /// value for a given parameter
        member this.Item
            with get (par: ExprT<'T>) = this.[parameterSet.ParameterOf par]
            and set (par: ExprT<'T>) value = this.[parameterSet.ParameterOf par] <- value

        /// Uses the values of this ParameterStorageT for the the corresponding
        /// ParameterSetT in the given variable environment.
        member this.Use varEnv =
            varEnv |> VarEnv.add parameterSet.Flat this.Flat

        /// Loads the parameter values from a previously saved HDF5 file.
        member this.Load filename = 
            use hdf = HDF5.OpenRead filename
            for KeyValue(vs, vsView) in parameterVals do
                let value = ArrayNDHDF.read hdf vs.Name
                vsView.[Fill] <- device.ToDev value

        /// Saves the parameter values to a HDF5 file.
        /// Each parameter is stored in a separate HDF5 record under its name.
        member this.Save filename =
            use hdf = HDF5.OpenWrite filename
            for KeyValue(vs, vsView) in parameterVals do
                let value = device.ToHost vsView
                ArrayNDHDF.write hdf vs.Name value

    /// A model builder.
    [<StructuredFormatDisplay("{PrettyString}")>]
    type ModelBuilder<'T when 'T: equality and 'T: comparison> 
                                            (context:       string,
                                             isSubModule:   bool) =

        let mutable subMBs = Map.empty
        let mutable parameters : Map<VarSpecT<'T>, ParameterInfo<'T>> = Map.empty
        let mutable vars : Set<IVarSpec> = Set.empty
        let mutable symSizes = []
        let mutable symSizeEnv = SymSizeEnv.empty
        let mutable varLocs : VarLocsT = Map.empty
        let mutable instantiated = false

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

        let defaultInitializer (seed: int) (shp: int list) =
            let rng = Random(seed)
            rng.SeqDouble (-0.01, 0.01) 
            |> Seq.map conv<'T>
            |> ArrayNDHost.ofSeqWithShape shp

        new (context: string) = ModelBuilder(context, false)

        /// creates and returns a submodule
        member this.Module (name: string) =
            if instantiated then failwith "cannot add module after model has been instantiated"

            let subContext = context + "." + name
            match Map.tryFind subContext subMBs with
            | Some subMC -> subMC
            | None ->
                let subMC = ModelBuilder (subContext, true)
                subMBs <- subMBs |> Map.add subContext subMC
                subMC

        /// Creates and returns a model variable.
        member this.Var<'V> (name: string) (shape: ShapeSpecT) : ExprT<'V> =
            let v = Expr.var (context + "." + name) shape
            vars <- vars |> Set.add (Expr.extractVar v :> IVarSpec)
            v

        /// Creates and returns a model parameter.
        member this.Param (name: string, shape: ShapeSpecT, ?initializer: Initializer<'T>) =
            let initializer = defaultArg initializer defaultInitializer
            if instantiated then failwith "cannot add parameter after model has been instantiated"

            let p = Expr.var (context + "." + name) shape
            let pRef = ref p
            parameters <- parameters |> Map.add (Expr.extractVar p) {ExprRef = pRef; Initializer = initializer}
            pRef

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
            |> Seq.fold Map.join parameters

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
            |> VarEnv.inferSymSizes symSizeEnv
            |> fun ss -> symSizeEnv <- ss

            varLocs <- varLocs |> Map.add (Expr.extractVar var |> UVarSpec.ofVarSpec) (ArrayND.location value)

        /// Inferred size symbol values
        member this.SymSizeEnv = symSizeEnv

        /// Variable locations
        member this.VarLocs = varLocs

        /// instantiates the model with numeric sizes for all size symbols and initializes the parameter values
        member this.Instantiate (device: IDevice, ?canDelay: bool) =
            let canDelay = defaultArg canDelay true
            if isSubModule then failwith "a submoule cannot be instantiated"
            if instantiated then failwith "this model has already been instantiated"

            // Check that SymSizes that are required for ParameterStorage instantiation can
            // be evaluated to a numeric value.
            let neededSymSizes = 
                this.Parameters
                |> Map.toSeq
                |> Seq.collect (fun (vs, _) -> VarSpec.shape vs)
                |> Set.ofSeq
            let missingSymSizes =
                neededSymSizes
                |> Set.filter (SymSizeEnv.subst symSizeEnv >> SizeSpec.canEval >> not)
            if not (Set.isEmpty missingSymSizes) then
                failwithf "Cannot instantiate model because size symbols %A have no value."
                    (missingSymSizes |> Set.toList)

            // apply default variable location
            let mutable varLocs = varLocs
            for var in vars do
                if not (varLocs |> Map.containsKey (UVarSpec.ofVarSpec var)) then
                    varLocs <- varLocs |> Map.add (UVarSpec.ofVarSpec var) device.DefaultLoc

            // create compile environement
            let compileEnv =
                {SymSizes=symSizeEnv; VarLocs=varLocs; ResultLoc=device.DefaultLoc; CanDelay=canDelay}

            // instantiate
            instantiated <- true
            let mi = ModelInstance (context, this.Parameters, device, compileEnv)
            mi.InitPars 0
            mi

            
        member this.PrettyString = 
            sprintf "%A {%s}"
                context (this.Parameters |> Map.toSeq |> Seq.map (fst >> sprintf "%A") |> String.concat ", ")


    /// A model with numeric sizes for all size symbols and allocated parameter storage.
    and ModelInstance<'T when 'T: equality and 'T: comparison> 
                                            (context:         string,
                                             parameters:      Map<VarSpecT<'T>, ParameterInfo<'T>>,
                                             device:          IDevice,                                             
                                             compileEnv:      CompileEnvT) =

        let mutable compileEnv = compileEnv

        // create parameter set and parameter storage
        let parameterSpecs = parameters |> Map.toSeq |> Seq.map fst
        let parameterSet = ParameterSetT (context, parameterSpecs)
        let parameterStorage = ParameterStorageT (parameterSet, compileEnv.SymSizes, device)        

        // set references to view into ParameterSet
        do for KeyValue(ps, pi) in parameters do
             pi.ExprRef := parameterSet.[ps]

        let compileSpec resultLoc = 
            // add ParameterStorage to variable locations
            let psVar = Expr.extractVar parameterSet.Flat |> UVarSpec.ofVarSpec
            let psVal = parameterStorage.Flat
            let varLocs =
                compileEnv.VarLocs
                |> Map.add psVar (ArrayND.location psVal)

            device.Compiler, {compileEnv with ResultLoc = resultLoc;
                                              VarLocs   = varLocs}

        let useParStorage = parameterStorage.Use

        /// ParameterSet of this module's (and all submodules') parameteres
        member this.ParameterSet = parameterSet

        /// symbolic flat parameter vector
        member this.ParameterVector = this.ParameterSet.Flat

        /// Derivative of "expr" w.r.t. flat vector containing all model parameters
        member this.WrtParameters expr =
            this.ParameterSet.WrtFlat expr

        /// Parameter values.
        member this.ParameterStorage = parameterStorage

        /// numeric flat parameter vector
        member this.ParameterValues = this.ParameterStorage.Flat

        /// Compile environment.
        member this.CompileEnv = compileEnv

        /// sets the location of the given variable
        member this.SetLoc var loc =
            let uvs = var |> Expr.extractVar |> UVarSpec.ofVarSpec

            match compileEnv.VarLocs |> Map.tryFind uvs with
            | Some prvLoc when prvLoc <> loc ->
                failwithf "cannot change location of variable %A from %A to %A after model instantiation"
                    uvs prvLoc loc
            | _ -> ()

            compileEnv <- {compileEnv with VarLocs = compileEnv.VarLocs |> Map.add uvs loc}

        /// Load parameter values.
        member this.LoadPars filename = this.ParameterStorage.Load filename

        /// Save parameter values.
        member this.SavePars filename = this.ParameterStorage.Save filename

        /// Initializes the specified paramter value using the initialization function.
        member this.InitPar (seed: int) (ps: VarSpecT<'T>) = 
            let pi = parameters.[ps]
            let shp = this.ParameterStorage.[ps].Shape
            this.ParameterStorage.[ps] <- pi.Initializer seed shp |> device.ToDev

        /// Initializes all parameter values using their respective initialization functions.
        member this.InitPars (globalSeed: int) : unit =
            let rng = Random globalSeed
            for KeyValue(ps, _) in parameters do
                let seed = rng.Next ()
                this.InitPar seed ps

        /// Creates a function from the given expression using the model's ParameterSet and ParameterStorage
        /// using the specified result location.
        member this.Func (resultLoc: ArrayLocT, expr0: ExprT<'T>) =
            Func.make (compileSpec resultLoc) expr0 << useParStorage

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (resultLoc: ArrayLocT, expr0: ExprT<'T>, expr1: ExprT<'T>) =
            Func.make2 (compileSpec resultLoc) expr0 expr1 << useParStorage

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (resultLoc: ArrayLocT, expr0: ExprT<'T>, expr1: ExprT<'T>, expr2: ExprT<'T>) =
            Func.make3 (compileSpec resultLoc) expr0 expr1 expr2 << useParStorage

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
        
