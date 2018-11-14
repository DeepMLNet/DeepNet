namespace SymTensor

open System

open Tensor
open Tensor.Backend
open DeepNet.Utils


[<AutoOpen>]
module ModelContextTypes =

    /// function that returns an initialization value for a parameter
    type Initializer<'T> = int -> int64 list -> Tensor<'T>

    /// model parameter
    type ParameterInfo<'T> = {
        Expr:           ExprT 
        Initializer:    Initializer<'T>
    }

    /// Model evaluation device specification.
    type IDevice =
        abstract member Allocator:      NShapeSpec -> Tensor<'T> 
        abstract member ToDev:          Tensor<'T> -> Tensor<'T>
        abstract member ToHost:         Tensor<'T> -> Tensor<'T>
        abstract member Compiler:       IUExprCompiler
        abstract member DefaultLoc:     ITensorDevice
        abstract member DefaultFactory: IUExprCompiler * CompileEnvT

    /// Evaluates the model on the host.
    let DevHost = { 
        new IDevice with
            member this.Allocator shp   = HostTensor.zeros shp 
            member this.ToDev ary       = ary
            member this.ToHost ary      = ary
            member this.Compiler        = { new IUExprCompiler with 
                                              member this.Name = "Host"
                                              member this.Compile env exprs = onHost env exprs }
            member this.DefaultLoc      = HostTensor.Dev
            member this.DefaultFactory  = this.Compiler, {CompileEnv.empty with ResultLoc=this.DefaultLoc}
    }

    /// A set of symbolic variables forming a set of parameters for a model.
    [<StructuredFormatDisplay("{Pretty}")>]
    type ParameterSetT<'T when 'T: equality and 'T: comparison> 
            (name:           string, 
             parameters:     Var seq) =

        let flatVarName = "PS_" + name
        let pars = parameters |> Seq.toList |> List.sort

        let shapes = pars |> List.map Var.shape

        /// layout of data vector
        let startIdxs, totalElems =
            shapes
            |> List.mapFold (fun pos shp -> pos, pos + ShapeSpec.nElem shp)
                SizeSpec.zero

        /// variable containing all parameters
        let flatVar : ExprT = Expr.var<'T> (flatVarName) [totalElems]

        /// parameter variables
        let parameterSubstExprs =
            (startIdxs, shapes)
            ||> List.map2 (fun startIdx shp ->
                let elems = ShapeSpec.nElem shp
                let v = flatVar.[startIdx .. startIdx + elems - 1L]
                Expr.reshape shp v)
            |> List.zip pars
            |> Map.ofList

        /// mapping from parameter expression to parameter variable
        let parameterOfSubstExpr =
            parameterSubstExprs
            |> Map.toSeq
            |> Seq.map (fun (var, expr) -> expr, var)
            |> Map.ofSeq
          
        member this.Name = name
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
            match derivs.Jacobians |> Map.tryFind (Expr.extractVar this.Flat) with
            | Some deriv -> deriv
            | None -> 
                printfn "Warning: ParameterSet %s does not occur in expression for differentiation." name
                Expr.zeros<'T> [expr.NElems; this.Flat.NElems]

        /// variable for a given parameter
        member this.Item
            with get (par: Var) = parameterSubstExprs.[par]

        /// returns the parameter for the given substituion expression
        member this.ParameterOfSubstExpr expr = 
            match parameterOfSubstExpr |> Map.tryFind expr with
            | Some var -> var
            | None -> failwith "this ParameterSet does not contain a variable for the specified expression"

        /// substitutes this ParameterSet into the given expression
        member this.Subst expr =
            let parSubst = pars |> List.map (fun par -> Expr.makeVar par, this.[par]) |> Map.ofList
            expr |> Expr.subst parSubst

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
             symSizes:        SymSizeEnv,
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
                let elems = List.fold (*) 1L shp
                let v = dataVal.[startIdx .. startIdx + elems - 1L]
                Tensor.reshapeView shp v)
            |> List.zip parameterSet.Parameters
            |> Map.ofList

        /// flat array containing all parameters
        member this.Flat 
            with get () = dataVal
            and set value = dataVal.[Fill] <- value

        /// value for a given parameter
        member this.Item
            with get (par: Var) : Tensor<'T> = parameterVals.[par]
            and set (par: Var) (value: Tensor<'T>) = parameterVals.[par].[Fill] <- value

        /// value for a given parameter
        member this.Item
            with get (par: ExprT) : Tensor<'T> = this.[Expr.extractVar par]
            and set (par: ExprT) (value: Tensor<'T>) = this.[Expr.extractVar par] <- value

        /// Uses the values of this ParameterStorageT for the the corresponding
        /// ParameterSetT in the given variable environment.
        member this.Use varEnv =
            varEnv |> VarEnv.add parameterSet.Flat this.Flat

        /// Loads the parameter values from a previously saved HDF5 file using the specified prefix.
        member this.Load (hdf, ?prefix) = 
            let prefix = defaultArg prefix ""
            for KeyValue(vs, vsView) in parameterVals do
                let value = HostTensor.read hdf (prefix + "/" + vs.Name)
                vsView.[Fill] <- device.ToDev value

        /// Saves the parameter values to a HDF5 file.
        /// Each parameter is stored in a separate HDF5 record under its name using the specified prefixed.
        member this.Save (hdf, ?prefix) =
            let prefix = defaultArg prefix ""
            for KeyValue(vs, vsView) in parameterVals do
                let value = device.ToHost vsView
                HostTensor.write hdf (prefix + "/" + vs.Name) value

        /// prints the shapes of all parameters contained in this ParameterStorage
        member this.PrintShapes () =
            printfn "ParameterStorage for %s contains the parameters:" parameterSet.Name
            for par, value in 
                    parameterVals |> Map.toList |> List.sortBy fst do
                printfn "%-50s %A" (par.Name + ": ") value.Shape


    /// A model builder. 
    /// The type 'T specifies the data type of the model parameters.
    [<StructuredFormatDisplay("{PrettyString}")>]
    type ModelBuilder<'T when 'T: equality and 'T: comparison> 
                                            (context:       string,
                                             isSubModule:   bool) =

        let mutable subMBs = Map.empty
        let mutable parameters : Map<Var, ParameterInfo<'T>> = Map.empty
        let mutable vars : Set<Var> = Set.empty
        let mutable symSizes = []
        let mutable symSizeEnv = SymSizeEnv.empty
        let mutable varLocs : VarLocsT = Map.empty
        let mutable varStrides : VarStridesT = Map.empty
        let mutable instantiated = false

        let toSizeSpec (name: string) =            
            SizeSpec.symbol name

        let toShapeSpec (shapeObj: obj list) =
            shapeObj |> List.map
                (function
                | :? string as symName -> toSizeSpec symName
                | :? int64 as f when f = -1L -> SizeSpec.broadcastable
                | :? int64 as f when f >= 0L -> SizeSpec.fix f
                | r -> failwithf "size must be either a size symbol name (string), \
                                  a fixed size (positive int64) or -1L for broadcast, but got %A" r)

        let checkVar var =
            if not (vars |> Set.contains var) then
                failwithf "this ModelBuilder does not contain the variable %A" var

        let defaultInitializer (seed: int) (shp: int64 list) =
            let rng = Random(seed)
            rng.SeqDouble (-0.01, 0.01) 
            |> Seq.map conv<'T>
            |> HostTensor.ofSeqWithShape shp

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
        [<RequiresExplicitTypeArguments>]
        member this.Var<'V> (name: string) (shape: ShapeSpec) : ExprT =
            let v = Expr.var<'V> (context + "." + name) shape
            vars <- vars |> Set.add (Expr.extractVar v)
            v

        /// Creates and returns a model parameter.
        member this.Param (name: string, shape: ShapeSpec, ?initializer: Initializer<'T>) =
            let initializer = defaultArg initializer defaultInitializer
            if instantiated then failwith "cannot add parameter after model has been instantiated"

            let p = Expr.var<'T> (context + "." + name) shape
            parameters <- parameters |> Map.add (Expr.extractVar p) {Expr=p; Initializer=initializer}
            p

        /// Creates and returns a symbolic size.
        /// If the name starts with ">" a dynamic size is created.
        member this.Size (name: string) =
            let s = toSizeSpec name
            symSizes <- s :: symSizes
            s

        /// Returns a fixed size. If size is -1L then a broadcastable size one is created.
        member this.Fix size =
            if size = -1L then SizeSpec.broadcastable
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

        /// sets a symbolic size to a numeric value
        member this.SetSize size value =
            match size with
            | SizeSpec.Base (BaseSize.Sym sym) -> symSizeEnv <- SymSizeEnv.add sym (SizeSpec.fix value) symSizeEnv
            | _ -> failwith "need a size symbol to set size"

        /// gets the numeric value of a previously set symbolic size
        member this.GetSize size =
            match size with
            | SizeSpec.Base (BaseSize.Sym sym) -> 
                match symSizeEnv |> Map.tryFind sym with
                | Some (SizeSpec.Base (BaseSize.Fixed _) as value) -> SizeSpec.eval value
                | _ -> failwith "size symbol is unknown or does not a have a numeric value"
            | _ -> failwith "need a size symbol to set size"

        /// sets the location of the given variable
        member this.SetLoc var loc =
            let vs = Expr.extractVar var
            checkVar vs
            varLocs <- varLocs |> Map.add vs loc

        /// sets the stride of the given variable
        member this.SetStride var stride =
            let vs = Expr.extractVar var
            checkVar vs
            varStrides <- varStrides |> Map.add vs stride            

        /// Infers variable location, variable strides and symbolic sizes by 
        /// matching a symbolic variable to the given value.
        member this.UseTmplVal var (value: Tensor<'T>) =
            // infer symbolic sizes
            let inferredSizes = 
                VarEnv.empty 
                |> VarEnv.add var value
                |> VarEnv.inferSymSizes symSizeEnv
            symSizeEnv <- Map.join symSizeEnv inferredSizes

            // infer location and strides
            varLocs <- varLocs |> Map.add (Expr.extractVar var) (Tensor.dev value)
            varStrides <- varStrides |> Map.add (Expr.extractVar var) (value.Layout.Stride)

        /// Inferred size symbol values
        member this.SymSizeEnv = symSizeEnv

        /// Variable locations
        member this.VarLocs = varLocs

        /// instantiates the model with numeric sizes for all size symbols and initializes 
        /// the parameter values
        member this.Instantiate (device: IDevice, ?sizeValues: Map<SizeSpec, int64>, ?canDelay: bool) =
            let canDelay = defaultArg canDelay true
            let sizeValues = defaultArg sizeValues Map.empty
            if isSubModule then failwith "a submoule cannot be instantiated"
            if instantiated then failwith "this model has already been instantiated"

            // set sizes
            for KeyValue(size, value) in sizeValues do
                this.SetSize size value

            // Check that SymSizes that are required for ParameterStorage instantiation can
            // be evaluated to a numeric value.
            let neededSymSizes = 
                this.Parameters
                |> Map.toSeq
                |> Seq.collect (fun (vs, _) -> Var.shape vs)
                |> Set.ofSeq
            let missingSymSizes =
                neededSymSizes
                |> Set.filter (SymSizeEnv.subst symSizeEnv >> SizeSpec.canEval >> not)
            if not (Set.isEmpty missingSymSizes) then
                failwithf "Cannot instantiate model because size symbols %A have no value."
                    (missingSymSizes |> Set.toList)

            // apply default variable location to variables with unspecified location
            let varLocs = 
                if canDelay then varLocs
                else
                    (varLocs, vars)
                    ||> Set.fold (fun varLocs var ->
                        match varLocs |> Map.tryFind var with
                        | Some _ -> varLocs
                        | None -> varLocs |> Map.add var device.DefaultLoc)

            // apply row-major strides to variables with unspecified strides
            let varStrides = 
                if canDelay then varStrides
                else
                    (varStrides, vars)
                    ||> Set.fold (fun varStrides var ->
                        match varStrides |> Map.tryFind var, ShapeSpec.tryEval var.Shape with
                        | None, Some nShape -> varStrides |> Map.add var (TensorLayout.rowMajorStride nShape)
                        | _, _ -> varStrides)

            // create compile environement
            let compileEnv = {
                SymSizes       = symSizeEnv
                VarLocs        = varLocs
                VarStrides     = varStrides
                ChannelStrides = Map.empty
                ResultLoc      = device.DefaultLoc
                CanDelay       = canDelay
            }

            // instantiate
            instantiated <- true
            let mi = ModelInstance<'T> (context, this.Parameters, device, compileEnv)
            mi.InitPars 0
            mi
            
        member this.PrettyString = 
            sprintf "%A {%s}"
                context (this.Parameters |> Map.toSeq |> Seq.map (fst >> sprintf "%A") |> String.concat ", ")


    /// A model with numeric sizes for all size symbols and allocated parameter storage.
    and ModelInstance<'T when 'T: equality and 'T: comparison> 
                                            (context:         string,
                                             parameters:      Map<Var, ParameterInfo<'T>>,
                                             device:          IDevice,                                             
                                             compileEnv:      CompileEnvT) =

        let mutable compileEnv = compileEnv

        // create parameter set and parameter storage
        let parameterSpecs = parameters |> Map.toSeq |> Seq.map fst
        let parameterSet = ParameterSetT<'T> (context, parameterSpecs)
        let parameterStorage = ParameterStorageT<'T> (parameterSet, compileEnv.SymSizes, device)        

        let compileSpec resultLoc = 
            // add ParameterStorage to variable locations
            let psVar = Expr.extractVar parameterSet.Flat 
            let psVal = parameterStorage.Flat
            let varLocs =
                compileEnv.VarLocs
                |> Map.add psVar (Tensor.dev psVal)
            let varStrides =
                compileEnv.VarStrides
                |> Map.add psVar psVal.Layout.Stride

            device.Compiler, {compileEnv with ResultLoc  = resultLoc
                                              VarLocs    = varLocs
                                              VarStrides = varStrides}

        let useParStorage = parameterStorage.Use

        /// substitutes the ParameterStorage into the given expression
        member this.Use expr =
            let parSubst = 
                parameters 
                |> Map.toSeq 
                |> Seq.map (fun (vs, pi) -> pi.Expr, parameterSet.[vs])
                |> Map.ofSeq
            expr
            |> Expr.subst parSubst
            |> Expr.substSymSizes compileEnv.SymSizes

        /// inserts the ParameterStorage into the given variable environment
        member this.Use varEnv =
            varEnv
            |> parameterStorage.Use 
            |> VarEnv.substSymSizes compileEnv.SymSizes

        /// the device this model instance is stored on
        member this.Device = device

        /// ParameterSet of this module's (and all submodules') parameteres
        member this.ParameterSet = parameterSet

        /// symbolic flat parameter vector
        member this.ParameterVector = 
            this.ParameterSet.Flat
            |> Expr.substSymSizes compileEnv.SymSizes

        /// Parameter values.
        member this.ParameterStorage = parameterStorage

        /// numeric flat parameter vector
        member this.ParameterValues 
            with get () = this.ParameterStorage.Flat
            and set value = this.ParameterStorage.Flat <- value

        /// Compile environment.
        member this.CompileEnv = compileEnv

        /// sets the location of the given variable
        member this.SetLoc var loc =
            let uvs = Expr.extractVar var 
            match compileEnv.VarLocs |> Map.tryFind uvs with
            | Some prvLoc when prvLoc <> loc ->
                failwithf "cannot change location of variable %A from %A to %A after model instantiation"
                    uvs prvLoc loc
            | _ -> ()
            compileEnv <- {compileEnv with VarLocs = compileEnv.VarLocs |> Map.add uvs loc}

        /// sets the stride of the given variable
        member this.SetStride var stride =
            let uvs = Expr.extractVar var 
            match compileEnv.VarStrides |> Map.tryFind uvs with
            | Some prvStride when prvStride <> stride ->
                failwithf "cannot change stride of variable %A from %A to %A after model instantiation"
                    uvs prvStride stride
            | _ -> ()
            compileEnv <- {compileEnv with VarStrides = compileEnv.VarStrides |> Map.add uvs stride}

        /// Load parameter values.
        member this.LoadPars (hdf, ?prefix) = this.ParameterStorage.Load (hdf, ?prefix=prefix)

        /// Save parameter values.
        member this.SavePars (hdf, ?prefix) = this.ParameterStorage.Save (hdf, ?prefix=prefix)

        /// Initializes the specified paramter value using the initialization function.
        member this.InitPar (seed: int) (ps: Var) = 
            let pi = parameters.[ps]
            let shp = this.ParameterStorage.[ps].Shape
            this.ParameterStorage.[ps] <- pi.Initializer seed shp |> device.ToDev

        /// Initializes all parameter values using their respective initialization functions.
        member this.InitPars (globalSeed: int) : unit =
            let rng = Random globalSeed
            for KeyValue(ps, _) in parameters do
                let seed = rng.Next ()
                this.InitPar seed ps

        /// value for a given parameter
        member this.Item
            with get (par: ExprT) : Tensor<'T> = this.ParameterStorage.[par]
            and set (par: ExprT) (value: Tensor<'T>) = this.ParameterStorage.[par] <- value

        member this.Func (resultLoc: ITensorDevice, exprs: ExprT list) =
            let exprs = exprs |> List.map this.Use 
            Func.makeMany<'T> (compileSpec resultLoc) exprs << useParStorage

        /// Creates a function from the given expression using the model's ParameterSet and ParameterStorage
        /// using the specified result location.
        member this.Func<'T0> (resultLoc: ITensorDevice, expr0: ExprT) =
            let expr0 = this.Use expr0
            Func.make<'T0> (compileSpec resultLoc) expr0 << useParStorage

        member this.Func (resultLoc: ITensorDevice, expr0: ExprT) = 
            this.Func<'T> (resultLoc, expr0)

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func<'T0, 'T1> (resultLoc: ITensorDevice, expr0: ExprT, expr1: ExprT) =
            let expr0 = this.Use expr0
            let expr1 = this.Use expr1
            Func.make2<'T0, 'T1> (compileSpec resultLoc) expr0 expr1 << useParStorage

        member this.Func (resultLoc: ITensorDevice, expr0: ExprT, expr1: ExprT) =
            this.Func<'T, 'T> (resultLoc, expr0, expr1)

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func<'T0, 'T1, 'T2> (resultLoc: ITensorDevice, expr0: ExprT, expr1: ExprT, expr2: ExprT) =
            let expr0 = this.Use expr0
            let expr1 = this.Use expr1
            let expr2 = this.Use expr2
            Func.make3<'T0, 'T1, 'T2> (compileSpec resultLoc) expr0 expr1 expr2 << useParStorage

        member this.Func (resultLoc: ITensorDevice, expr0: ExprT, expr1: ExprT, expr2: ExprT) =
            this.Func<'T, 'T, 'T> (resultLoc, expr0, expr1, expr2)

        member this.Func (exprs: ExprT list) =
            this.Func (device.DefaultLoc, exprs)

        /// Creates a function from the given expression using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func<'T0> (expr0: ExprT) =
            this.Func<'T0> (device.DefaultLoc, expr0)

        member this.Func (expr0: ExprT) =
            this.Func (device.DefaultLoc, expr0)

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (expr0: ExprT, expr1: ExprT) =
            this.Func (device.DefaultLoc, expr0, expr1)

        member this.Func<'T0, 'T1> (expr0: ExprT, expr1: ExprT) =
            this.Func<'T0, 'T1> (device.DefaultLoc, expr0, expr1)

        /// Creates a function from the given expressions using the model's ParameterSet and ParameterStorage
        /// using the devices default result location.
        member this.Func (expr0: ExprT, expr1: ExprT, expr2: ExprT) =
            this.Func (device.DefaultLoc, expr0, expr1, expr2)
        
        member this.Func<'T0, 'T1, 'T2> (expr0: ExprT, expr1: ExprT, expr2: ExprT) =
            this.Func<'T0, 'T1, 'T2> (device.DefaultLoc, expr0, expr1, expr2)
