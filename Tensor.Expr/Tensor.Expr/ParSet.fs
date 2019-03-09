namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Backend



/// A parameter.
type Par = {
    /// Placeholder variable.
    Placeholder: Var
    /// Initialization function.
    Init: ITensor -> unit
} with
    /// Name of placeholder variable.
    member this.Name = this.Placeholder.Name



/// A parameter instance.
/// This provides storage for the value of the parameter.
type ParInst = {
    /// The parameter this parameter instance is representing.
    Par: Par
    /// The expression used to access the parameter value.
    Expr: UExpr
    /// The value of this parameter.
    Data: ITensor
}

    

/// A set of parameters.
type ParSet = private {
    /// The context path to store the parameter set.
    StoreBasePath: ContextPath
    /// Parameters in this set.
    Pars: Par list
    /// Placeholder variables in this set.
    Placeholders: Set<VarName>
} with

    /// Create new parameter group using the specified storage path.
    static member create (storePath: ContextPath): ParSet =
        {
            StoreBasePath = storePath
            Pars = List.empty
            Placeholders = Set.empty
        }

    /// No initialization.
    static member noInit (_: #ITensor) = ()

    /// Add a placeholder variable to this parameter group.
    static member addUntyped (var: Var) (init: ITensor -> unit) (pg: ParSet) =
        if pg.Placeholders |> Set.contains var.Name then
            failwithf "ParameterGroup %A already contains variable %A." pg.StoreBasePath var
        let pv = {
            Placeholder = var
            Init = init
        }
        {pg with 
            Pars = pv :: pg.Pars
            Placeholders = pg.Placeholders |> Set.add var.Name
        }

    /// Add a placeholder variable to this parameter group.
    static member add (var: Var<'T>) (init: Tensor<'T> -> unit) (pg: ParSet) =
        let initUntyped (t: ITensor) = init (t :?> Tensor<'T>)
        ParSet.addUntyped var.Untyped initUntyped pg

    /// Remove a placeholder variable from this parameter group.
    static member remove (varName: VarName) (pg: ParSet) =
        {pg with
            Pars = pg.Pars |> List.filter (fun pv -> pv.Placeholder.Name <> varName)
            Placeholders = pg.Placeholders |> Set.remove varName
        }

    /// Instantiate this parameter set
    static member inst (sizeEnv: SymSizeEnv) (pg: ParSet) =
        // substitute symbolic sizes into all variables 
        let pvs =
            pg.Pars
            |> List.map (fun pv -> {pv with Placeholder = pv.Placeholder |> Var.substSymSizes sizeEnv})

        // check that shapes of all variables can be evaluated
        for pv in pvs do
            if not (ShapeSpec.canEval pv.Placeholder.Shape) then
                failwithf "Cannot evaluate shape of variable %A." pv.Placeholder

        // group variables by data type and storage device
        let pvGroups =
            pg.Pars
            |> List.groupBy (fun pv -> pv.Placeholder.TypeName, pv.Placeholder.Dev)
            |> Map.ofList
        
        // for each group perform instantiation
        let pvGroupStoragesAndInsts =
            pvGroups
            |> Map.map (fun (typeName, dev) pvs ->
                // create symbolic group storage variable
                let groupSize = pvs |> List.sumBy (fun pv -> ShapeSpec.nElem pv.Placeholder.Shape)
                let groupName = pg.StoreBasePath / sprintf "<%A@%A>" typeName dev
                let groupVar = Var.make (VarName groupName, typeName.Type, dev, [groupSize])
                let groupExpr = UExpr groupVar
                
                // create group storage tensor
                let groupData = ITensor.zeros typeName.Type dev [SizeSpec.eval groupSize]

                // slice group storage variable/tensor to obtain expressions/subtensors for each parameter
                let pvInsts =
                    (SizeSpec.zero, pvs)
                    ||> List.mapFold (fun pos pv ->
                        let size = ShapeSpec.nElem pv.Placeholder.Shape
                        let varSlice = groupExpr.[pos .. pos + size - 1L] |> UExpr.reshape pv.Placeholder.Shape
                        let nPos, nSize = SizeSpec.eval pos, SizeSpec.eval size
                        let nShape = ShapeSpec.eval pv.Placeholder.Shape
                        let dataSlice = groupData.[nPos .. nPos + nSize - 1L] |> ITensor.reshape nShape
                        (pv.Placeholder.Name, {Par=pv; Expr=varSlice; Data=dataSlice}), pos + size)
                    |> fst
                    |> Map.ofList

                (groupVar, groupData), pvInsts)

        // extract storage for each group
        let pvGroupStorages = pvGroupStoragesAndInsts |> Map.map (fun _ (storage, insts) -> storage)

        // merge instantiations from all groups 
        let pvInsts = 
            pvGroupStoragesAndInsts 
            |> Map.toSeq
            |> Seq.map (fun (_, (storage, insts)) -> insts)
            |> Map.joinMany

        // build and initalize
        let pgi = {
            _ParSet = pg
            _TypeDeviceGroups = pvGroupStorages
            _ParInsts = pvInsts
        }
        ParSetInst.init pgi
        pgi



/// An instantiated parameter set.
and ParSetInst = private {
    /// The parameter set this instance belongs to.
    _ParSet: ParSet
    /// Storages for each contained type/device combination.
    _TypeDeviceGroups: Map<TypeName * ITensorDevice, Var * ITensor>
    /// Contained parameter instances.
    _ParInsts: Map<VarName, ParInst>
} with 

    /// The parameter set this instance belongs to.
    member this.ParSet = this._ParSet

    /// Contained parameter instances.
    member this.ParInsts = this._ParInsts

    /// Storages for each contained type/device combination.
    member this.TypeDeviceGroups = this._TypeDeviceGroups

    /// Variable containing all parameters for a type/device combination.
    member this.TypeDeviceVars = this.TypeDeviceGroups |> Map.map (fun _ (var, value) -> var)

    /// Tensor containing all parameters for a type/device combination.
    member this.TypeDeviceValues = this.TypeDeviceGroups |> Map.map (fun _ (var, value) -> value)

    /// Initializes all parameters.
    static member init (pgi: ParSetInst) =
        for KeyValue (varName, pi) in pgi.ParInsts do
            pi.Par.Init pi.Data

    /// Placeholder substituions to use this parameter group instance in an expression.
    member this.VarSubsts =
        this.ParInsts |> Map.map (fun _ pi -> pi.Expr)

    /// Variable values to use this parameter group instance for evaluation of an expression.
    member this.VarEnv =
        this.TypeDeviceGroups
        |> Map.toSeq
        |> Seq.map (fun (_, (var, data)) -> var.Name, data)
        |> Map.ofSeq
        |> VarEnv

    /// Uses this ParameterGroupInstance for the placeholder variables in the expression.
    member this.Use (expr: UExpr) =
        expr |> UExpr.substVars this.VarSubsts
    
    /// Uses this ParameterGroupInstance for the placeholder variables in the expression.
    member this.Use (expr: Expr<'T>) =
        expr |> Expr<'T>.substVars this.VarSubsts

    /// Uses this ParameterGroupInstance for the placeholder variables in the expression.
    member this.Use (expr: MultiChannelExpr) =
        expr |> MultiChannelExpr.substVars this.VarSubsts

    /// Uses this parameter group instance for evaluation of an expression.
    member this.Use (varEnv: VarEnv) =
        VarEnv.join varEnv this.VarEnv

        