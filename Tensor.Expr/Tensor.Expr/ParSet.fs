namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Backend



/// A parameter instance.
/// This provides storage for the value of the parameter.
type ParInst = {
    /// The placeholder variable for this parameter.
    Var: Var
    /// The expression used to access the parameter value.
    Expr: UExpr
    /// The value of this parameter.
    Data: ITensor
}

    

/// A set of parameters.
type ParSet = private {
    /// Variables that are placeholders for parameters.
    _Vars: Var list
    /// Variables names of variables in this set.
    _VarNames: Set<VarName>
} with

    /// Variables that are placeholders for parameters.
    member this.Vars = this._Vars

    /// Create new parameter group using the specified storage path.
    static member empty : ParSet = {
        _Vars = List.empty
        _VarNames = Set.empty
    }

    /// Add a placeholder variable to this parameter group.
    static member addUntyped (var: Var) (pg: ParSet) =
        if var.Par.IsNone then
            failwithf "Variable %A is not a parameter." var.Par
        if pg._VarNames |> Set.contains var.Name then
            failwithf "Parameter set already contains variable %A." var
        {pg with 
            _Vars = var :: pg.Vars
            _VarNames = pg._VarNames |> Set.add var.Name
        }

    /// Add a placeholder variable to this parameter group.
    static member add (var: Var<'T>) (pg: ParSet) =
        ParSet.addUntyped var.Untyped pg

    /// Remove a placeholder variable from this parameter group.
    static member remove (varName: VarName) (pg: ParSet) =
        {pg with
            _Vars = pg.Vars |> List.filter (fun pv -> pv.Name <> varName)
            _VarNames = pg._VarNames |> Set.remove varName
        }

    /// Create a parameter set from all parameter within the given expression that are
    /// under the specified path.
    static member fromUExpr (path: ContextPath) (expr: UExpr)  =
        expr.VarMap
        |> Map.toSeq
        |> Seq.choose (fun (_, var) ->
            match var.Par with
            | Some par when var.Name.Path |> ContextPath.startsWith path -> Some var
            | _ -> None)
        |> Seq.fold (fun parSet var -> parSet |> ParSet.addUntyped var) ParSet.empty

    /// Create a parameter set from all parameter within the given expression that are
    /// under the specified path.
    static member fromExpr (path: ContextPath) (expr: Expr<_>)  =
        ParSet.fromUExpr path expr.Untyped

    /// Merges two parameter set.
    /// The variables contained in the sets must be disjoint.
    static member merge (a: ParSet) (b: ParSet) =
        (a, b.Vars)
        ||> Seq.fold (fun a var -> a |> ParSet.addUntyped var)

    /// Instantiate the parameter set using the specified path for storage.
    static member inst (storePath: ContextPath) (pg: ParSet) =
        // check that shapes of all variables can be evaluated
        for pv in pg.Vars do
            if not (ShapeSpec.canEval pv.Shape) then
                failwithf "Cannot evaluate shape of variable %A." pv

        // group variables by data type and storage device
        let pvGroups =
            pg.Vars
            |> List.groupBy (fun pv -> pv.TypeName, pv.Dev)
            |> Map.ofList
        
        // for each group perform instantiation
        let pvGroupStoragesAndInsts =
            pvGroups
            |> Map.map (fun (typeName, dev) pvs ->
                // create symbolic group storage variable
                let groupSize = pvs |> List.sumBy (fun pv -> ShapeSpec.nElem pv.Shape)
                let groupName = storePath / sprintf "ParSet<%A@%A>" typeName dev
                let groupVar = Var.make (VarName groupName, typeName.Type, dev, [groupSize])
                let groupExpr = UExpr groupVar
                
                // create group storage tensor
                let groupData = ITensor.zeros typeName.Type dev [SizeSpec.eval groupSize]

                // slice group storage variable/tensor to obtain expressions/subtensors for each parameter
                let pvInsts =
                    (SizeSpec.zero, pvs)
                    ||> List.mapFold (fun pos pv ->
                        let size = ShapeSpec.nElem pv.Shape
                        let varSlice = groupExpr.[pos .. pos + size - 1L] |> UExpr.reshape pv.Shape
                        let nPos, nSize = SizeSpec.eval pos, SizeSpec.eval size
                        let nShape = ShapeSpec.eval pv.Shape
                        let dataSlice = groupData.[nPos .. nPos + nSize - 1L] |> ITensor.reshape nShape
                        (pv.Name, {Var=pv; Expr=varSlice; Data=dataSlice}), pos + size)
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
            _StorePath = storePath
            _TypeDeviceGroups = pvGroupStorages
            _ParInsts = pvInsts
        }
        ParSetInst.init pgi
        pgi



/// An instantiated parameter set.
and ParSetInst = private {
    /// The parameter set this instance belongs to.
    _ParSet: ParSet
    /// Base path of parameter storage.
    _StorePath: ContextPath
    /// Storages for each contained type/device combination.
    _TypeDeviceGroups: Map<TypeName * ITensorDevice, Var * ITensor>
    /// Contained parameter instances.
    _ParInsts: Map<VarName, ParInst>
} with 

    /// The parameter set this instance belongs to.
    member this.ParSet = this._ParSet

    /// Base path of parameter storage.
    member this.StorePath = this._StorePath

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
            pi.Var.Par.Value.Init.Value pi.Data

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

        