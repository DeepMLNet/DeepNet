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
    static member inst (storePath: ContextPath) (sizeEnv: SizeEnv) (pg: ParSet) =
        // substitutes shapes into parameter variables
        let vars = pg.Vars |> List.map (Var.substSymSizes sizeEnv)

        // check that shapes of all variables can be evaluated
        for pv in vars do
            if not (Shape.canEval pv.Shape) then
                failwithf "Cannot evaluate shape of variable %A." pv

        // group variables by data type and storage device
        let pvGroups =
            vars
            |> List.groupBy (fun pv -> pv.TypeName, pv.Dev)
            |> Map.ofList
        
        // for each group perform instantiation
        let pvGroupStoragesAndInsts =
            pvGroups
            |> Map.map (fun (typeName, dev) pvs ->
                // create symbolic group storage variable
                let groupSize = pvs |> List.sumBy (fun pv -> Shape.nElem pv.Shape)
                let groupName = storePath / sprintf "ParSet<%A@%A>" typeName dev
                let groupVar = Var.make (VarName groupName, typeName.Type, dev, [groupSize])
                let groupExpr = UExpr groupVar
                
                // create group storage tensor
                let groupData = ITensor.zeros typeName.Type dev [Size.eval groupSize]

                // slice group storage variable/tensor to obtain expressions/subtensors for each parameter
                let pvInsts =
                    (Size.zero, pvs)
                    ||> List.mapFold (fun pos pv ->
                        let size = Shape.nElem pv.Shape
                        let varSlice = groupExpr.[pos .. pos + size - 1L] |> UExpr.reshape pv.Shape
                        let nPos, nSize = Size.eval pos, Size.eval size
                        let nShape = Shape.eval pv.Shape
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
            _SizeEnv = sizeEnv
            _StorePath = storePath
            _TypeDeviceGroups = pvGroupStorages
            _ParInsts = pvInsts
        }
        pgi.Init()
        pgi



/// An instantiated parameter set.
and ParSetInst = private {
    /// The parameter set this instance belongs to.
    _ParSet: ParSet
    /// Symbolic size substitutions.
    _SizeEnv: SizeEnv
    /// Base path of parameter storage.
    _StorePath: ContextPath
    /// Storages for each contained type/device combination.
    _TypeDeviceGroups: Map<TypeName * ITensorDevice, Var * ITensor>
    /// Contained parameter instances.
    _ParInsts: Map<VarName, ParInst>
} with 

    /// The parameter set this instance belongs to.
    member this.ParSet = this._ParSet

    /// Symbolic size substitutions.
    member this.SizeEnv = this._SizeEnv

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
    member this.Init () =
        for KeyValue (varName, pi) in this.ParInsts do
            pi.Var.Par.Value.Init.Value pi.Data

    /// Placeholder substituions to use this parameter group instance in an expression.
    member this.VarSubsts =
        this.ParInsts |> Map.map (fun _ pi -> pi.Expr)

    /// Variable values to use this parameter group instance for evaluation of an expression.
    member this.VarEnv =
        // flat storages
        let storageVarEnv =
            this.TypeDeviceGroups
            |> Map.toSeq
            |> Seq.map (fun (_, (var, data)) -> var.Name, data)
            |> Map.ofSeq
            |> VarEnv
        // individual parameter storages
        let parVarEnv =
            this.ParInsts
            |> Map.map (fun _varName pi -> pi.Data)
            |> VarEnv
        VarEnv.join storageVarEnv parVarEnv

    /// Uses this ParameterGroupInstance for the placeholder variables in the expression.
    member this.Use (expr: UExpr) =
        expr |> UExpr.substSymSizes this.SizeEnv |> UExpr.substVars this.VarSubsts
    
    /// Uses this ParameterGroupInstance for the placeholder variables in the expression.
    member this.Use (expr: Expr<'T>) =
        expr |> Expr<'T>.substSymSizes this.SizeEnv |> Expr<'T>.substVars this.VarSubsts

    /// Uses this ParameterGroupInstance for the placeholder variables in the expression.
    member this.Use (expr: MultiChannelExpr) =
        expr |> MultiChannelExpr.substSymSizes this.SizeEnv |> MultiChannelExpr.substVars this.VarSubsts

    /// Uses this ParameterGroupInstance for the placeholder variables in the EvalUpdateBundle.
    member this.Use (bndl: EvalUpdateBundle) =
        let exprs = bndl.Exprs |> Set.map this.Use
        let varUpdates = bndl.VarUpdates |> Map.map (fun _ expr -> this.Use expr)
        let dataUpdates = bndl.DataUpdates |> Map.map (fun _ expr -> this.Use expr)
        EvalUpdateBundle.make exprs varUpdates dataUpdates

    /// Uses this parameter group instance for evaluation of an expression or EvalUpdateBundle.
    member this.Use (varEnv: VarEnv) =
        VarEnv.join varEnv this.VarEnv

    /// Loads the parameter values from a previously saved HDF5 file using the specified prefix.
    /// If no prefix is specified, the store path is used as prefix.
    member this.Load (hdf, ?prefix) = 
        let prefix = defaultArg prefix this.StorePath.Str
        for KeyValue(_, pi) in this.ParInsts do
            let value = HostTensor.readUntyped hdf (prefix + "/" + pi.Var.Name.Str)
            pi.Data.TransferFrom value
            
    /// Saves the parameter values to a HDF5 file.
    /// Each parameter is stored in a separate HDF5 record under its name using the specified prefixed.
    /// If no prefix is specified, the store path is used as prefix.
    member this.Save (hdf, ?prefix) =
        let prefix = defaultArg prefix this.StorePath.Str
        for KeyValue(_, pi) in this.ParInsts do
            let value = ITensor.transfer HostTensor.Dev pi.Data
            HostTensor.write hdf (prefix + "/" + pi.Var.Name.Str) value

    /// Copies all parameter values from the specified parameter set instance.
    member this.CopyFrom (src: ParSetInst) =
        for KeyValue (key, (_, dstValue)) in this.TypeDeviceGroups do
            let _, srcValue = src.TypeDeviceGroups.[key]
            dstValue.CopyFrom srcValue

    /// Clones the specified parameter set instance.
    static member copy (psi: ParSetInst) =
        let clone = ParSet.inst psi.StorePath psi.SizeEnv psi.ParSet
        clone.CopyFrom psi
        clone



        