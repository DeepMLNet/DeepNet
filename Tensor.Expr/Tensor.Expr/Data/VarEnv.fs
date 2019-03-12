namespace Tensor.Expr

open Tensor
open DeepNet.Utils



/// Contains numeric values for variables.
type VarEnv = VarEnv of Map<VarName, ITensor> with

    /// Map from variable names to numeric values.
    member this.Values =
        let (VarEnv values) = this
        values

    /// Variable names contained in this VarEnv.
    member this.VarNames = this.Values |> Map.keys

    /// Get variable value by name.
    member this.Item
        with get(varName: VarName) : ITensor = 
            match this.Values |> Map.tryFind varName with
            | Some value -> value
            | None ->
                failwithf "The variable %A does not exist in the variable environment." varName

    /// Creates a new, empty VarEnv.
    static member empty = VarEnv Map.empty
        
    /// Add variable value by variable name.
    static member addVarName (varName: VarName) (value: ITensor) (varEnv: VarEnv) =
        varEnv.Values |> Map.add varName value |> VarEnv

    /// Add base variable value to environment.
    static member addBaseVar (baseVar: Var) (value: ITensor) (varEnv: VarEnv) =
        if value.DataType <> baseVar.DataType then
            failwithf "Variable %A is of type %A but specified value is of type %A."
                      baseVar.Name baseVar.DataType value.DataType
        if value.Dev <> baseVar.Dev then
            failwithf "Variable %A is stored on device %A but specified value is stored on device %A."
                      baseVar.Name baseVar.Dev value.Dev
        match ShapeSpec.tryEval baseVar.Shape with
        | Some varShp when varShp <> value.Shape ->
            failwithf "Variable %A has shape %A but specified value is of shape %A."
                      baseVar.Name varShp value.Shape
        | _ -> ()
        varEnv |> VarEnv.addVarName baseVar.Name value

    /// Add variable value to environment.
    static member add (var: Var<'T>) (value: Tensor<'T>) (varEnv: VarEnv)  =
        varEnv |> VarEnv.addBaseVar var.Untyped value

    /// Remove variable by name from environment.
    static member remove (varName: VarName) (varEnv: VarEnv) : VarEnv =
        varEnv.Values |> Map.remove varName |> VarEnv

    /// joins two variable environments
    static member join (a: VarEnv) (b: VarEnv) : VarEnv =
        Map.join a.Values b.Values |> VarEnv    

    /// Constructs a VarEnv from a sequence of variable, value tuples.
    static member ofSeq (entries: (Var * ITensor) seq) =
        (VarEnv.empty, entries)
        ||> Seq.fold (fun ve (var, value) -> ve |> VarEnv.addBaseVar var value)
    

    ///// gets the type names of the variable value arrays
    //let valueTypeNames (varEnv: VarEnv) =
    //    varEnv |> Map.map (fun _ (vs, vVal) -> vs.TypeName)

    ///// gets the locations of the variable value arrays
    //let valueLocations (varEnv: VarEnv) : VarLocs =
    //    varEnv |> Map.map (fun _ (vs, vVal) -> ITensor.dev vVal)

    ///// gets the strides of the variable value arrays
    //let valueStrides (varEnv: VarEnv) : VarStrides =
    //    varEnv |> Map.map (fun _ (vs, vVal) -> vVal.Layout.Stride)


    ///// specification of variable storage locations
    //type VarLocs = Map<VarName, ITensorDevice>

    ///// specification of variable strides
    //type VarStrides = Map<VarName, int64 list>



