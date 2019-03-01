namespace Tensor.Expr

open Tensor
open Tensor.Backend
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
        with get(varName: VarName) : ITensor = this.Values.[varName] 

    /// Creates a new, empty VarEnv.
    static member empty = VarEnv Map.empty
        
    /// Add variable value by variable name.
    static member addVarName (varName: VarName) (value: ITensor) (varEnv: VarEnv) =
        varEnv.Values |> Map.add varName value |> VarEnv

    /// Add base variable value to environment.
    static member addBaseVar (baseVar: BaseVar) (value: ITensor) (varEnv: VarEnv) =
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
    static member add (var: Var<'T>) (value: ITensor) (varEnv: VarEnv)  =
        varEnv |> VarEnv.addBaseVar var.BaseVar value

    /// Remove variable by name from environment.
    static member remove (varName: VarName) (varEnv: VarEnv) : VarEnv =
        varEnv.Values |> Map.remove varName |> VarEnv

    /// joins two variable environments
    static member join (a: VarEnv) (b: VarEnv) : VarEnv =
        Map.join a.Values b.Values |> VarEnv    

    /// Constructs a VarEnv from a sequence of variable, value tuples.
    static member ofSeq (entries: (BaseVar * ITensor) seq) =
        (VarEnv.empty, entries)
        ||> Seq.fold (fun ve (var, value) -> ve |> VarEnv.addBaseVar var value)



///// specification of variable storage locations
//type VarLocs = Map<VarName, ITensorDevice>

///// specification of variable strides
//type VarStrides = Map<VarName, int64 list>

    ///// infers symbol sizes from the variable environment
    //static member inferSymSizes (symSizeEnv: SymSizeEnv) (varEnv: VarEnv) : SymSizeEnv =
    //    (symSizeEnv, varEnv) ||> Map.fold 
    //        (fun env vName (vSym, vVal) ->   
    //            if Var.nDims vSym <> ITensor.nDims vVal then
    //                failwithf "dimensionality mismatch: a value of shape %A was provided for variable %A"
    //                    (ITensor.shape vVal) vSym

    //            (Var.shape vSym, ITensor.shape vVal)
    //            ||> List.zip
    //            |> List.fold (fun env (svSym, svVal) ->
    //                let failShape () =
    //                    let vSymShp = vSym.Shape |> ShapeSpec.substSymbols env 
    //                    failwithf "expected variable %A with (inferred) shape %A but got value with shape %A"
    //                        vSym vSymShp vVal.Shape
    //                match svSym |> SizeSpec.substSymbols env |> SizeSpec.simplify  with
    //                | SizeSpec.Base (BaseSize.Sym sym) -> env |> SymSizeEnv.add sym (SizeSpec.fix svVal)
    //                | SizeSpec.Base (BaseSize.Fixed f) -> 
    //                    if f .= svVal then env
    //                    else failShape ()
    //                | SizeSpec.Broadcast ->
    //                    if 1L = svVal then env
    //                    else failShape ()
    //                | SizeSpec.Multinom m -> failShape ()
    //            ) env)


    ///// substitues the given symbol sizes into the variable environment
    //let substSymSizes symSizes (varEnv: VarEnv) : VarEnv =
    //    varEnv 
    //    |> Map.toSeq
    //    |> Seq.map (fun (name, (vs, value)) -> name, (Var.substSymSizes symSizes vs, value))
    //    |> Map.ofSeq

    ///// checks that the values are valid in type and shape for the variables
    //let check (varEnv: VarEnv) =
    //    varEnv |> Map.iter (fun name (vs, value) ->
    //        if TypeName.ofTypeInst value.DataType <> vs.TypeName then
    //            failwithf "variable %A was expected to be of type %A but a \
    //                       value with type %A was provided" vs.Name vs.TypeName.Type value.DataType

    //        let ss = Var.shape vs
    //        match ShapeSpec.tryEval ss with
    //        | Some ns when ITensor.shape value <> ns ->
    //            failwithf "variable %A was expected to be of shape %A (%A) but a \
    //                       value with shape %A was provided" vs.Name ns ss (ITensor.shape value)
    //        | None -> failwithf "variable %A contains size symbols that cannot be evaluated" vs
    //        | _ -> ()
    //    )
        
    ///// gets the type names of the variable value arrays
    //let valueTypeNames (varEnv: VarEnv) =
    //    varEnv |> Map.map (fun _ (vs, vVal) -> vs.TypeName)

    ///// gets the locations of the variable value arrays
    //let valueLocations (varEnv: VarEnv) : VarLocs =
    //    varEnv |> Map.map (fun _ (vs, vVal) -> ITensor.dev vVal)

    ///// gets the strides of the variable value arrays
    //let valueStrides (varEnv: VarEnv) : VarStrides =
    //    varEnv |> Map.map (fun _ (vs, vVal) -> vVal.Layout.Stride)


