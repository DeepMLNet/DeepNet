namespace Tensor.Expr

open Tensor
open Tensor.Backend
open DeepNet.Utils


/// Contains numeric values for variables.
type VarEnv = Map<string, Var * ITensor>

/// specification of variable storage locations
type VarLocs = Map<string, ITensorDevice>

/// specification of variable strides
type VarStrides = Map<string, int64 list>


/// Functions for working with VarEnv.
module VarEnv = 

    /// add variable value to environment
    let add (vs: Var) (value: #ITensor) (varEnv: VarEnv) : VarEnv =
        if TypeName.ofTypeInst value.DataType <> vs.TypeName then
            failwithf "Variable %s is of type %A but specified value is of type %A."
                      vs.Name vs.TypeName (TypeName.ofTypeInst value.DataType)
        varEnv |> Map.add vs.Name (vs, value :> ITensor)

    /// remove variable value from environment
    let remove (vs: Var) (varEnv: VarEnv) : VarEnv =
        varEnv |> Map.remove vs.Name

    /// get variable value from environment
    let get (vs: Var) (varEnv: VarEnv) : #ITensor =
        match varEnv |> Map.tryFind vs.Name with
        | Some (vs, value) -> vs |> box |> unbox
        | None -> failwithf "Variable %s is not present in the specified VarEnv." vs.Name

    ///// add variable value to environment
    //let add (var: Expr) (value: #ITensor) (varEnv: VarEnv) : VarEnv =
    //    addVarSpec (Expr.extractVar var) value varEnv

    ///// remove variable value from environment
    //let remove (var: Expr) (varEnv: VarEnv) : VarEnv =
    //    removeVarSpec (Expr.extractVar var) varEnv

    ///// get variable value from environment
    //let get (var: Expr) (varEnv: VarEnv) : #ITensor =
    //    getVarSpec (Expr.extractVar var) varEnv

    /// empty variable environment
    let (empty: VarEnv) =
        Map.empty

    /// joins two variable environments
    let join (a: VarEnv) (b: VarEnv) : VarEnv =
        Map.join a b

    /// Constructs a VarEnv from a sequence of variable, value tuples.
    let ofSeq (entries: (Var * #ITensor) seq) =
        (empty, entries)
        ||> Seq.fold (fun ve (var, value) -> ve |> add var value)

    /// infers symbol sizes from the variable environment
    let inferSymSizes (symSizeEnv: SymSizeEnv) (varEnv: VarEnv) : SymSizeEnv =
        (symSizeEnv, varEnv) ||> Map.fold 
            (fun env vName (vSym, vVal) ->   
                if Var.nDims vSym <> ITensor.nDims vVal then
                    failwithf "dimensionality mismatch: a value of shape %A was provided for variable %A"
                        (ITensor.shape vVal) vSym

                (Var.shape vSym, ITensor.shape vVal)
                ||> List.zip
                |> List.fold (fun env (svSym, svVal) ->
                    let failShape () =
                        let vSymShp = vSym.Shape |> ShapeSpec.substSymbols env 
                        failwithf "expected variable %A with (inferred) shape %A but got value with shape %A"
                            vSym vSymShp vVal.Shape
                    match svSym |> SizeSpec.substSymbols env |> SizeSpec.simplify  with
                    | SizeSpec.Base (BaseSize.Sym sym) -> env |> SymSizeEnv.add sym (SizeSpec.fix svVal)
                    | SizeSpec.Base (BaseSize.Fixed f) -> 
                        if f .= svVal then env
                        else failShape ()
                    | SizeSpec.Broadcast ->
                        if 1L = svVal then env
                        else failShape ()
                    | SizeSpec.Multinom m -> failShape ()
                ) env)


    /// substitues the given symbol sizes into the variable environment
    let substSymSizes symSizes (varEnv: VarEnv) : VarEnv =
        varEnv 
        |> Map.toSeq
        |> Seq.map (fun (name, (vs, value)) -> name, (Var.substSymSizes symSizes vs, value))
        |> Map.ofSeq

    /// checks that the values are valid in type and shape for the variables
    let check (varEnv: VarEnv) =
        varEnv |> Map.iter (fun name (vs, value) ->
            if TypeName.ofTypeInst value.DataType <> vs.TypeName then
                failwithf "variable %A was expected to be of type %A but a \
                           value with type %A was provided" vs.Name vs.TypeName.Type value.DataType

            let ss = Var.shape vs
            match ShapeSpec.tryEval ss with
            | Some ns when ITensor.shape value <> ns ->
                failwithf "variable %A was expected to be of shape %A (%A) but a \
                           value with shape %A waTensorded" vs.Name ns ss (ITensor.shape value)
            | None -> failwithf "variable %A contains size symbols that cannot be evaluated" vs
            | _ -> ()
        )
        
    /// gets the type names of the variable value arrays
    let valueTypeNames (varEnv: VarEnv) =
        varEnv |> Map.map (fun _ (vs, vVal) -> vs.TypeName)

    /// gets the locations of the variable value arrays
    let valueLocations (varEnv: VarEnv) : VarLocs =
        varEnv |> Map.map (fun _ (vs, vVal) -> ITensor.dev vVal)

    /// gets the strides of the variable value arrays
    let valueStrides (varEnv: VarEnv) : VarStrides =
        varEnv |> Map.map (fun _ (vs, vVal) -> vVal.Layout.Stride)


