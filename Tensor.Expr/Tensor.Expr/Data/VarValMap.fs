namespace Tensor.Expr

open Tensor
open Tensor.Backend
open DeepNet.Utils


/// A mapping from variable specification to variable value.
type VarValMap = Map<Var, ITensor>


/// Tools for working with VarMap.
module VarValMap =

    /// Creates a VarMap from a VarEnv and variable specifications.
    let make (VarEnv varEnv) (vars: Map<VarName, Var>) : VarValMap =
        vars
        |> Map.mapKeyValue (fun varName var -> var, varEnv.[varName])


    /// Checks that variable values are valid in type, device and shape.
    let check (varMap: VarValMap) =
        for KeyValue(vSym, vVal) in varMap do 
            if vVal.DataType <> vSym.DataType then
                failwithf "Variable %A was given value of data type %A." vSym vVal.DataType
            if vVal.Dev <> vSym.Dev then
                failwithf "Variable %A was given value stored on device %A." vSym vVal.Dev
            match ShapeSpec.tryEval vSym.Shape with
            | Some shp when vVal.Shape <> shp ->
                failwithf "Variable %A was given value with shape %A." vSym vVal.Shape
            | Some shp -> ()
            | None -> 
                failwithf "Variable %A shape cannot be evaluated." vSym


    /// Infers symbol sizes from variables values.
    let inferSymSizes (symSizeEnv: SymSizeEnv) (varMap: VarValMap) : SymSizeEnv =
        // TODO: allow shape inference for multinoms
        (symSizeEnv, varMap) 
        ||> Map.fold (fun env vSym vVal -> 
        
            if vSym.Shape.Length <> vVal.NDims then
                failwithf "Variable %A was given value with shape %A." vSym vVal.Shape

            (env, List.zip vSym.Shape vVal.Shape)
            ||> List.fold (fun env (svSym, svVal) ->

                let failShape () =
                    failwithf "Variable %A with shape %A is not compatible with value of shape %A." 
                        vSym (vSym.Shape |> ShapeSpec.substSymbols env) vVal.Shape

                match svSym |> Size.substSyms env with
                | Size.Atom (SizeAtom.Sym sym) -> 
                    env |> SymSizeEnv.add sym (Size.fix svVal)
                | Size.Atom (SizeAtom.Fixed f) -> 
                    if f .= svVal then env
                    else failShape ()
                | Size.Broadcast ->
                    if 1L = svVal then env
                    else failShape ()
                | Size.Multinom m -> 
                    failShape ()
            )
        )


