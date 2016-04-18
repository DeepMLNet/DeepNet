namespace SymTensor

open System.Reflection
open FSharp.Reflection

open Basics
open ArrayNDNS

module private RecordParamsId =
    let mutable id = 0   
    let next () =
        id <- id + 1
        id

type private RecordParamsHelpers () =
    static member VarOfType<'T> varname : ExprT<'T> = Expr.var varname []
    static member PublishLoc<'T when 'T: equality> varname loc (mb: ModelBuilder<'T>) =
        mb.SetLoc (RecordParamsHelpers.VarOfType<'T> varname) loc
    static member ValueArrayOnDev<'T> (value: 'T) (dev: IDevice) = 
        ArrayNDHost.scalar value |> dev.ToDev :> IArrayNDT

type RecordParams<'PVal, 'PExpr when 'PVal : equality> (dev: IDevice) =

    do 
        if not (FSharpType.IsRecord typeof<'PVal> && FSharpType.IsRecord typeof<'PExpr>) then
            failwith "'PVal and 'PExpr must both be record types"

    let pValName = typeof<'PVal>.Name
    let pValFields = FSharpType.GetRecordFields typeof<'PVal>
    let pExprFields = FSharpType.GetRecordFields typeof<'PExpr>

    do
        if Array.length pValFields <> Array.length pExprFields then
            failwith "'PVal and 'PExpr must both have the same number of fields"

        for pValField, pExprField in Seq.zip pValFields pExprFields do
            if pValField.Name <> pExprField.Name then
                failwithf "name mismatch for fields %s and %s" pValField.Name pExprField.Name
            let reqType =
                match pValField.PropertyType with
                | t when t=typeof<int>    -> typeof<ExprT<int>>
                | t when t=typeof<single> -> typeof<ExprT<single>>
                | t when t=typeof<double> -> typeof<ExprT<double>>
                | t -> failwithf "unsupported value type %A" t
            if pExprField.PropertyType <> reqType then
                failwithf "type mismatch for field %s: 'PVal type %A requires 'PExpr type %A but got %A"
                    pValField.Name pValField.PropertyType reqType pExprField.PropertyType

    let id = RecordParamsId.next()       
    let pVarNames = pValFields |> Array.map (fun f -> sprintf "%s%d.%s" pValName id f.Name)

    let mutable cache = None

    member this.Expr : 'PExpr =
        let exprs =
            (pValFields, pVarNames)
            ||> Array.map2 (fun f varname ->
                let mi = typeof<RecordParamsHelpers>.GetMethod("VarOfType", allBindingFlags)
                let m = mi.MakeGenericMethod f.PropertyType
                m.Invoke(null, [|box varname|])
            )
        FSharpValue.MakeRecord(typeof<'PExpr>, exprs) :?> 'PExpr

    member this.VarEnv (value: 'PVal) : VarEnvT =        
        match cache with
        | Some (lastValue, lastVarEnv) when lastValue = value -> lastVarEnv
        | _ ->
            let myVarEnv =
                (VarEnv.empty, Seq.zip3 pVarNames (FSharpValue.GetRecordFields value) pValFields)
                ||> Seq.fold (fun varEnv (varName, valData, valField) ->
                    let vs = UVarSpec.ofNameShapeAndTypeName varName [] (TypeName.ofTypeInst valField.PropertyType)
                    let mi = typeof<RecordParamsHelpers>.GetMethod("ValueArrayOnDev", allBindingFlags) 
                    let m = mi.MakeGenericMethod valField.PropertyType
                    let ary = m.Invoke(null, [|box valData; box dev|]) :?> IArrayNDT
                    varEnv |> VarEnv.addUVarSpec vs ary
                )
            cache <- Some (value, myVarEnv)
            myVarEnv      
        
    member this.Use (f: VarEnvT -> 'R) =
        fun (ve: VarEnvT) (value: 'PVal) -> f (VarEnv.join ve (this.VarEnv value))

    member this.PublishLoc (mb: ModelBuilder<_>) =
        (pValFields, pVarNames)
        ||> Array.iter2 (fun f varname ->
            let mi = typeof<RecordParamsHelpers>.GetMethod("PublishLoc", allBindingFlags)
            let m = mi.MakeGenericMethod f.PropertyType
            m.Invoke(null, [|box varname; dev.DefaultLoc; mb|]) |> ignore
        )
