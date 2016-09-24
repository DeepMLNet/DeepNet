namespace SymTensor

open System
open System.Reflection
open FSharp.Reflection

open Basics
open ArrayNDNS
open UExprTypes

type private VarRecordHelpers () =
    static member PublishLoc<'T when 'T: equality and 'T: comparison> (expr: ExprT) (loc: ArrayLocT) (mi: ModelInstance<'T>) =
        mi.SetLoc expr loc
    static member ValueArrayOnDev<'T> (value: 'T) (dev: IDevice) = 
        ArrayNDHost.scalar value |> dev.ToDev :> IArrayNDT
    static member UVarSpecOfExpr<'T> (expr: ExprT) =
        UVarSpec.ofExpr expr
    static member WriteArrayToHDF<'T> (hdf: HDF5) (dev: IDevice) (name: string) (value: ArrayNDT<'T>) =
        value |> dev.ToHost |> ArrayNDHDF.write hdf name
    static member WriteScalarToHDF<'T> (hdf: HDF5) (dev: IDevice) (name: string) (value: 'T) =
        value |> ArrayNDHost.scalar |> ArrayNDHDF.write hdf name
    static member ReadArrayFromHDF<'T> (hdf: HDF5) (dev: IDevice) (name: string) : ArrayNDT<'T> =
        ArrayNDHDF.read hdf name |> dev.ToDev
    static member ReadScalarFromHDF<'T> (hdf: HDF5) (dev: IDevice) (name: string) : 'T =
        ArrayNDHDF.read hdf name |> ArrayND.value

type private ValueType =
    | Scalar of Type
    | Array of Type

type private RFieldInfo = {
    Expr:           obj
    VarSpec:        UVarSpecT
    ValueType:      ValueType
}

/// Maps a value record (containing scalars or ArrayNDTs) to a expression record
/// (containing ExprTs).
type VarRecord<'RVal, 'RExpr when 'RVal: equality> (rExpr:      'RExpr,
                                                    dev:        IDevice) =
    do 
        if not (FSharpType.IsRecord typeof<'RVal> && FSharpType.IsRecord typeof<'RExpr>) then
            failwith "'PVal and 'PExpr must both be record types"

    let valFields = FSharpType.GetRecordFields typeof<'RVal>
    let exprFields = FSharpType.GetRecordFields typeof<'RExpr>
    let exprDatas = FSharpValue.GetRecordFields rExpr

    do if Array.length valFields <> Array.length exprFields then
        failwith "'PVal and 'PExpr must both have the same number of fields"

    let fieldInfos = 
        seq {
            for valField, exprField, exprData in Seq.zip3 valFields exprFields exprDatas do
                if valField.Name <> exprField.Name then
                    failwithf "name mismatch for fields %s and %s" valField.Name exprField.Name

                // get value type and corresponding expression type
                let baseType, valueType, exprType =                   
                    if valField.PropertyType.IsGenericType && 
                            valField.PropertyType.GetGenericTypeDefinition() = typedefof<ArrayNDT<_>> then
                        // ArrayNDT<'T> => ExprT
                        let bt = valField.PropertyType.GetGenericArguments().[0]
                        bt, Array bt, typeof<ExprT>
                    else
                        // 'T => ExprT (scalar)
                        let bt = valField.PropertyType
                        bt, Scalar bt, typeof<ExprT>

                if exprField.PropertyType <> exprType then
                    failwithf "type mismatch for field %s: 'PVal type %A requires 'PExpr type %A but got %A"
                        valField.Name valField.PropertyType exprType exprField.PropertyType

                // extract UVarSpecT
                let mi = typeof<VarRecordHelpers>.GetMethod("UVarSpecOfExpr", allBindingFlags) 
                let m = mi.MakeGenericMethod baseType
                let varSpec = m.Invoke(null, [|exprData|]) :?> UVarSpecT

                yield {Expr=exprData; VarSpec=varSpec; ValueType=valueType}
        } 

    let mutable varEnvCache = None

    /// the storage device
    member this.Dev  = dev

    /// the expression record
    member this.Expr = rExpr

    /// the VarEnv containing the values in the passed value record
    member this.VarEnv (value: 'RVal) : VarEnvT =        
        match varEnvCache with
        | Some (lastValue, lastVarEnv) when lastValue = value -> lastVarEnv
        | _ ->
            let values = FSharpValue.GetRecordFields value
            let varEnv =
                (VarEnv.empty, Seq.zip fieldInfos values)
                ||> Seq.fold (fun varEnv (fi, value) ->
                    match fi.ValueType with
                    | Scalar baseType ->
                        let mi = typeof<VarRecordHelpers>.GetMethod("ValueArrayOnDev", allBindingFlags) 
                        let m = mi.MakeGenericMethod baseType
                        let valueAry = m.Invoke(null, [|box value; box dev|]) :?> IArrayNDT
                        varEnv |> VarEnv.addUVarSpec fi.VarSpec valueAry
                    | Array _ ->
                        varEnv |> VarEnv.addUVarSpec fi.VarSpec (value :?> IArrayNDT)
                )
            varEnvCache <- Some (value, varEnv)
            varEnv      
        
    /// extends the given function to accept a value record
    member this.Use (f: VarEnvT -> 'R) =
        fun (ve: VarEnvT) (value: 'RVal) -> f (VarEnv.join ve (this.VarEnv value))

    /// publishes the locations of the used variables to the given ModelInstance
    member this.PublishLoc (model: ModelInstance<'T>) =
        fieldInfos
        |> Seq.iter (fun fi ->
            match fi.ValueType with
            | Scalar baseType | Array baseType ->
                let mi = typeof<VarRecordHelpers>.GetMethod("PublishLoc", allBindingFlags)
                let m = mi.MakeGenericMethod typeof<'T>
                m.Invoke(null, [|fi.Expr; dev.DefaultLoc; model|]) |> ignore
        )

    /// Saves the record values as a HDF5 file.
    member this.SaveValue path (value: 'RVal) =
        use hdf = HDF5.OpenWrite path
        let values = FSharpValue.GetRecordFields value
        for fi, value in Seq.zip fieldInfos values do
            match fi.ValueType with
            | Scalar typ ->
                let mi = typeof<VarRecordHelpers>.GetMethod("WriteScalarToHDF", allBindingFlags)
                let m = mi.MakeGenericMethod typ
                m.Invoke(null, [|box hdf; box dev; box fi.VarSpec.Name; value|]) |> ignore
            | Array typ ->
                let mi = typeof<VarRecordHelpers>.GetMethod("WriteArrayToHDF", allBindingFlags)
                let m = mi.MakeGenericMethod typ
                m.Invoke(null, [|box hdf; box dev; box fi.VarSpec.Name; value|]) |> ignore

    /// Load the record value from a HDF5 file.
    member this.LoadValue path : 'RVal =
        use hdf = HDF5.OpenRead path
        let values = seq {
            for fi in fieldInfos do
                match fi.ValueType with
                | Scalar typ ->
                    let mi = typeof<VarRecordHelpers>.GetMethod("ReadScalarFromHDF", allBindingFlags)
                    let m = mi.MakeGenericMethod typ
                    yield m.Invoke(null, [|box hdf; box dev; box fi.VarSpec.Name|]) 
                | Array typ ->
                    let mi = typeof<VarRecordHelpers>.GetMethod("ReadArrayFromHDF", allBindingFlags)
                    let m = mi.MakeGenericMethod typ
                    yield m.Invoke(null, [|box hdf; box dev; box fi.VarSpec.Name|])         
        }
        FSharpValue.MakeRecord (typeof<'RVal>, Array.ofSeq values) :?> 'RVal

