namespace SymTensor

open System
open FSharp.Reflection

open Tensor
open Tensor.Backend
open DeepNet.Utils



type private VarRecordHelpers () =
    static member PublishLocStride<'T when 'T: equality and 'T: comparison> 
            (expr: ExprT, loc: ITensorDevice, stride: int64 list option, mi: ModelInstance<'T>) =
        mi.SetLoc expr loc
        match stride with
        | Some stride -> mi.SetStride expr stride
        | None -> ()
    static member ValueArrayOnDev<'T> (value: 'T, dev: IDevice) = 
        HostTensor.scalar value |> dev.ToDev :> ITensor
    static member UVarSpecOfExpr<'T> (expr: ExprT) =
        Expr.extractVar expr
    static member WriteArrayToHDF<'T> (hdf: HDF5, dev: IDevice, name: string, value: Tensor<'T>) =
        value |> dev.ToHost |> HostTensor.write hdf name
    static member WriteScalarToHDF<'T> (hdf: HDF5, dev: IDevice, name: string, value: 'T) =
        value |> HostTensor.scalar |> HostTensor.write hdf name
    static member ReadArrayFromHDF<'T> (hdf: HDF5, dev: IDevice, name: string) : Tensor<'T> =
        HostTensor.read hdf name |> dev.ToDev
    static member ReadScalarFromHDF<'T> (hdf: HDF5, dev: IDevice, name: string) : 'T =
        HostTensor.read hdf name |> Tensor.value

type private ValueType =
    | Scalar of Type
    | Array of Type

type private RFieldInfo = {
    Expr:           obj
    VarSpec:        VarSpecT
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
                            valField.PropertyType.GetGenericTypeDefinition() = typedefof<Tensor<_>> then
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
                let varSpec = Generic.callGeneric<VarRecordHelpers, VarSpecT> "UVarSpecOfExpr" [baseType] exprData

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
                        let valueAry = 
                            Generic.callGeneric<VarRecordHelpers, ITensor> "ValueArrayOnDev" [baseType] (value, dev)
                        varEnv |> VarEnv.addVarSpec fi.VarSpec valueAry
                    | Array _ ->
                        varEnv |> VarEnv.addVarSpec fi.VarSpec (value :?> ITensor)
                )
            varEnvCache <- Some (value, varEnv)
            varEnv      
        
    /// extends the given function to accept a value record
    member this.Use (f: VarEnvT -> 'R) =
        fun (ve: VarEnvT) (value: 'RVal) -> f (VarEnv.join ve (this.VarEnv value))

    /// publishes the locations and strides of the used variables to the given ModelInstance
    member this.PublishLocAndStride (model: ModelInstance<'T>) =        
        fieldInfos
        |> Seq.iter (fun fi ->
            let loc = dev.DefaultLoc
            let shp = 
                fi.VarSpec.Shape 
                |> SymSizeEnv.substShape model.CompileEnv.SymSizes
                |> ShapeSpec.tryEval
            let stride = Option.map TensorLayout.rowMajorStride shp
            match fi.ValueType with
            | Scalar baseType | Array baseType ->
                Generic.callGeneric<VarRecordHelpers, unit> "PublishLocStride" [typeof<'T>] (fi.Expr, loc, stride, model)
        )

    /// Saves the record values as a HDF5 file.
    member this.SaveValue hdf prefix (value: 'RVal) =
        let values = FSharpValue.GetRecordFields value
        for fi, value in Seq.zip fieldInfos values do
            match fi.ValueType with
            | Scalar typ ->
                Generic.callGeneric<VarRecordHelpers, unit> "WriteScalarToHDF" [typ] 
                    (hdf, dev, prefix + "/" + fi.VarSpec.Name, value)
            | Array typ ->
                Generic.callGeneric<VarRecordHelpers, unit> "WriteArrayToHDF" [typ]
                    (hdf, dev, prefix + "/" + fi.VarSpec.Name, value)

    /// Load the record value from a HDF5 file using the specifed prefix
    member this.LoadValue hdf prefix : 'RVal =
        let values = seq {
            for fi in fieldInfos do
                match fi.ValueType with
                | Scalar typ ->
                    yield Generic.callGeneric<VarRecordHelpers, obj> "ReadScalarFromHDF" [typ] 
                        (hdf, dev, prefix + "/" + fi.VarSpec.Name)
                | Array typ ->
                    yield Generic.callGeneric<VarRecordHelpers, obj> "ReadArrayFromHDF" [typ]
                        (hdf, dev, prefix + "/" + fi.VarSpec.Name)
        }
        FSharpValue.MakeRecord (typeof<'RVal>, Array.ofSeq values) :?> 'RVal

