namespace Tensor.Cuda

open System
open System.IO
open System.Threading
open System.Reflection
open System.Reflection.Emit
open System.Runtime.InteropServices
open System.Security.Cryptography
open System.Text
open System.Text.RegularExpressions

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor.Utils
open DeepNet.Utils


/// C++ data type helpers
module internal Cpp =
    /// C++ data type for given type instance
    let cppTypeInst (typ: System.Type) = 
        match typ with
        | _ when typ = typeof<single>    -> "float"
        | _ when typ = typeof<double>    -> "double"
        | _ when typ = typeof<sbyte>     -> "int8_t"
        | _ when typ = typeof<byte>      -> "uint8_t"
        | _ when typ = typeof<int32>     -> "int32_t"
        | _ when typ = typeof<uint32>    -> "uint32_t"
        | _ when typ = typeof<int64>     -> "int64_t"
        | _ when typ = typeof<uint64>    -> "uint64_t"
        | _ when typ = typeof<bool>      -> "bool"
        | _ when typ = typeof<nativeint> -> "ptr_t"
        | _ -> failwithf "no C++ datatype for %A" typ

    /// C++ data type for given type 
    let cppType<'T> = cppTypeInst typeof<'T>



/// Dynamic type helpers.
module internal DynamicTypes =
    let currentDomain = Thread.GetDomain()
    let dynAsmName = new AssemblyName("TensorDynamicTypes")
    let asmBuilder = AssemblyBuilder.DefineDynamicAssembly(dynAsmName, AssemblyBuilderAccess.Run)
    let modBuilder = asmBuilder.DefineDynamicModule("Module")


/// C++ tensor marshaling
type NativeTensor = {
    DataType:       Type
    BasePtr:        nativeint
    Offset:         int64
    Shape:          int64 list
    Stride:         int64 list
    Storage:        obj
}
   
/// C++ tensor marshaling
type internal NativeTensorInfo = {
    DataType:       Type
    NDims:          int
}


/// C++ tensor marshaling
module internal NativeTensor =
    let private typeCache = Dictionary<string, Type> ()

    let getType (dataType: Type) (nDims: int) =
        lock DynamicTypes.modBuilder (fun () ->
            let typeName = sprintf "Tensor_%s_%d" dataType.Name nDims
            match typeCache.TryFind typeName with
            | Some typ -> typ
            | None ->
                // define new value type with attribute [<Struct; StructLayout(LayoutKind.Sequential)>]
                let mb = DynamicTypes.modBuilder
                let tb = mb.DefineType(typeName, 
                                       TypeAttributes.Public ||| TypeAttributes.SequentialLayout,
                                       typeof<ValueType>)

                // define fields
                tb.DefineField("Base", typeof<nativeint>, FieldAttributes.Public) |> ignore
                tb.DefineField("Offset", typeof<int64>, FieldAttributes.Public) |> ignore
                for d = 0 to max (nDims-1) 0 do
                    tb.DefineField(sprintf "Shape%d" d, typeof<int64>, FieldAttributes.Public) |> ignore
                for d = 0 to max (nDims-1) 0 do
                    tb.DefineField(sprintf "Stride%d" d, typeof<int64>, FieldAttributes.Public) |> ignore

                // create defined type and cache it
                let typ = tb.CreateTypeInfo().AsType()
                typeCache.[typeName] <- typ
                typ
        )

    /// C++ Tensor<T, nDims> struct ready for marshaling
    let marshal (nt: NativeTensor) =          
        let nDims = nt.Shape.Length
        if nt.Stride.Length <> nDims then
            failwith "shape and stride must have same length"

        // create struct 
        let strctType = getType nt.DataType nDims
        let strct = Activator.CreateInstance(strctType)

        // set data
        strctType.GetField("Base").SetValue(strct, nt.BasePtr)
        strctType.GetField("Offset").SetValue(strct, nt.Offset)
        for d, (size, str) in List.indexed (List.zip nt.Shape nt.Stride) do
            strctType.GetField(sprintf "Shape%d" d).SetValue(strct, size)
            strctType.GetField(sprintf "Stride%d" d).SetValue(strct, str)
        strct

    /// C++ native tensor type string
    let cppName (nti: NativeTensorInfo) =
        sprintf "Tensor<%s, %d>" (Cpp.cppTypeInst nti.DataType) nti.NDims

    let mangledName (nti: NativeTensorInfo) =
        sprintf "Tensor_%s_%d" (Cpp.cppTypeInst nti.DataType) nti.NDims

    let validInstance (nti: NativeTensorInfo) (nt: NativeTensor) =
        nt.DataType = nti.DataType && 
        nt.Shape.Length = nti.NDims && 
        nt.Stride.Length = nti.NDims


/// C++ NativeIdx 
type NativeIdxTensors = {
    NDims:      int                      // not present in native struct
    Idxs:       NativeTensor option list
} 
   
/// C++ NativeIdx template info
type NativeIdxTensorsInfo = {
    NDims:      int
    NIdxs:      int
}


/// C++ NativeIdx marshalling
module internal NativeIdxTensors =
    let private typeCache = Dictionary<string, Type> ()

    let private getType (nDims: int) (nIdxs: int) =
        lock DynamicTypes.modBuilder (fun () ->
            let typeName = sprintf "IdxTensors_%d_%d" nDims nIdxs
            match typeCache.TryFind typeName with
            | Some typ -> typ
            | None ->
                // define new value type with attribute [<Struct; StructLayout(LayoutKind.Sequential)>]
                let mb = DynamicTypes.modBuilder
                let tb = mb.DefineType(typeName, 
                                       TypeAttributes.Public ||| TypeAttributes.SequentialLayout,
                                       typeof<ValueType>)

                // define fields
                let nt = NativeTensor.getType typeof<int64> nDims
                for d = 0 to max (nIdxs-1) 0 do
                    tb.DefineField(sprintf "Idxs%d" d, nt, FieldAttributes.Public) |> ignore
                for d = 0 to max (nIdxs-1) 0 do
                    tb.DefineField(sprintf "Specified%d" d, typeof<byte>, FieldAttributes.Public) |> ignore

                // create defined type and cache it
                let typ = tb.CreateTypeInfo().AsType()
                typeCache.[typeName] <- typ
                typ
        )

    /// C++ IdxTensors<nDims, nIdxs> struct ready for marshalling
    let marshal (nit: NativeIdxTensors) =          
        let nIdxs = nit.Idxs.Length
        let nDims = nit.NDims
        if not (nit.Idxs |> List.forall (Option.forall (fun it -> it.Shape.Length = nDims))) then
            failwith "NDims does not match dimensionality of Idxs tensors"

        // create struct 
        let strctType = getType nDims nIdxs
        let strct = Activator.CreateInstance(strctType)

        // set data
        let unspecIdx = {
            DataType    = typeof<int64>
            BasePtr     = nativeint 0
            Offset      = 0L
            Shape       = List.replicate nDims 0L
            Stride      = List.replicate nDims 0L 
            Storage     = null
        }
        for d, idx in List.indexed nit.Idxs do
            let idx, specified = 
                match idx with
                | Some idx -> NativeTensor.marshal idx, 1uy
                | None -> NativeTensor.marshal unspecIdx, 0uy
            strctType.GetField(sprintf "Idxs%d" d).SetValue(strct, idx)
            strctType.GetField(sprintf "Specified%d" d).SetValue(strct, specified)
        strct

    /// C++ type string
    let cppName (niti: NativeIdxTensorsInfo) =
        sprintf "IdxTensors<%d, %d>" niti.NDims niti.NIdxs

    /// C++ mangled name
    let mangledName (niti: NativeIdxTensorsInfo) =
        sprintf "IdxTensors_%d_%d" niti.NDims niti.NIdxs

    let validInstance (niti: NativeIdxTensorsInfo) (nit: NativeIdxTensors) =
        nit.NDims = niti.NDims &&
        nit.Idxs.Length = niti.NIdxs
