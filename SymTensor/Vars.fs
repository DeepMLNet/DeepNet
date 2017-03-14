namespace SymTensor

open System.Runtime.InteropServices

open Basics
open ShapeSpec


[<AutoOpen>]
/// TypeName types
module TypeNameTypes =

    /// assembly qualified name of a .NET type
    [<StructuredFormatDisplay("{Pretty}")>]
    type TypeNameT = 
        | TypeName of string
        with 
            /// gets the System.Type associated by this TypeName        
            member this.Type = 
                match this with
                | TypeName tn -> System.Type.GetType(tn)

            /// gets the size of the represented type in bytes
            member this.Size =
                Marshal.SizeOf this.Type

            /// pretty string
            member this.Pretty =
                sprintf "%s" this.Type.Name
    
/// assembly qualified name of a .NET type
module TypeName =

    /// gets the System.Type associated by this TypeName
    let getType (tn: TypeNameT) =
        tn.Type

    /// gets the size of the represented type in bytes
    let size (tn: TypeNameT) =
        tn.Size

    /// gets the size of the represented type in bytes as int64
    let size64 (tn: TypeNameT) =
        tn.Size |> int64

    /// gets the TypeName associated with the given type
    let ofType<'T> =
        TypeName (typeof<'T>.AssemblyQualifiedName)

    /// gets the TypeName associated with the given System.Type object
    let ofTypeInst (t: System.Type) =
        TypeName t.AssemblyQualifiedName

    /// gets the TypeName associated with the type of then given object
    let ofObject (o: obj) =
        ofTypeInst (o.GetType())


[<AutoOpen>]
/// scalar constant value types
module ConstSpecTypes =

    /// scalar constant value
    [<StructuralEquality; StructuralComparison>]
    type ConstSpecT = 
        | ConstInt of int
        | ConstInt64 of int64
        | ConstDouble of double
        | ConstSingle of single
        | ConstBool of bool
        with
            /// the type name of the constant
            member this.TypeName = 
                match this with
                | ConstInt _    -> TypeName.ofType<int>
                | ConstInt64 _  -> TypeName.ofType<int64>
                | ConstDouble _ -> TypeName.ofType<double>
                | ConstSingle _ -> TypeName.ofType<single>
                | ConstBool _   -> TypeName.ofType<bool>

            /// the type of the constant
            member this.Type =
                this.TypeName.Type

            /// gets the value which must be of type 'T
            member this.GetValue() : 'T =
                match this with
                | ConstInt v    -> v |> box |> unbox
                | ConstInt64 v  -> v |> box |> unbox
                | ConstDouble v -> v |> box |> unbox
                | ConstSingle v -> v |> box |> unbox
                | ConstBool v   -> v |> box |> unbox  
            
            /// the value as object
            member this.Value =
                match this with
                | ConstInt v    -> v |> box 
                | ConstInt64 v  -> v |> box 
                | ConstDouble v -> v |> box
                | ConstSingle v -> v |> box 
                | ConstBool v   -> v |> box 
                
            /// gets the value converting it to type 'T
            member this.GetConvertedValue<'T>() : 'T =   
                this.Value |> conv<'T>          
    
    /// matches a zero constant of any type
    let (|ConstZero|_|) cs =
        match cs with
        | ConstInt 0
        | ConstInt64 0L
        | ConstSingle 0.0f
        | ConstDouble 0.0 -> Some ()
        | _ -> None
        

/// scalar constant value
module ConstSpec =

    /// creates a ConstSpecT from a scalar value
    let ofValue (value: obj) =
        match value.GetType() with
        | t when t = typeof<int> -> ConstInt (value |> unbox)
        | t when t = typeof<int64> -> ConstInt64 (value |> unbox)
        | t when t = typeof<double> -> ConstDouble (value |> unbox)
        | t when t = typeof<single> -> ConstSingle (value |> unbox)
        | t when t = typeof<bool> -> ConstBool (value |> unbox)
        | t -> failwithf "unsupported constant type: %A" t

    /// gets the value 
    let value (cs: ConstSpecT) =
        cs.GetValue ()

    /// the type name of the constant
    let typeName (cs: ConstSpecT) =
        cs.TypeName

    /// the type of the constant
    let typ (cs: ConstSpecT) =
        cs.Type

    /// one of specified type
    let one (typ: System.Type) =
        1 |> convTo typ |> ofValue

    /// two of specified type
    let two (typ: System.Type) =
        1 |> convTo typ |> ofValue

    /// zero constant of specified type
    let zero typ =
        match typ with
        | _ when typ = typeof<int>    -> ConstInt 0
        | _ when typ = typeof<int64>  -> ConstInt64 0L
        | _ when typ = typeof<double> -> ConstDouble 0.0
        | _ when typ = typeof<single> -> ConstSingle 0.0f
        | _ when typ = typeof<bool>   -> ConstBool false
        | _ -> failwithf "unsupported type %A" typ

    /// minimum value constant of specified type
    let minValue typ =
        match typ with
        | _ when typ = typeof<int>    -> ConstInt (System.Int32.MinValue)
        | _ when typ = typeof<int64>  -> ConstInt64 (System.Int64.MinValue)
        | _ when typ = typeof<double> -> ConstDouble (System.Double.MinValue)
        | _ when typ = typeof<single> -> ConstSingle (System.Single.MinValue)
        | _ when typ = typeof<bool>   -> ConstBool false
        | _ -> failwithf "unsupported type %A" typ

    /// maximum value constant of specified type
    let maxValue typ =
        match typ with
        | _ when typ = typeof<int>    -> ConstInt (System.Int32.MaxValue)
        | _ when typ = typeof<int64>  -> ConstInt64 (System.Int64.MaxValue)
        | _ when typ = typeof<double> -> ConstDouble (System.Double.MaxValue)
        | _ when typ = typeof<single> -> ConstSingle (System.Single.MaxValue)
        | _ when typ = typeof<bool>   -> ConstBool true
        | _ -> failwithf "unsupported type %A" typ

    /// true if constant is zero
    let isZero cs =
        match cs with
        | ConstZero -> true
        | _ -> false


[<AutoOpen>]
/// variable specification types
module VarSpecTypes =

    /// variable specification: has a name, type and shape specificaiton
    [<StructuredFormatDisplay("{Pretty}")>]
    type VarSpecT = {
        Name:      string
        Shape:     ShapeSpecT
        TypeName:  TypeNameT
    } with
        member this.Type = TypeName.getType this.TypeName
        member this.Pretty = sprintf "%s<%s>%A" this.Name this.Type.Name this.Shape
        member this.NShape = this.Shape |> ShapeSpec.eval
        

/// variable specification
module VarSpec =

    let create name typ shape : VarSpecT =
        {Name=name; Shape=shape; TypeName=TypeName.ofTypeInst typ}

    /// create variable specifation by name and shape and type
    let inline ofNameShapeAndTypeName name shape typeName : VarSpecT =
        {Name=name; Shape=shape; TypeName=typeName}

    /// name of variable
    let name (vs: VarSpecT) =
        vs.Name

    /// shape of variable
    let shape (vs: VarSpecT) =
        vs.Shape

    /// number of dimensions of variable
    let nDims vs =
        shape vs |> ShapeSpec.nDim

    /// type of variable
    let typ (vs: VarSpecT) = 
        vs.TypeName |> TypeName.getType 

    /// typename of variable
    let typeName (vs: VarSpecT) =
        vs.TypeName

    /// substitutes the size symbol environment into the variable
    let substSymSizes symSizes (vs: VarSpecT) = 
        {vs with Shape=SymSizeEnv.substShape symSizes vs.Shape} 

    /// gets variable by name
    let tryFindByName (vs: VarSpecT) map =
        map |> Map.tryPick 
            (fun cvs value -> 
                if name cvs = name vs then Some value
                else None)

    /// gets variable by name
    let findByName vs map =
        match tryFindByName vs map with
        | Some value -> value
        | None -> raise (System.Collections.Generic.KeyNotFoundException())
