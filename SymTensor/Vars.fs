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
                sprintf "%A" this.Type
    
/// assembly qualified name of a .NET type
module TypeName =

    /// gets the System.Type associated by this TypeName
    let getType (tn: TypeNameT) =
        tn.Type

    /// gets the size of the represented type in bytes
    let size (tn: TypeNameT) =
        tn.Size

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
        | ConstDouble of double
        | ConstSingle of single
        | ConstBool of bool
        with
            /// the type name of the constant
            member this.TypeName = 
                match this with
                | ConstInt _ -> TypeName.ofType<int>
                | ConstDouble _ -> TypeName.ofType<double>
                | ConstSingle _ -> TypeName.ofType<single>
                | ConstBool _ -> TypeName.ofType<bool>

            /// gets the value which must be of type 'T
            member this.GetValue() : 'T =
                match this with
                | ConstInt v -> v |> box |> unbox
                | ConstDouble v -> v |> box |> unbox
                | ConstSingle v -> v |> box |> unbox
                | ConstBool v -> v |> box |> unbox  
                
            /// gets the value converting it to type 'T
            member this.GetConvertedValue<'T>() : 'T =             
                match this with
                | ConstInt v -> v |> conv<'T>
                | ConstDouble v -> v |> conv<'T>
                | ConstSingle v -> v |> conv<'T>
                | ConstBool v -> v |> conv<'T>

/// scalar constant value
module ConstSpec =

    /// creates a ConstSpecT from a scalar value
    let ofValue (value: obj) =
        match value.GetType() with
        | t when t = typeof<int> -> ConstInt (value |> unbox)
        | t when t = typeof<double> -> ConstDouble (value |> unbox)
        | t when t = typeof<single> -> ConstSingle (value |> unbox)
        | t when t = typeof<bool> -> ConstBool (value |> unbox)
        | t -> failwithf "unsupported constant type: %A" t

    /// gets the value 
    let value (cs: ConstSpecT) =
        cs.GetValue ()

    /// the type name
    let typeName (cs: ConstSpecT) =
        cs.TypeName

    let zeroOfType (typ: System.Type) =
        0 |> convTo typ |> ofValue

    let oneOfType (typ: System.Type) =
        1 |> convTo typ |> ofValue

    let twoOfType (typ: System.Type) =
        1 |> convTo typ |> ofValue


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
