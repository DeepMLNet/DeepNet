namespace SymTensor

open System.Runtime.InteropServices

open Basics
open ShapeSpec


[<AutoOpen>]
module TypeNameTypes =

    /// assembly qualified name of a .NET type
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
module ConstSpecTypes =

    /// scalar constant value
    type ConstSpecT = 
        | ConstInt of int
        | ConstDouble of double
        | ConstSingle of single
        | ConstBool of bool
        with
            member this.TypeName = 
                match this with
                | ConstInt _ -> TypeName.ofType<int>
                | ConstDouble _ -> TypeName.ofType<double>
                | ConstSingle _ -> TypeName.ofType<single>
                | ConstBool _ -> TypeName.ofType<bool>

            member this.GetValue() : 'T =
                match this with
                | ConstInt v -> v |> box |> unbox
                | ConstDouble v -> v |> box |> unbox
                | ConstSingle v -> v |> box |> unbox
                | ConstBool v -> v |> box |> unbox  
                
            member this.GetConvertedValue<'T>() : 'T =             
                match this with
                | ConstInt v -> v |> conv<'T>
                | ConstDouble v -> v |> conv<'T>
                | ConstSingle v -> v |> conv<'T>
                | ConstBool v -> v |> conv<'T>


module ConstSpec =
    let ofValue (value: obj) =
        match value.GetType() with
        | t when t = typeof<int> -> ConstInt (value |> unbox)
        | t when t = typeof<double> -> ConstDouble (value |> unbox)
        | t when t = typeof<single> -> ConstSingle (value |> unbox)
        | t when t = typeof<bool> -> ConstBool (value |> unbox)
        | t -> failwithf "unsupported constant type: %A" t

    let value (cs: ConstSpecT) =
        cs.GetValue ()

    let typeName (cs: ConstSpecT) =
        cs.TypeName

[<AutoOpen>]
module VarSpecTypes =

    /// non-generic variable specification interface
    type IVarSpec =
        inherit System.IComparable
        abstract member Name : string 
        abstract member Shape: ShapeSpecT
        abstract member Type: System.Type
        abstract member TypeName: TypeNameT
        abstract member SubstSymSizes: SymSizeEnvT -> IVarSpec

    /// variable specification: has a name, type and shape specificaiton
    [<StructuredFormatDisplay("\"{Name}\" {Shape}")>]
    type VarSpecT = 
        {
            Name:      string
            Shape:     ShapeSpecT
            TypeName:  TypeNameT
        }
        
        interface IVarSpec with
            member this.Name = this.Name
            member this.Shape = this.Shape
            member this.Type = TypeName.getType this.TypeName
            member this.TypeName = this.TypeName
            member this.SubstSymSizes symSizes = 
                {this with Shape=SymSizeEnv.substShape symSizes this.Shape} :> IVarSpec

//        interface System.IComparable with
//            member this.CompareTo otherObj =
//                let this = this :> IVarSpec
//                match otherObj with
//                | :? IVarSpec as othr -> 
//                    compare 
//                        (this.Name, this.Shape, this.TypeName) 
//                        (othr.Name, othr.Shape, othr.TypeName)
//                | _ -> invalidArg "otherObj" "cannot compare values of different types"
//



module VarSpec =

    /// create variable specifation by name and shape
    let inline ofNameAndShape name shape =
        {VarSpecT.Name=name; Shape=shape; TypeName=failwith "TODO"}

    /// name of variable
    let name (vs: #IVarSpec) = vs.Name

    /// shape of variable
    let shape (vs: #IVarSpec) = vs.Shape

    /// typename of variable
    let typeName (vs: #IVarSpec) = vs.TypeName

    /// substitutes the size symbol environment into the variable
    let substSymSizes symSizeEnv (vs: 'T when 'T :> IVarSpec) = 
        vs.SubstSymSizes symSizeEnv :?> 'T

