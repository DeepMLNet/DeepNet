namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Backend


/// Variable name.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type VarName = VarName of string with
    member this.Pretty =
        let (VarName name) = this
        name


/// Variable (a value that is passed in at runtime) specification.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type Var = {
    /// variable name
    Name:      VarName
    /// data type
    TypeName:  TypeName
    /// storage device
    Dev:       ITensorDevice
    /// symbolic shape
    Shape:     ShapeSpec
} with

    /// Create variable using name, shape, data type name and storage device.
    static member make (name, typeName, dev, shape) : Var =
        {Name=name; TypeName=typeName; Shape=shape; Dev=dev}

    /// Create variable using name, shape, data type and storage device.
    static member make (name, typ, dev, shape) : Var =
        {Name=name; TypeName=TypeName.ofTypeInst typ; Shape=shape; Dev=dev}

    /// Create variable using name, shape, data type and storage device.
    static member make (name, typ, dev, shape) : Var =
        {Name=VarName name; TypeName=TypeName.ofTypeInst typ; Shape=shape; Dev=dev}

    /// Create variable from name, shape and storage device.
    static member make<'T> (name, dev, shape) : Var =
        {Name=name; TypeName=TypeName.ofType<'T>; Shape=shape; Dev=dev}

    /// Create variable from name, shape and storage device.
    static member make<'T> (name, dev, shape) : Var = 
        {Name=VarName name; TypeName=TypeName.ofType<'T>; Shape=shape; Dev=dev}

    /// data type
    member this.Type = TypeName.getType this.TypeName

    /// pretty string representation
    member this.Pretty = 
        sprintf "%A<%s@%A>%A" this.Name this.Type.Name this.Dev this.Shape

    /// numeric shape
    member this.NShape = this.Shape |> ShapeSpec.eval

    /// name of variable
    static member name (vs: Var) = vs.Name

    /// shape of variable
    static member shape (vs: Var) = vs.Shape

    /// number of dimensions of variable
    static member nDims vs = Var.shape vs |> ShapeSpec.nDim

    /// type of variable
    static member typ (vs: Var) = vs.TypeName |> TypeName.getType 

    /// typename of variable
    static member typeName (vs: Var) = vs.TypeName

    /// storage device of variable
    static member dev (vs: Var) = vs.Dev

    /// substitutes the size symbol environment into the variable
    static member substSymSizes symSizes (vs: Var) = 
        {vs with Shape=SymSizeEnv.substShape symSizes vs.Shape} 

    /// gets variable by name
    static member tryFindByName (vs: Var) map =
        map |> Map.tryPick 
            (fun cvs value -> 
                if Var.name cvs = Var.name vs then Some value
                else None)

    /// gets variable by name
    static member findByName vs map =
        match Var.tryFindByName vs map with
        | Some value -> value
        | None -> raise (System.Collections.Generic.KeyNotFoundException())

