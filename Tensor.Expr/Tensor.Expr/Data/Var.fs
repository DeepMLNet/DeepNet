namespace Tensor.Expr

open DeepNet.Utils


/// Variable (a value that is passed in at runtime) specification.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type Var = {
    /// variable name
    Name:      string
    /// symbolic shape
    Shape:     ShapeSpec
    /// data type
    TypeName:  TypeName
} with

    /// data type
    member this.Type = TypeName.getType this.TypeName

    /// pretty string representation
    member this.Pretty = sprintf "%s<%s>%A" this.Name this.Type.Name this.Shape

    /// numeric shape
    member this.NShape = this.Shape |> ShapeSpec.eval

    /// Creates a variable specification using the specified name, type and symbolic shape.
    static member create name typ shape : Var =
        {Name=name; Shape=shape; TypeName=TypeName.ofTypeInst typ}

    /// create variable specifation by name and shape and type
    static member inline ofNameShapeAndTypeName name shape typeName : Var =
        {Name=name; Shape=shape; TypeName=typeName}

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

