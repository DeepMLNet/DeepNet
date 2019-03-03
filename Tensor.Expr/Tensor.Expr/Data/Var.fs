namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Backend


/// Variable name.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type VarName = VarName of string with
    member this.Pretty =
        let (VarName name) = this
        name


/// Variable specification (not generic over data type).
[<StructuredFormatDisplay("{Pretty}")>]        
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
    /// data type
    member this.DataType = TypeName.getType this.TypeName

    /// pretty string representation
    member this.Pretty = 
        sprintf "%A<%s@%A>%A" this.Name this.DataType.Name this.Dev this.Shape

    /// Create variable using name, shape, data type name and storage device.
    static member make (name, typeName, dev, shape) : Var =
        {Name=name; TypeName=typeName; Shape=shape; Dev=dev}

    /// Create variable using name, shape, data type and storage device.
    static member make (name, dataType, dev, shape) : Var =
        {Name=name; TypeName=TypeName.ofTypeInst dataType; Shape=shape; Dev=dev}

    /// name of variable
    static member name (vs: Var) = vs.Name

    /// shape of variable
    static member shape (vs: Var) = vs.Shape

    /// number of dimensions of variable
    static member nDims (vs: Var) = vs.Shape |> ShapeSpec.nDim

    /// type of variable
    static member dataType (vs: Var) = vs.DataType 

    /// typename of variable
    static member typeName (vs: Var) = vs.TypeName

    /// storage device of variable
    static member dev (vs: Var) = vs.Dev

    /// substitutes the size symbol environment into the variable
    static member substSymSizes symSizes (vs: Var) = 
        {vs with Shape=SymSizeEnv.substShape symSizes vs.Shape} 


