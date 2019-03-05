namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Backend


/// Variable name.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type VarName = VarName of VarPath * string with

    /// Pretty string.
    member this.Pretty =
        let (VarName (context, name)) = this
        context.Parts @ [name] |> String.concat "/"

    /// Unique variable path string.
    member this.Str =
        let (VarName (context, name)) = this
        context.Parts @ [name] 
        |> List.map (fun part -> "[" + part + "]")
        |> String.concat "/"        

    /// Create variable name from context and name.
    static member from (vp: VarPath, name: string) =
        VarName (vp, name)

    /// Create variable name in root context.
    static member from (name: string) = 
        VarName (VarPath.root, name)

    /// Create variable name by treating last part of context as variable name.
    static member from (vp: VarPath) =
        let ctx, name = VarPath.splitLast vp
        VarName (ctx, name)


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

    /// Create variable using context, data type and shape.
    static member make (ctx: Context, dataType, shape) : Var =
        {Name=VarName.from ctx.Path; TypeName=TypeName.ofTypeInst dataType; Shape=shape; Dev=ctx.Dev}

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


