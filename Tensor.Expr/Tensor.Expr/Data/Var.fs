namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Backend


/// Variable name.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type VarName = VarName of string with
    member this.Pretty =
        let (VarName name) = this
        name


/// Base variable specification (non-generic over data type).
[<StructuredFormatDisplay("{Pretty}")>]        
type BaseVar = {
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
    static member make (name, typeName, dev, shape) : BaseVar =
        {Name=name; TypeName=typeName; Shape=shape; Dev=dev}

    /// Create variable using name, shape, data type and storage device.
    static member make (name, dataType, dev, shape) : BaseVar =
        {Name=name; TypeName=TypeName.ofTypeInst dataType; Shape=shape; Dev=dev}

    /// name of variable
    static member name (vs: BaseVar) = vs.Name

    /// shape of variable
    static member shape (vs: BaseVar) = vs.Shape

    /// number of dimensions of variable
    static member nDims (vs: BaseVar) = vs.Shape |> ShapeSpec.nDim

    /// type of variable
    static member dataType (vs: BaseVar) = vs.DataType 

    /// typename of variable
    static member typeName (vs: BaseVar) = vs.TypeName

    /// storage device of variable
    static member dev (vs: BaseVar) = vs.Dev

    /// substitutes the size symbol environment into the variable
    static member substSymSizes symSizes (vs: BaseVar) = 
        {vs with Shape=SymSizeEnv.substShape symSizes vs.Shape} 



/// Variable: a value that is passed in at runtime.
[<StructuredFormatDisplay("{Pretty}")>]
type Var<'T> (baseVar: BaseVar) =
    do
        if typeof<'T> <> baseVar.DataType then
            failwithf "Cannot create Var<%A> for BaseVar of data type %A."
                      typeof<'T> baseVar.DataType

    // Base variable specification.
    member this.BaseVar = baseVar

    /// Create variable using name, shape and storage device.
    new (name, dev, shape) =
        Var<'T> ({Name=name; TypeName=TypeName.ofType<'T>; Dev=dev; Shape=shape})

    /// Create variable using name, shape and storage device.
    new (name, dev, shape) =
        Var<'T> ({Name=VarName name; TypeName=TypeName.ofType<'T>; Dev=dev; Shape=shape})

    interface System.IEquatable<Var<'T>> with
        member this.Equals other = this.BaseVar = other.BaseVar

    override this.Equals other =
        match other with
        | :? Var<'T> as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<Var<'T>> with
        member this.CompareTo other = compare this.BaseVar other.BaseVar

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Var<'T> as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare Var to type %A." (other.GetType())

    override this.GetHashCode() = hash this.BaseVar
        
    /// name of variable
    member this.Name = this.BaseVar.Name
    /// name of variable    
    static member name (vs: Var<'T>) = vs.Name

    /// storage device of variable
    member this.Dev = this.BaseVar.Dev
    /// storage device of variable    
    static member dev (vs: Var<'T>) = vs.Dev

    /// shape of variable
    member this.Shape = this.BaseVar.Shape
    /// shape of variable    
    static member shape (vs: Var<'T>) = vs.Shape

    /// data type of variable
    member this.DataType = this.BaseVar.DataType
    /// type of variable
    static member dataType (vs: Var<'T>) = vs.DataType

    /// pretty string representation
    member this.Pretty = this.BaseVar.Pretty 

    /// number of dimensions of variable
    static member nDims vs = Var.shape vs |> ShapeSpec.nDim

    /// substitutes the size symbol environment into the variable
    static member substSymSizes symSizes (vs: Var<'T>) = 
        vs.BaseVar |> BaseVar.substSymSizes symSizes |> Var<'T>



    ///// gets variable by name
    //static member tryFindByName (vs: Var) map =
    //    map |> Map.tryPick 
    //        (fun cvs value -> 
    //            if Var.name cvs = Var.name vs then Some value
    //            else None)

    ///// gets variable by name
    //static member findByName vs map =
    //    match Var.tryFindByName vs map with
    //    | Some value -> value
    //    | None -> raise (System.Collections.Generic.KeyNotFoundException())

