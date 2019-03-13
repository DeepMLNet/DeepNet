namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Backend


/// Variable name.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type VarName = VarName of ContextPath with

    /// The context path of this variable.
    member this.Path =
        let (VarName cp) = this
        cp

    /// Pretty string.
    member this.Pretty = this.Path.Pretty

    /// Unique variable path string.
    member this.Str = this.Path.Str



type ParameterSpec = {
    Init:   OrdRef<ITensor -> unit>
}


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
    Shape:     Shape
    /// parameter
    Par:       ParameterSpec option
} with
    /// data type
    member this.DataType = TypeName.getType this.TypeName

    /// pretty string representation
    member this.Pretty = 
        let sep =
            if this.Par.IsSome then "@@"
            else "@"
        sprintf "%A<%s%s%A>%A" this.Name this.DataType.Name sep this.Dev this.Shape

    /// Create variable using name, shape, data type and storage device.
    static member make (name, dataType, dev, shape) : Var = {
        Name = name
        TypeName = TypeName.ofTypeInst dataType
        Shape = shape
        Dev = dev
        Par = None
    }

    /// Create variable using context, data type and shape.
    static member make (ctx: Context, dataType, shape) : Var = {
        Name = VarName ctx.Path
        TypeName = TypeName.ofTypeInst dataType
        Shape = shape
        Dev = ctx.Dev
        Par = None    
    }

    static member toPar (init: ITensor -> unit) (var: Var) = 
        let parSpec = {Init = OrdRef init}
        {var with Par = Some parSpec}

    /// name of variable
    static member name (vs: Var) = vs.Name

    /// shape of variable
    static member shape (vs: Var) = vs.Shape

    /// number of dimensions of variable
    static member nDims (vs: Var) = vs.Shape |> Shape.nDim

    /// type of variable
    static member dataType (vs: Var) = vs.DataType 

    /// typename of variable
    static member typeName (vs: Var) = vs.TypeName

    /// storage device of variable
    static member dev (vs: Var) = vs.Dev

    /// substitutes the size symbol environment into the variable
    static member substSymSizes symSizes (vs: Var) = 
        {vs with Shape=Shape.subst symSizes vs.Shape} 



/// Variable specification.
[<StructuredFormatDisplay("{Pretty}")>]
type Var<'T> (_var: Var) =
    do
        if typeof<'T> <> _var.DataType then
            failwithf "Cannot use Var<%A> for variable of data type %A."
                      typeof<'T> _var.DataType

    /// Non-generic variable specification.
    member this.Untyped = _var

    /// Create variable using name, shape and storage device.
    new (name, dev, shape) =
        Var<'T> (Var.make (name, typeof<'T>, dev, shape))

    /// Create variable using context and shape.
    new (ctx: Context, shape) =
        Var<'T> (Var.make (ctx, typeof<'T>, shape))

    static member toPar (init: Tensor<'T> -> unit) (var: Var<'T>) =
        let uInit (t: ITensor) = init (t :?> Tensor<'T>)
        var.Untyped |> Var.toPar uInit |> Var<'T>

    interface System.IEquatable<Var<'T>> with
        member this.Equals other = this.Untyped = other.Untyped

    override this.Equals other =
        match other with
        | :? Var<'T> as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<Var<'T>> with
        member this.CompareTo other = compare this.Untyped other.Untyped

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Var<'T> as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare %A to type %A." (this.GetType()) (other.GetType())

    override this.GetHashCode() = hash this.Untyped
        
    /// name of variable
    member this.Name = this.Untyped.Name
    /// name of variable    
    static member name (vs: Var<'T>) = vs.Name

    /// storage device of variable
    member this.Dev = this.Untyped.Dev
    /// storage device of variable    
    static member dev (vs: Var<'T>) = vs.Dev

    /// shape of variable
    member this.Shape = this.Untyped.Shape
    /// shape of variable    
    static member shape (vs: Var<'T>) = vs.Shape

    /// data type of variable
    member this.DataType = this.Untyped.DataType
    /// type of variable
    static member dataType (vs: Var<'T>) = vs.DataType

    /// pretty string representation
    member this.Pretty = this.Untyped.Pretty 

    /// number of dimensions of variable
    static member nDims vs = Var.shape vs |> Shape.nDim

    /// substitutes the size symbol environment into the variable
    static member substSymSizes symSizes (vs: Var<'T>) = 
        vs.Untyped |> Var.substSymSizes symSizes |> Var<'T>



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

