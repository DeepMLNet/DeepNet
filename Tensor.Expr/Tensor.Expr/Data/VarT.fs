namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Backend



/// Variable specification.
[<StructuredFormatDisplay("{Pretty}")>]
type Var<'T> (_var: Var) =
    do
        if typeof<'T> <> _var.DataType then
            failwithf "Cannot use Var<%A> for variable of data type %A."
                      typeof<'T> _var.DataType

    /// Non-generic variable specification.
    member this.Var = _var

    /// Create variable using name, shape and storage device.
    new (name, dev, shape) =
        Var<'T> ({Name=name; TypeName=TypeName.ofType<'T>; Dev=dev; Shape=shape})

    /// Create variable using name in root context, shape and storage device.
    new (name: string, dev, shape) =
        Var<'T> ({Name=VarName.from name; TypeName=TypeName.ofType<'T>; Dev=dev; Shape=shape})

    /// Create variable using context and shape.
    new (ctx: Context, shape) =
        Var<'T> ({Name=VarName.from ctx.Path; TypeName=TypeName.ofType<'T>; Dev=ctx.Dev; Shape=shape})

    interface System.IEquatable<Var<'T>> with
        member this.Equals other = this.Var = other.Var

    override this.Equals other =
        match other with
        | :? Var<'T> as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<Var<'T>> with
        member this.CompareTo other = compare this.Var other.Var

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Var<'T> as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare Var to type %A." (other.GetType())

    override this.GetHashCode() = hash this.Var
        
    /// name of variable
    member this.Name = this.Var.Name
    /// name of variable    
    static member name (vs: Var<'T>) = vs.Name

    /// storage device of variable
    member this.Dev = this.Var.Dev
    /// storage device of variable    
    static member dev (vs: Var<'T>) = vs.Dev

    /// shape of variable
    member this.Shape = this.Var.Shape
    /// shape of variable    
    static member shape (vs: Var<'T>) = vs.Shape

    /// data type of variable
    member this.DataType = this.Var.DataType
    /// type of variable
    static member dataType (vs: Var<'T>) = vs.DataType

    /// pretty string representation
    member this.Pretty = this.Var.Pretty 

    /// number of dimensions of variable
    static member nDims vs = Var.shape vs |> ShapeSpec.nDim

    /// substitutes the size symbol environment into the variable
    static member substSymSizes symSizes (vs: Var<'T>) = 
        vs.Var |> Var.substSymSizes symSizes |> Var<'T>



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

