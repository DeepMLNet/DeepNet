namespace SymTensor

open ShapeSpec


[<AutoOpen>]
module VarSpecTypes =

    /// assembly qualified name of a .NET type
    type TypeNameT = TypeName of string

    /// non-generic variable specification interface
    type IVarSpec =
        inherit System.IComparable
        abstract member Name : string 
        abstract member Shape: ShapeSpecT
        abstract member Type: System.Type
        abstract member TypeName: TypeNameT

    /// variable specification: has a name, type and shape specificaiton
    [<StructuredFormatDisplay("Var \"{Name}\"")>]
    type VarSpecT<'T> = 
        {Name: string; Shape: ShapeSpecT;}
        
        interface IVarSpec with
            member this.Name = this.Name
            member this.Shape = this.Shape
            member this.Type = typeof<'T>
            member this.TypeName = TypeName (typeof<'T>.AssemblyQualifiedName)


module TypeName =

    /// gets the System.Type associated by this TypeName
    let getType (TypeName tn) =
        System.Type.GetType(tn)



module VarSpec =

    /// create variable specifation by name and shape
    let inline ofNameAndShape name shape =
        {Name=name; Shape=shape;}

    /// name of variable
    let name (vs: IVarSpec) = vs.Name

    /// shape of variable
    let shape (vs: IVarSpec) = vs.Shape

