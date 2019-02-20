namespace SymTensor

open System.Runtime.InteropServices


/// Assembly qualified name of a .NET type.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type TypeName = TypeName of string with 

    /// gets the System.Type associated by this TypeName        
    member this.Type = 
        let (TypeName tn) = this
        System.Type.GetType(tn)

    /// gets the size of the represented type in bytes
    member this.Size = Marshal.SizeOf this.Type

    /// pretty string
    member this.Pretty = sprintf "%s" this.Type.Name
    
    /// gets the System.Type associated by this TypeName
    static member getType (tn: TypeName) = tn.Type

    /// gets the size of the represented type in bytes
    static member size (tn: TypeName) = tn.Size

    /// gets the size of the represented type in bytes as int64
    static member size64 (tn: TypeName) = tn.Size |> int64

    /// gets the TypeName associated with the given System.Type object
    static member ofTypeInst (t: System.Type) = TypeName t.AssemblyQualifiedName

    /// gets the TypeName associated with the type of then given object
    static member ofObject (o: obj) = TypeName.ofTypeInst (o.GetType())

/// Assembly qualified name of a .NET type.
module TypeName =
    /// gets the TypeName associated with the given type
    let ofType<'T> = TypeName (typeof<'T>.AssemblyQualifiedName)
