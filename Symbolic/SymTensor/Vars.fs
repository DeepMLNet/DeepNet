namespace SymTensor

open System.Runtime.InteropServices

open DeepNet.Utils


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


/// scalar constant value
[<RequireQualifiedAccess; StructuralEquality; StructuralComparison>]
type Const = 
    | Int of int
    | Int64 of int64
    | Double of double
    | Single of single
    | Bool of bool
    with
        /// the type name of the constant
        member this.TypeName = 
            match this with
            | Const.Int _    -> TypeName.ofType<int>
            | Const.Int64 _  -> TypeName.ofType<int64>
            | Const.Double _ -> TypeName.ofType<double>
            | Const.Single _ -> TypeName.ofType<single>
            | Const.Bool _   -> TypeName.ofType<bool>

        /// the type of the constant
        member this.Type =
            this.TypeName.Type

        /// gets the value which must be of type 'T
        member this.GetValue() : 'T =
            match this with
            | Const.Int v    -> v |> box |> unbox
            | Const.Int64 v  -> v |> box |> unbox
            | Const.Double v -> v |> box |> unbox
            | Const.Single v -> v |> box |> unbox
            | Const.Bool v   -> v |> box |> unbox  
            
        /// the value as object
        member this.Value =
            match this with
            | Const.Int v    -> v |> box 
            | Const.Int64 v  -> v |> box 
            | Const.Double v -> v |> box
            | Const.Single v -> v |> box 
            | Const.Bool v   -> v |> box 
                
        /// gets the value converting it to type 'T
        member this.GetConvertedValue<'T>() : 'T =   
            this.Value |> conv<'T>          
    
        /// creates a Const from a scalar value
        static member ofValue (value: obj) =
            match value.GetType() with
            | t when t = typeof<int> -> Const.Int (value |> unbox)
            | t when t = typeof<int64> -> Const.Int64 (value |> unbox)
            | t when t = typeof<double> -> Const.Double (value |> unbox)
            | t when t = typeof<single> -> Const.Single (value |> unbox)
            | t when t = typeof<bool> -> Const.Bool (value |> unbox)
            | t -> failwithf "unsupported constant type: %A" t

        /// gets the value 
        static member value (cs: Const) =
            cs.GetValue ()

        /// the type name of the constant
        static member typeName (cs: Const) =
            cs.TypeName

        /// the type of the constant
        static member typ (cs: Const) =
            cs.Type

        /// one of specified type
        static member one (typ: System.Type) =
            1 |> convTo typ |> Const.ofValue

        /// two of specified type
        static member two (typ: System.Type) =
            1 |> convTo typ |> Const.ofValue

        /// zero constant of specified type
        static member zero typ =
            match typ with
            | _ when typ = typeof<int>    -> Const.Int 0
            | _ when typ = typeof<int64>  -> Const.Int64 0L
            | _ when typ = typeof<double> -> Const.Double 0.0
            | _ when typ = typeof<single> -> Const.Single 0.0f
            | _ when typ = typeof<bool>   -> Const.Bool false
            | _ -> failwithf "unsupported type %A" typ

        /// minimum value constant of specified type
        static member minValue typ =
            match typ with
            | _ when typ = typeof<int>    -> Const.Int (System.Int32.MinValue)
            | _ when typ = typeof<int64>  -> Const.Int64 (System.Int64.MinValue)
            | _ when typ = typeof<double> -> Const.Double (System.Double.MinValue)
            | _ when typ = typeof<single> -> Const.Single (System.Single.MinValue)
            | _ when typ = typeof<bool>   -> Const.Bool false
            | _ -> failwithf "unsupported type %A" typ

        /// maximum value constant of specified type
        static member maxValue typ =
            match typ with
            | _ when typ = typeof<int>    -> Const.Int (System.Int32.MaxValue)
            | _ when typ = typeof<int64>  -> Const.Int64 (System.Int64.MaxValue)
            | _ when typ = typeof<double> -> Const.Double (System.Double.MaxValue)
            | _ when typ = typeof<single> -> Const.Single (System.Single.MaxValue)
            | _ when typ = typeof<bool>   -> Const.Bool true
            | _ -> failwithf "unsupported type %A" typ

/// Active patterns for Const.         
module Const = 

    /// matches a zero constant of any type
    let (|Zero|_|) cs =
        match cs with
        | Const.Int 0
        | Const.Int64 0L
        | Const.Single 0.0f
        | Const.Double 0.0 -> Some ()
        | _ -> None

    /// true if constant is zero
    let isZero cs =
        match cs with
        | Zero -> true
        | _ -> false
        

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

