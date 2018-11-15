namespace SymTensor

open DeepNet.Utils


/// scalar constant value
[<Struct; RequireQualifiedAccess; StructuralEquality; StructuralComparison>]
type Const = 
    | Int of intValue:int
    | Int64 of int64Value:int64
    | Double of doubleValue:double
    | Single of singleValue:single
    | Bool of boolValue:bool
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
        