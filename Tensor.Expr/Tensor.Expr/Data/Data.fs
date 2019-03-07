namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Backend


/// Data name.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type DataName = DataName of ContextPath with

    /// Pretty string.
    member this.Pretty =
        let (DataName cp) = this
        cp.Pretty

    /// Unique variable path string.
    member this.Str =
        let (DataName cp) = this
        cp.Str  



/// Instantiated data (non-generic type).
[<StructuredFormatDisplay("{Pretty}")>]
type Data = {
    /// Name
    Name:        DataName
    /// Data type.
    TypeName:    TypeName
    /// Storage device.
    Dev:         ITensorDevice
    /// Shape.
    Shape:       ShapeSpec    
    /// Data value (mutable reference wrapped in ordered reference)
    ValueOrdRef: OrdRef<ITensor option ref>
    /// Initialization function.
    Init:        OrdRef<ITensor -> unit> option
} with
    
    static member from (name, value: ITensor, ?init): Data = {
        Name = name
        TypeName = TypeName.ofTypeInst value.DataType
        Dev = value.Dev
        Shape = value.Shape |> List.map SizeSpec.fix
        ValueOrdRef = OrdRef (ref (Some value))
        Init = init |> Option.map OrdRef
    }
        
    static member from (name, dataType, dev, shape, ?init) = {
        Name = name
        TypeName = TypeName.ofTypeInst dataType
        Dev = dev
        Shape = shape
        ValueOrdRef = OrdRef (ref None)
        Init = init |> Option.map OrdRef
    }

    /// The reference to the value.
    /// Changing it will affect all instances that share the value reference.
    member this.ValueRef = this.ValueOrdRef.Value

    /// The value, which can either be instantiated or uninstantiated.
    member this.Value = !this.ValueRef

    /// The instantiated value.
    /// Throws an exception, if data is not instantiated.
    member this.InstValue =
        match this.Value with
        | Some v -> v
        | _ -> failwithf "Data %A is not instantiated." this.Name

    /// Checks that instantiated value is consistent with respect to type, device and shape.
    member this.Check () =
        match this.Value with
        | Some value ->
            if value.DataType <> this.TypeName.Type then
                failwithf "Value of data %A with type %A has inconsistent type %A."
                    this.Name this.TypeName.Type value.DataType
            if value.Dev <> this.Dev then
                failwithf "Value of data %A with storage device %A has inconsistent storage device %A."
                    this.Name this.Dev value.Dev
            match ShapeSpec.tryEval this.Shape with
            | Some shape when shape <> value.Shape ->
                failwithf "Value of data %A with shape %A has inconsistent shape %A."
                    this.Name shape value.Shape
            | _ -> ()
        | None -> ()

    /// Data type.
    member this.DataType = this.TypeName.Type

    /// Pretty string.
    member this.Pretty =
        let sep =
            match this.Value with
            | Some _ -> "@"
            | None -> "@@"
        sprintf "%A<%A>%A%s%A" this.Name this.DataType this.Shape sep this.Dev

    /// Instantiates and initializes the data for all instance that share the same value reference.
    /// If the value is already instantiated, it is replaced with a new instance.
    /// Throws an exception, if the symbolic shape cannot be evaluated.
    static member inst (data: Data) =
        let shp = 
            match ShapeSpec.tryEval data.Shape with
            | Some shp -> shp
            | None -> 
                failwithf "Cannot instantiate data %A because its shape %A cannot be evaluated."
                    data.Name data.Shape
        let t = Tensor.NewOfType (shp, data.TypeName.Type, data.Dev)
        data.ValueRef := Some t
        Data.init data

    /// Instantiates and initializes the data for all instance that share the same value reference,
    /// if the value has not already been instantiated.
    static member instOnce (data: Data) =
        match data.Value with
        | Some _ -> ()
        | None -> Data.inst data

    /// Initializes the data using the provided initialization function.
    /// If no function was provided, it is initialized to zero.
    static member init (data: Data) =
        match data.Init with
        | Some init -> init.Value data.InstValue
        | None -> data.InstValue.FillZero()

    static member substSymSizes env (data: Data) =
        {data with Shape=data.Shape |> ShapeSpec.substSymbols env}

    static member canEvalAllSymSizes (data: Data) =
        ShapeSpec.canEval data.Shape

 

/// Instantiated data.
[<StructuredFormatDisplay("{Pretty}")>]
type Data<'T> (data: Data) =
    do
        if typeof<'T> <> data.DataType then
            failwithf "Cannot use Data<%A> for data of data type %A."
                      typeof<'T> data.DataType
                      
    /// Non-generic data.
    member this.Untyped = data

   /// The value, which can either be instantiated or uninstantiated.
    member this.Value = 
        this.Untyped.Value |> Option.map (fun v -> v :?> Tensor<'T>)

    /// The instantiated value.
    /// Throws an exception, if data is not instantiated.
    member this.InstValue =
        this.Untyped.InstValue :?> Tensor<'T>
        
    /// Data type.
    member this.DataType = this.Untyped.DataType

    /// Symbolic shape.
    member this.Shape = this.Untyped.Shape

    /// Storage device.
    member this.Dev = this.Untyped.Dev

    /// Create instantiated data using name, value and optional initializer.
    new (name, value: Tensor<'T>, ?init: Tensor<'T> -> unit) =
        let init = 
            init 
            |> Option.map (fun init -> (fun (t: ITensor) -> init (t :?> Tensor<'T>)))
        Data<'T> (Data.from (name, value, ?init=init))

    /// Create instantiated data using context and value.
    /// The data storage location must match the context device.
    new (ctx: Context, value: Tensor<'T>, ?init) =
        if ctx.Dev <> value.Dev then
            failwithf "Context has device %A, but provided value is stored on device %A."
                      ctx.Dev value.Dev
        Data<'T> (DataName ctx.Path, value, ?init=init)

    /// Create uninstantiated data.
    new (name, dev, shape, ?init) =
        let init = 
            init 
            |> Option.map (fun init -> (fun (t: ITensor) -> init (t :?> Tensor<'T>)))
        Data<'T> (Data.from (name, typeof<'T>, dev, shape, ?init=init))

    /// Create uninstantiated data.
    new (ctx: Context, shape: ShapeSpec, ?init) =
        Data<'T> (DataName ctx.Path, ctx.Dev, shape, ?init=init)

    /// Creates instantiated and initialized data of the specified shape.
    new (ctx: Context, shape: int64 list, ?init) =
        let value = Tensor<'T>.zeros ctx.Dev shape
        match init with
        | Some init -> init value
        | None -> ()
        Data<'T> (ctx, value, ?init=init)

    interface System.IEquatable<Data<'T>> with
        member this.Equals other = this.Untyped = other.Untyped

    override this.Equals other =
        match other with
        | :? Data<'T> as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<Data<'T>> with
        member this.CompareTo other = compare this.Untyped other.Untyped

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Data<'T> as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare %A to type %A." (this.GetType()) (other.GetType())

    override this.GetHashCode() = hash this.Untyped
        
    member this.Pretty = this.Untyped.Pretty

    static member inst (data: Data<'T>) = Data.inst data.Untyped
    static member instOnce (data: Data<'T>) = Data.instOnce data.Untyped
    static member init (data: Data<'T>) = Data.init data.Untyped

