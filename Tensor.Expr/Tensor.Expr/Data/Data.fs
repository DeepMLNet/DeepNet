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
[<StructuredFormatDisplay("Pretty")>]
type Data = {
    /// Name
    Name:       DataName
    /// data value
    ValueRef:   OrdRef<ITensor>
    /// initialization function
    Init:       OrdRef<ITensor -> unit> option
} with
    
    /// The data tensor.
    member this.Value = this.ValueRef.Value

    member this.Pretty =
        sprintf "%A<%A>%A@%A" this.Name this.Value.DataType this.Value.Shape this.Value.Dev

    /// Initializes the data using the provided initialization function.
    /// If no function was provided, it is initialized to zero.
    static member init (data: Data) =
        match data.Init with
        | Some init -> init.Value data.Value
        | None -> data.Value.FillZero()

 

/// Instantiated data.
[<StructuredFormatDisplay("Pretty")>]
type Data<'T> (data: Data) =
    do
        if typeof<'T> <> data.Value.DataType then
            failwithf "Cannot use Data<%A> for data of data type %A."
                      typeof<'T> data.Value.DataType
                      
    /// Non-generic data.
    member this.Untyped = data

    /// The data tensor.
    member this.Value = this.Untyped.Value :?> Tensor<'T>

    /// Create data using name and value and optional initializer.
    new (name, value: Tensor<'T>, ?init: Tensor<'T> -> unit) =
        Data<'T> {
            Name = name
            Init = 
                init 
                |> Option.map (fun init -> 
                    OrdRef (fun (t: ITensor) -> init (t :?> Tensor<'T>)))
            ValueRef = OrdRef (value :> ITensor)
        }

    /// Create data using context and value.
    /// The data storage location must match the context device.
    new (ctx: Context, value: Tensor<'T>, ?init) =
        if ctx.Dev <> value.Dev then
            failwithf "Context has device %A, but provided value is stored on device %A."
                      ctx.Dev value.Dev
        Data<'T> (DataName ctx.Path, value, ?init=init)

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

    static member init (data: Data<'T>) = Data.init data.Untyped

