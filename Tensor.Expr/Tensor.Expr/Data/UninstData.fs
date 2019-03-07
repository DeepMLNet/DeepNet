namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Backend



/// Uninstantiated data. (non-generic type)      
type UninstData = {
    /// Name
    Name:       DataName
    /// data type
    TypeName:   TypeName
    /// storage device
    Dev:        ITensorDevice
    /// shape
    Shape:      ShapeSpec
    /// initialization function
    Init:       OrdRef<ITensor -> unit> option
} with

    member this.Pretty =
        sprintf "%A<%A>%A@@%A" this.Name this.TypeName.Type this.Shape this.Dev

    member this.DataType = this.TypeName.Type

    /// Instantiates and initializes the data.
    static member inst (ud: UninstData) =
        let shp = 
            match ShapeSpec.tryEval ud.Shape with
            | Some shp -> shp
            | None -> 
                failwithf "Cannot instantiate data %A because its shape %A cannot be evaluated."
                          ud.Name ud.Shape
        let data: Data = {
            Name = ud.Name
            Init = ud.Init
            ValueRef = OrdRef (Tensor.NewOfType (shp, ud.TypeName.Type, ud.Dev))
        }
        Data.init data
        data
            


/// Instantiated data.
[<StructuredFormatDisplay("Pretty")>]
type UninstData<'T> (data: UninstData) =
    do
        if typeof<'T> <> data.DataType then
            failwithf "Cannot use UninstData<%A> for data of data type %A."
                      typeof<'T> data.DataType
                      
    /// Non-generic variable specification.
    member this.Untyped = data

    member this.TypeName = this.Untyped.TypeName
    member this.DataType = this.Untyped.DataType
    member this.Dev = this.Untyped.Dev
    member this.Shape = this.Untyped.Shape

    /// Create data using name and value.
    new (name, dev, shape, ?init: (Tensor<'T> -> unit)) =
        UninstData<'T> {
            Name = name
            TypeName = TypeName.ofType<'T>
            Dev = dev
            Shape = shape
            Init = 
                init 
                |> Option.map (fun init -> 
                    OrdRef (fun (t: ITensor) -> init (t :?> Tensor<'T>)))
        }

    /// Create data using context and value.
    /// The data storage location must match the context device.
    new (ctx: Context, shape, ?init: (Tensor<'T> -> unit)) =
        UninstData<'T> (DataName ctx.Path, ctx.Dev, shape, ?init=init)

    interface System.IEquatable<UninstData<'T>> with
        member this.Equals other = this.Untyped = other.Untyped

    override this.Equals other =
        match other with
        | :? UninstData<'T> as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<UninstData<'T>> with
        member this.CompareTo other = compare this.Untyped other.Untyped

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? UninstData<'T> as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare %A to type %A." (this.GetType()) (other.GetType())

    override this.GetHashCode() = hash this.Untyped
        
    member this.Pretty = this.Untyped.Pretty

    /// Instantiates and initializes the data.
    static member inst (ud: UninstData<'T>) =
        UninstData.inst ud.Untyped |> Data<'T>

