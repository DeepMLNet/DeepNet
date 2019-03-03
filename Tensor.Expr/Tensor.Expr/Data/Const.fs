namespace Tensor.Expr


open DeepNet.Utils
open Tensor


/// Scalar constant value of any type with support for comparison and equality.
/// The data type must implement the System.IComparable interface.
type Const (value: obj) = 

    let _value =
        match value with
        | :? System.IComparable as value -> value
        | _ -> 
            failwithf "Constant values must implement the System.IComparable interface, \
                       but the type %A does not." (value.GetType())

    let _typeName = TypeName.ofObject value

    /// the value 
    member this.Value = value
    
    /// gets the value 
    static member value (cs: Const) = cs.Value

    /// gets the value which must be of type 'T
    member this.GetValue() : 'T = this.Value |> unbox

    /// the type name of the constant
    member this.TypeName = _typeName

    /// the type name of the constant
    static member typeName (cs: Const) = cs.TypeName

    /// the type of the constant
    member this.DataType = value.GetType()

    /// the type of the constant
    static member dataType (cs: Const) = cs.DataType

    interface System.IComparable<Const> with
        member this.CompareTo other =
            if this.TypeName <> other.TypeName then
                compare this.TypeName other.TypeName
            else
                _value.CompareTo other.Value

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Const as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare Const to %A." (other.GetType())

    interface System.IEquatable<Const> with
        member this.Equals other = 
            this.TypeName = other.TypeName && this.Value.Equals other.Value

    override this.Equals other =
        match other with
        | :? Const as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    override this.GetHashCode() = value.GetHashCode()

    override this.ToString() = value.ToString()                   

    /// the value as an scalar ITensor stored on the specified device
    static member asITensor dev (cs: Const) : ITensor =
        ITensor.scalar dev cs.Value


/// Scalar constant value with support for comparison and equality.
module Const =

    /// Constant with value zero of specified type.
    let zeroOf (typ: System.Type) =
        Const (zeroOf typ :?> System.IComparable)

    /// Matches a constant of value zero.
    let (|Zero|_|) (cs: Const) =
        if cs = zeroOf cs.DataType then Some ()
        else None

    /// True if constant is zero.
    let isZero cs =
        match cs with
        | Zero -> true
        | _ -> false

    /// Constant with value one of specified type.
    let oneOf (typ: System.Type) =
        Const (oneOf typ :?> System.IComparable)

    /// Matches a constant of value one.
    let (|One|_|) (cs: Const) =
        if cs = oneOf cs.DataType then Some ()
        else None

    /// True if constant is one.
    let isOne cs =
        match cs with
        | One -> true
        | _ -> false

    /// minimum value constant of specified type
    let minValueOf typ =
        Const (minValueOf typ :?> System.IComparable)

    /// maximum value constant of specified type
    let maxValue typ =
        Const (maxValueOf typ :?> System.IComparable)
        

/// Recognizers for Const type.
[<AutoOpen>]
module ConstRecogniziers =

    /// Matches a Const and returns its value.
    let (|Const|) (cs: Const) : obj =
        cs.Value

    /// Matches a Const of type 'T and returns its value.
    let (|ConstT|_|) (cs: Const) : 'T option =
        if cs.DataType = typeof<'T> then Some (unbox cs.Value)
        else None
