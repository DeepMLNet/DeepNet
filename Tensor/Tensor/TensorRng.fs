namespace Tensor

open System

open DeepNet.Utils
open Tensor.Utils



/// Special constants that can be passed or returned instead of indices.
[<AutoOpen>]
module internal SpecialIdx =

    /// For slicing: inserts a new axis of size one.
    let NewAxis = Int64.MinValue + 1L

    /// For slicing: fills all remaining axes with size one. 
    /// Cannot be used together with NewAxis.
    let Fill = Int64.MinValue + 2L

    /// For reshape: remainder, so that number of elements stays constant.
    let Remainder = Int64.MinValue + 3L
    
    /// For search: value was not found.
    let NotFound = Int64.MinValue + 4L


/// Shape type.
type Shape = int64 list

/// Shape constants.
module Shape =
    /// Shape of a scalar (0-dimensional tensor).
    let scalar: Shape = []


/// Range over a dimension of a tensor.
[<RequireQualifiedAccess; StructuredFormatDisplay("{Pretty}")>]
type Rng = 
    /// The single element specified.
    | Elem of int64
    /// Range of elements, including first and last.
    | Rng of first:int64 option * last:int64 option
    /// Insert broadcastable axis of size 1.
    | NewAxis
    /// Take all elements of remaining dimensions.
    | AllFill

    /// All elements.
    static member All = Rng (None, None)    

    /// Pretty string.
    member this.Pretty =
        match this with
        | Elem e -> sprintf "%d" e
        | Rng (Some first, Some last) -> sprintf "%d..%d" first last
        | Rng (Some first, None) -> sprintf "%d.." first 
        | Rng (None, Some last) -> sprintf "0..%d" last
        | Rng (None, None) -> "*"
        | NewAxis -> "NewAxis"
        | AllFill -> "Fill"

    /// Converts arguments to a .NET Item property or GetSlice, SetSlice method to a TensorRng list.
    static member internal ofItemOrSliceArgs (allArgs: obj[]) =
        let invalid () =
            invalidArg "item" "Specified items/slices are invalid: %A." allArgs
        let rec toRng (args: obj list) =
            match args with            
            | [:? (Rng list) as rngs] ->             // direct range specification
                rngs
            | (:? (int64 option) as so) :: (:? (int64 option) as fo) :: rest -> // slice
                if so |> Option.contains SpecialIdx.NewAxis || so |> Option.contains SpecialIdx.Fill ||
                   fo |> Option.contains SpecialIdx.NewAxis || fo |> Option.contains SpecialIdx.Fill then
                    invalid ()
                Rng (so, fo) :: toRng rest
            | (:? int64 as i) :: rest when i = SpecialIdx.NewAxis ->  // new axis
                NewAxis :: toRng rest
            | (:? int64 as i) :: rest when i = SpecialIdx.Fill ->     // fill
                AllFill :: toRng rest
            | (:? int64 as i) :: rest ->                   // single item
                Elem i  :: toRng rest
            | [] -> []
            | _  -> invalid ()
        allArgs |> Array.toList |> toRng




/// Memory ordering of a tensor.
type TensorOrder =
    /// Row-major (C) memory order.
    | RowMajor
    /// Column-major (Fortran) memory order.
    | ColumnMajor
    /// The specified custom memory ordering of dimensions.
    | CustomOrder of int list



/// Upper or lower trianguler part of a matrix.
[<RequireQualifiedAccess>]
type MatrixPart =
    /// Upper triangular part of the matrix.
    | Upper
    /// Lower triangular part of the matrix.
    | Lower
    