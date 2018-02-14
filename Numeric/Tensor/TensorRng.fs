namespace Tensor

open System

open Tensor.Utils



/// invalid tensor range specification
exception InvalidTensorRng of msg:string with override __.Message = __.msg

/// specified tensor index is out of range
exception IndexOutOfRange of msg:string with override __.Message = __.msg


/// Special constants that can be passed or returned instead of indices.
[<AutoOpen>]
module SpecialIdx =

    /// For slicing: inserts a new axis of size one.
    let NewAxis = Int64.MinValue + 1L

    /// For slicing: fills all remaining axes with size one. 
    /// Cannot be used together with NewAxis.
    let Fill = Int64.MinValue + 2L

    /// For reshape: remainder, so that number of elements stays constant.
    let Remainder = Int64.MinValue + 3L
    
    /// For search: value was not found.
    let NotFound = Int64.MinValue + 4L



/// Range over a dimension of a tensor.
[<StructuredFormatDisplay("{Pretty}")>]
type TensorRng = 
    /// The single element specified.
    | RngElem of int64
    /// Range of elements, including first and last.
    | Rng of first:int64 option * last:int64 option
    /// Insert broadcastable axis of size 1.
    | RngNewAxis
    /// Take all elements of remaining dimensions.
    | RngAllFill

    /// Pretty string.
    member this.Pretty =
        match this with
        | RngElem e -> sprintf "%d" e
        | Rng (Some first, Some last) -> sprintf "%d..%d" first last
        | Rng (Some first, None) -> sprintf "%d.." first 
        | Rng (None, Some last) -> sprintf "0..%d" last
        | Rng (None, None) -> "*"
        | RngNewAxis -> "NewAxis"
        | RngAllFill -> "Fill"

    /// Converts arguments to a .NET Item property or GetSlice, SetSlice method to a TensorRng list.
    static member internal ofItemOrSliceArgs (allArgs: obj[]) =
        let invalid () =
            raise (InvalidTensorRng (sprintf "specified items/slices are invalid: %A" allArgs))
        let rec toRng (args: obj list) =
            match args with            
            | [:? (TensorRng list) as rngs] ->             // direct range specification
                rngs
            | (:? (int64 option) as so) :: (:? (int64 option) as fo) :: rest -> // slice
                if so |> Option.contains NewAxis || so |> Option.contains Fill ||
                   fo |> Option.contains NewAxis || fo |> Option.contains Fill then
                    invalid ()
                Rng (so, fo) :: toRng rest
            | (:? int64 as i) :: rest when i = NewAxis ->  // new axis
                RngNewAxis :: toRng rest
            | (:? int64 as i) :: rest when i = Fill ->     // fill
                RngAllFill :: toRng rest
            | (:? int64 as i) :: rest ->                   // single item
                RngElem i  :: toRng rest
            | [] -> []
            | _  -> invalid ()
        allArgs |> Array.toList |> toRng



/// Special constants that can be passed or returned instead of tensor ranges.
[<AutoOpen>]
module SpecialRng =

    /// All elements.
    let RngAll = Rng (None, None)

