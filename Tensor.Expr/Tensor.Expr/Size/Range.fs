namespace Tensor.Expr

open System
open DeepNet.Utils


/// basic range specification for one dimension
type BaseRange = Size * Size

/// basic range specification for multiple dimensions
type BaseRanges = BaseRange list

/// Functions for working with BaseRanges.
module BaseRanges =

    /// Try to evalualte a BaseRangesT to a numeric range.
    let tryEval (rng: BaseRanges) =
        let rec doEval rng =
            match rng with
            | (first, last) :: rrng ->
                match Size.tryEval first, Size.tryEval last, doEval rrng with
                | Some first, Some last, Some rrng -> Some ((first, last) :: rrng)
                | _ -> None
            | [] -> Some []
        doEval rng

    /// True if a BaseRangesT can be evaluated to a numeric range.
    let canEval (rng: BaseRanges) =
        match tryEval rng with
        | Some _ -> true
        | None -> false

    /// Evaluates a BaseRangesT to a numeric range.
    let eval (rng: BaseRanges) =
        match tryEval rng with
        | Some rng -> rng
        | None -> failwithf "cannot evaluate BaseRangesT %A to numeric range" rng

    /// checks that the BaseRanges is valid
    let check (rng: BaseRanges) =
        match tryEval rng with
        | Some rng ->
            for first, last in rng do
                if last < first then 
                    failwithf "invalid BaseRanges: %A" rng
        | None -> ()

    /// True if two BaseRanges overlap.
    /// Both BaseRanges must be evaluateble to numeric ranges.
    let overlapping (a: BaseRanges) (b: BaseRanges) =
        check a; check b
        (eval a, eval b)
        ||> List.forall2 (fun (aFirst, aLast) (bFirst, bLast) ->
            aFirst <= bFirst && bFirst <= aLast ||
            aFirst <= bLast  && bLast  <= aLast ||
            bFirst <= aFirst && aLast  <= bLast)

    /// True if any two ranges are overlapping.
    /// This has complexity O(N^2) currently.
    /// All BaseRanges must be evaluateble to numeric ranges.
    let areOverlapping (rngs: BaseRanges list) =       
        let rec testOvlp nonOvlp cands =
            match cands with
            | cand::rCands ->
                if nonOvlp |> List.exists (overlapping cand) then true
                else testOvlp (cand::nonOvlp) rCands
            | [] -> false
        testOvlp [] rngs

    /// True if the BaseRangesTs cover a tensor of the specified shape completely without overlap.
    /// All BaseRangesT and the ShapeT must be evaluatable to numeric ranges and a
    /// numeric shape respectively.
    let areCoveringWithoutOverlap (shp: Shape) (rngs: BaseRanges list) =       
        if areOverlapping rngs then false
        else
            let shpElems = 
                shp 
                |> Shape.eval 
                |> List.fold (*) 1L
            let rngElems = 
                rngs
                |> List.map eval
                |> List.map (List.fold (fun p (first, last) -> p * (last - first + 1L)) 1L)
                |> List.sum
            shpElems = rngElems


/// Dynamic element specification.
type IDynElem =
    inherit IComparable

/// symbolic/dynamic range specification for one dimension
[<RequireQualifiedAccess; StructuralComparison; StructuralEquality>]
type Range = 
    // ranges with symbolic size (length)
    | SymElem            of Size                           
    | DynElem            of IDynElem
    | SymStartSymEnd     of (Size option) * (Size option)
    | DynStartSymSize    of IDynElem * Size                    
    | NewAxis                                                   
    | AllFill                                                   
    //| RngSymStartDynEnd     of SizeSpecT * ExprT<int>              // size: dynamic
    //| RngDynStartDynEnd     of ExprT<int> * ExprT<int>             // size: dynamic
    //| RngDynStartSymEnd     of ExprT<int> * SizeSpecT              // size: dynamic
    //| RngDynStartToEnd      of ExprT<int>                          // size: dynamic

    static member All = Range.SymStartSymEnd (None, None)

// symbolic/dynamic subtensor specification
type Ranges = Range list

/// Simple range specification for one dimension.
[<RequireQualifiedAccess; StructuralComparison; StructuralEquality; StructuredFormatDisplay("{Pretty}")>]
type SimpleRange =
    | SymStartSymEnd     of Size * (Size option)
    | DynStartSymSize    of IDynElem * Size                    

    member this.Pretty =
        match this with
        | SimpleRange.SymStartSymEnd (first, Some last) -> sprintf "%A..%A" first last
        | SimpleRange.SymStartSymEnd (first, None) -> sprintf "%A.." first
        | SimpleRange.DynStartSymSize (first, size) -> sprintf "D%A..D%A+%A-1" first first size
    
    static member All = SimpleRange.SymStartSymEnd (Size.zero, None)
     
    ///// evaluate a SimpleRange to a Tensor.Rng
    //static member eval (dynEvaluator: IDynElem -> int64) (rs: SimpleRange) =
    //    match rs with
    //    | SimpleRange.SymStartSymEnd (s, fo) -> 
    //        Tensor.Rng.Rng (Some (Size.eval s), Option.map Size.eval fo)
    //    | SimpleRange.DynStartSymSize (s, elems) -> 
    //        let sv = dynEvaluator s
    //        Tensor.Rng.Rng (Some sv, Some (sv + Size.eval elems))

    /// evaluate a SimpleRange to a Tensor.Rng
    static member eval (rs: SimpleRange) =
        match rs with
        | SimpleRange.SymStartSymEnd (s, fo) -> 
            Tensor.Rng.Rng (Some (Size.eval s), Option.map Size.eval fo)
        | SimpleRange.DynStartSymSize (s, elems) -> 
            failwith "Dynamic elements must be resolved before evaluating a SimpleRange."

    static member canEvalSymbols (rs: SimpleRange) =
        match rs with
        | SimpleRange.SymStartSymEnd (s, fo) ->
            Size.canEval s && Option.forall Size.canEval fo
        | SimpleRange.DynStartSymSize (_, elems) ->
            Size.canEval elems

    static member isDynamic (rs: SimpleRange) =
        match rs with
        | SimpleRange.DynStartSymSize _ -> true
        | _ -> false

    static member toBaseRange (size: Size) (rs: SimpleRange) =
        match rs with
        | SimpleRange.SymStartSymEnd (first, Some last) -> first, last
        | SimpleRange.SymStartSymEnd (first, None) -> first, size - 1L
        | _ -> failwithf "cannot convert %A to BaseRange" rs



/// Active patterns for SimpleRange.
module SimpleRange =

   let (|Dynamic|Static|) rs =
        if SimpleRange.isDynamic rs then Dynamic else Static


/// Simple range specification for multiple dimensions.
type SimpleRanges = SimpleRange list

/// Functions for working with SimpleRanges.
module SimpleRanges =

    ///// evaluate a RangesT to a RangeT list
    //let eval dynEvaluator rs =
    //    rs |> List.map (SimpleRange.eval dynEvaluator)

    /// evaluate a RangesT to a RangeT list
    let eval rs =
        rs |> List.map SimpleRange.eval

    let isDynamic rs =
        rs |> List.exists SimpleRange.isDynamic

    let canEvalSymbols rs =
        rs |> List.forall SimpleRange.canEvalSymbols

    let (|Dynamic|Static|) rs =
        if isDynamic rs then Dynamic else Static

    let toBaseRanges (shape: Shape) rs =
        (shape, rs) ||> List.map2 SimpleRange.toBaseRange

    /// substitutes all symbols into the simplified range specification
    let subst env (srs: SimpleRanges) = 
        srs
        |> List.map (function
                     | SimpleRange.SymStartSymEnd (s, fo) -> 
                         SimpleRange.SymStartSymEnd (Size.subst env s, Option.map (Size.subst env) fo)
                     | SimpleRange.DynStartSymSize (s, elems) ->
                         SimpleRange.DynStartSymSize (s, Size.subst env elems))