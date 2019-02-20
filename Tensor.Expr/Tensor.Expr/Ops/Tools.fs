namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor.Expr


/// Helper functions for checking validity of expressions.
module Check =

    let sameType (exprs: BaseExprCh list) =
        let types = exprs |> List.map (fun e -> e.TypeName)
        if types |> List.exists ((<>) types.Head) then
            failwithf "All arguments are expected to be of same type, but types are: %A." types

    let sameShape (exprs: BaseExprCh list) =
        let shapes = exprs |> List.map (fun e -> e.Shape)
        if shapes |> List.exists ((<>) shapes.Head) then
            failwithf "All arguments are expected to be of same shape, but shapes are: %A." shapes

    let bool (exprs: BaseExprCh list) =
        let types = exprs |> List.map (fun e -> e.TypeName)
        if types |> List.exists ((<>) TypeName.ofType<bool>) then
            failwithf "All arguments are expected to be of type bool, but types are: %A." types

    let axis (ax: int) (expr: BaseExprCh) =
        if not (0 <= ax && ax < ShapeSpec.nDim expr.Shape) then
            failwithf "Cannot apply reduction operation over non-existant axis %d of tensor with shape %A." 
                      ax expr.Shape

    let range (range: SimpleRangesSpec) (x: BaseExprCh) =
        if range.Length <> x.NDims then
            failwithf "Invalid range specification for expression of shape %A." x.Shape                
        range |> List.iter (function 
            | SimpleRangeSpec.SymStartSymEnd _ -> ()
            | SimpleRangeSpec.DynStartSymSize (s, _) -> 
                if (s :?> BaseExpr).OnlyCh.DataType <> typeof<int64> then
                    failwithf "Dynamic range start must be of type int64.")


[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Ch =
    let only x = Map [Ch.Default, x]

    let onlyOne = Set [Ch.Default]


/// Helper functions for working with argument values (of any type) of ops.
module ArgValue =
   
    let leaf = Map.empty

    let unary x = Map [Arg.Only, x]
    let unaryX (am: Map<_, _>) = am.[Arg.Only] 

    let binary x y = Map [Arg.X, x; 
                          Arg.Y, y]
    let binaryX (am: Map<_, _>) = am.[Arg.X]
    let binaryY (am: Map<_, _>) = am.[Arg.Y]

    let nary xs =
        xs 
        |> Seq.indexed 
        |> Seq.map (fun (i,v) -> Arg.N i, v) 
        |> Map.ofSeq
    let naryXs (am: Map<Arg, _>) =
        let xs = 
            am 
            |> Map.toList 
            |> List.choose (fun (arg, v) -> 
                match arg with 
                | Arg.N i -> Some (i, v)
                | _ -> None)
            |> List.sortBy fst 
        if [0 .. xs.Length-1] <> List.map fst xs then
            failwithf "Cannot convert argument map to argument list: %A" am
        xs |> List.map snd

    let naryOpt (list: 'a option list) =
        list 
        |> List.indexed 
        |> List.choose (function 
                        | i, Some v -> Some (Arg.N i, v)
                        | _, None -> None)
        |> Map.ofList

    let naryOptXs length args =
        [0 .. length-1]
        |> List.map (fun i -> args |> Map.tryFind (Arg.N i))


/// Helper functions for working with arguments of ops.
module Args =

    let leaf: Map<Arg, BaseExprCh> = ArgValue.leaf 

    let unary x : Map<Arg, BaseExprCh>  = ArgValue.unary x
    let unaryX (am: Map<Arg, BaseExprCh> ) = am |> ArgValue.unaryX 

    let binary x y : Map<Arg, BaseExprCh>  = ArgValue.binary x y
    let binaryX (am: Map<Arg, BaseExprCh> ) = am |> ArgValue.binaryX 
    let binaryY (am: Map<Arg, BaseExprCh> ) = am |> ArgValue.binaryY 

    let nary xs : Map<Arg, BaseExprCh> = xs |> ArgValue.nary
    let naryXs (am: Map<Arg, BaseExprCh>) = am |> ArgValue.naryXs

    let naryOpt (list: BaseExprCh option list) = ArgValue.naryOpt list
    let naryOptXs length (args: Map<Arg, BaseExprCh>) = ArgValue.naryOptXs length args



/// Functions for converting SimpleRangesSpec from and to args.
module SimpleRangesSpecArgs =

    /// Returns a map of all dynamic elements within the specified range.
    let toArgs (srs: SimpleRangesSpec) =
        srs 
        |> List.indexed
        |> List.choose (function
                        | _, SimpleRangeSpec.SymStartSymEnd _  -> None
                        | i, SimpleRangeSpec.DynStartSymSize (s, _) -> Some (Arg.N i, s))
        |> Map.ofList

    /// Replaces all dynamic elements within the specified range using the specified replacement map.
    let replaceFromArgs (args: Map<Arg, IDynElem>) (srs: SimpleRangesSpec) : SimpleRangesSpec =
        srs
        |> List.indexed
        |> List.map (function
                     | _, (SimpleRangeSpec.SymStartSymEnd _ as r) -> r
                     | i, SimpleRangeSpec.DynStartSymSize (s, elems) -> 
                        SimpleRangeSpec.DynStartSymSize (args.[Arg.N i], elems))

    /// Replaces all dynamic elements within the specified range using the specified replacement map.
    let resolveDynElems (map: Map<Arg, SizeSpec>) (srs: SimpleRangesSpec) : SimpleRangesSpec =
        srs
        |> List.indexed
        |> List.map (function
                     | _, (SimpleRangeSpec.SymStartSymEnd _ as r) -> r
                     | i, SimpleRangeSpec.DynStartSymSize (s, elems) -> 
                        let s = map.[Arg.N i]
                        SimpleRangeSpec.SymStartSymEnd (s, Some (s + elems)))

