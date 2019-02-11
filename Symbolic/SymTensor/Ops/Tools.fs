namespace SymTensor.Ops

open DeepNet.Utils
open SymTensor


/// Helper functions for checking validity of expressions.
module Check =

    let sameType (exprs: BaseExpr list) =
        let types = exprs |> List.map (fun e -> e.TypeName)
        if types |> List.exists ((<>) types.Head) then
            failwithf "All arguments are expected to be of same type, but types are: %A." types

    let sameShape (exprs: BaseExpr list) =
        let shapes = exprs |> List.map (fun e -> e.Shape)
        if shapes |> List.exists ((<>) shapes.Head) then
            failwithf "All arguments are expected to be of same shape, but shapes are: %A." shapes

    let bool (exprs: BaseExpr list) =
        let types = exprs |> List.map (fun e -> e.TypeName)
        if types |> List.exists ((<>) TypeName.ofType<bool>) then
            failwithf "All arguments are expected to be of type bool, but types are: %A." types

    let axis (ax: int) (expr: BaseExpr) =
        if not (0 <= ax && ax < ShapeSpec.nDim expr.Shape) then
            failwithf "Cannot apply reduction operation over non-existant axis %d of tensor with shape %A." 
                      ax expr.Shape

    let range (range: SimpleRangesSpec) (x: BaseExpr) =
        if range.Length <> x.NDims then
            failwithf "Invalid range specification for expression of shape %A." x.Shape                
        range |> List.iter (function 
            | SimpleRangeSpec.SymStartSymEnd _ -> ()
            | SimpleRangeSpec.DynStartSymSize (s, _) -> 
                if (s :?> BaseExpr).DataType <> typeof<int64> then
                    failwithf "Dynamic range start must be of type int64.")


/// Helper functions for working with arguments of ops.
module Args =
    
    let leaf = Map.empty

    let unary x = Map ["X", x]
    let unaryX (am: Map<string, _>) = am.["X"]

    let binary x y = Map ["X", x; "Y", y]
    let binaryX (am: Map<string, _>) = am.["X"]
    let binaryY (am: Map<string, _>) = am.["Y"]

    let nary xs =
        xs |> List.indexed |> List.map (fun (i,v) -> i.ToString(), v) |> Map.ofList
    let naryXs (am: Map<string, _>) =
        let xs = 
            am 
            |> Map.toList 
            |> List.choose (fun (s,v) -> 
                s |> Int32.tryParse |> Option.map (fun i -> i,v))
            |> List.sortBy fst 
        if [0 .. xs.Length-1] <> List.map fst xs then
            failwithf "Cannot convert argument map to argument list: %A" am
        xs |> List.map snd
