namespace Tensor.Expr.Opt

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops
open Tensor.Expr.Opt.Tools



/// Optimizes elements expressions.
[<Optimizer>]
type ElementsOptimizer() =

    /// Get element expression specification.
    static member internal decompose (expr: UExpr) =
        match expr with 
        | UExpr.Elements (resShape, elemExpr, args) -> resShape, elemExpr, args
        | _ -> failwith "not an elements expression"


    /// Returns a list containing one element for each axis in the result of the elements expression.
    /// The element is true if the result can have different values over the corresponding axis.
    /// I.e. an axis is significant if values change depending on it.
    static member internal elementsAxesSignificancy elements =
        let resShape, elemExpr, args = ElementsOptimizer.decompose elements

        let argsBroadcasted =
            List.indexed args
            |> List.map (fun (argPos, arg) -> Elem.Arg argPos, axesBroadcasted arg)
            |> Map.ofList

        let rec sigInDim dim elemExpr =
            let dimSym = Elem.Expr.idxSymbol dim
            match elemExpr with
            | Elem.Expr.Leaf (Elem.SizeValue (ss, _)) -> ss.ContainedSyms.Contains dimSym 
            | Elem.Expr.Leaf (Elem.ArgElement ((arg, sa), _)) ->
                List.zip sa argsBroadcasted.[arg]
                |> List.exists (fun (dimSs, dimBc) ->
                    dimSs.ContainedSyms.Contains dimSym && not dimBc.IsBC)
            | Elem.Expr.Leaf _ -> false

            | Elem.Expr.Unary (Elem.Sum (_, first, last), a) ->
                first.ContainedSyms.Contains dimSym 
                || last.ContainedSyms.Contains dimSym
                || sigInDim dim a
            | Elem.Expr.Unary (Elem.KroneckerRng (ss, first, last), a) ->
                ss.ContainedSyms.Contains dimSym
                || first.ContainedSyms.Contains dimSym
                || last.ContainedSyms.Contains dimSym
                || sigInDim dim a
            | Elem.Expr.Unary (_, a) ->
                sigInDim dim a

            | Elem.Expr.Binary (Elem.IfThenElse (left, right), a, b) ->
                left.ContainedSyms.Contains dimSym
                || right.ContainedSyms.Contains dimSym
                || sigInDim dim a || sigInDim dim b
            | Elem.Expr.Binary (_, a, b) ->
                sigInDim dim a || sigInDim dim b

        let nDims = List.length resShape
        [0 .. nDims-1]
        |> Seq.map (fun dim -> sigInDim dim elemExpr)


    /// pulls summations out of elements expressions
    static member internal pullSumOutOfElements (optRec: UExpr -> UExpr) elements =
        let resShape, elemExpr, args = ElementsOptimizer.decompose elements

        let nDims = Shape.nDim resShape
        let mutable nArgs = args.Length 
        let newArg () =
            let res = nArgs
            nArgs <- nArgs + 1
            res

        let rec splitSum elemExpr =
            match elemExpr with
            | Elem.Expr.Unary (Elem.Sum (sym, first, last), summand) ->     
                // replace sum by argument access
                let typ = (Elem.Expr.typeName elemExpr).Type
                let sumArgPos = newArg ()
                let sumArgIdx =
                    [for d=0 to nDims - 1 do yield Elem.Expr.idx d]
                let sumArg = Elem.Expr.argElemWithType typ sumArgPos sumArgIdx

                // add summation dimension to the right
                let sumElems = last - first + 1L
                let sumandShape = resShape @ [sumElems]
                let sumandIdx = first + Elem.Expr.idx nDims

                // substitute summation symbol with last dimension index
                let subst = Map [sym, sumandIdx]
                let summandSubst = summand |> Elem.Expr.substSymSizes subst

                // build sumand elements expression and sum over last dimensions
                let summandExpr = UExpr.elements sumandShape summandSubst args
                let summedExpr = summandExpr |> UExpr.sumAxis nDims

                sumArg, [optRec summedExpr]

            | Elem.Expr.Leaf (op) -> 
                Elem.Expr.Leaf (op), []
            | Elem.Expr.Unary (op, a) ->
                let aSplit, aArgs = splitSum a
                Elem.Expr.Unary (op, aSplit), aArgs
            | Elem.Expr.Binary (op, a, b) ->
                let aSplit, aArgs = splitSum a
                let bSplit, bArgs = splitSum b
                Elem.Expr.Binary (op, aSplit, bSplit), aArgs @ bArgs

        let elemExprWithoutSum, sumArgs = splitSum elemExpr
        UExpr.elements resShape elemExprWithoutSum (args @ sumArgs)


    /// replaces result dimensions that compute identical values for all elements 
    /// by broadcasting
    static member internal broadcastInsignificantElementsAxes (optRec: UExpr -> UExpr) elements =
        let resShape, elemExpr, args = ElementsOptimizer.decompose elements

        // determine significancy of all axes
        let axSigs = ElementsOptimizer.elementsAxesSignificancy elements
        let insigAx = 
            Seq.indexed axSigs 
            |> Seq.tryFind (fun (ax, axSig) ->
                not axSig && resShape.[ax] <> Size.broadcastable)

        match insigAx with
        | Some (insigAx, _) ->
            //printfn "removing insignificant axis %d with shape %A of expr:\n%A"
            //    insigAx resShape.[insigAx] elemExpr

            // replace insignificant axis by axis with one broadcastable element
            let sigResShape = resShape |> Shape.set insigAx Size.broadcastable
            let sigElements = UExpr.elements sigResShape elemExpr args

            // broadcast result to original shape
            let bcElements = sigElements |> UExpr.broadcast resShape
            optRec bcElements
        | None -> elements


    /// optimizes an element expression
    static member internal optimizeElemExpr (optRec: UExpr -> UExpr) (elemExpr: Elem.Expr) =
        match elemExpr with

        // replace powers with integral exponents less than 5 with iterated multiplications
        | Elem.Expr.Binary (Elem.Power, a, Elem.Expr.Leaf (Elem.Const cs)) ->
            let rec repMul cnt arg =
                match cnt with
                | 0 -> Elem.Expr.constSpec (Const.oneOf elemExpr.Type)
                | 1 -> arg
                | _ when cnt > 0 ->
                    arg * repMul (cnt - 1) arg
                | _ when cnt < 0 ->
                    Elem.Expr.constSpec (Const.oneOf elemExpr.Type) / repMul (-cnt) arg
                | _ -> failwith "impossible"

            match cs.Value with
            | Util.Integral p when abs p < 5 -> repMul p a
            | _ -> elemExpr                

        | Elem.Expr.Leaf _ -> 
            elemExpr
        | Elem.Expr.Unary (op, a) -> 
            Elem.Expr.Unary(op, ElementsOptimizer.optimizeElemExpr optRec a)
        | Elem.Expr.Binary (op, a, b) -> 
            Elem.Expr.Binary (op, 
                ElementsOptimizer.optimizeElemExpr optRec a, 
                ElementsOptimizer.optimizeElemExpr optRec b)


    /// optimizes elements expression in an expression
    static member internal optimizeElements (optRec: UExpr -> UExpr) elements =
        let resShape, elemExpr, args = ElementsOptimizer.decompose elements
        UExpr.elements resShape (ElementsOptimizer.optimizeElemExpr optRec elemExpr) args


    interface IOptimizer with
        member __.Order = 30

    interface IUExprOptimizer with
        member __.Optimize subOpt expr =
            let optRec (expr: UExpr) =
                subOpt expr.BaseExpr |> UExpr

            match expr with
            | UExpr.Elements _ as elemExpr ->
                elemExpr
                |> ElementsOptimizer.optimizeElements optRec
                |> ElementsOptimizer.pullSumOutOfElements optRec
                |> ElementsOptimizer.broadcastInsignificantElementsAxes optRec

            | _ -> expr

