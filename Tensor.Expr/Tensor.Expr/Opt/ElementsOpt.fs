namespace Tensor.Expr.Opt

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops
open Tensor.Expr.Opt.Tools



module internal ElementOptTools =

    /// Get element expression specification.
    let decompose (expr: UExpr) =
        match expr with 
        | UExpr.Elements (resShape, elemExpr, args) -> resShape, elemExpr, args
        | _ -> failwith "not an elements expression"


    /// Returns a list containing one element for each axis in the result of the elements expression.
    /// The element is true if the result can have different values over the corresponding axis.
    /// I.e. an axis is significant if values change depending on it.
    let elementsAxesSignificancy elements =
        let resShape, elemExpr, args = decompose elements

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

open ElementOptTools


type internal CorrespondingElemOp =
    | NotCorresponding
    | Leaf of Elem.LeafOp
    | Unary of Elem.UnaryOp * UExpr
    | Binary of Elem.BinaryOp * UExpr * UExpr

    static member get op =
        match op with
        | UExpr.Scalar cs           -> Leaf (Elem.Const cs)
        | UExpr.SizeValue value     -> Leaf (Elem.SizeValue (value, TypeName.ofType<int64>))
        | UExpr.Negate x            -> Unary (Elem.Negate, x)
        | UExpr.Abs x               -> Unary (Elem.Abs, x)
        | UExpr.SignT x             -> Unary (Elem.SignT, x)
        | UExpr.Log x               -> Unary (Elem.Log, x)
        | UExpr.Log10 x             -> Unary (Elem.Log10, x)  
        | UExpr.Exp x               -> Unary (Elem.Exp, x)
        | UExpr.Sin x               -> Unary (Elem.Sin, x) 
        | UExpr.Cos x               -> Unary (Elem.Cos, x) 
        | UExpr.Tan x               -> Unary (Elem.Tan, x)  
        | UExpr.Asin x              -> Unary (Elem.Asin, x)
        | UExpr.Acos x              -> Unary (Elem.Acos, x)
        | UExpr.Atan x              -> Unary (Elem.Atan, x)
        | UExpr.Sinh x              -> Unary (Elem.Sinh, x)
        | UExpr.Cosh x              -> Unary (Elem.Cosh, x)
        | UExpr.Tanh x              -> Unary (Elem.Tanh, x)
        | UExpr.Sqrt x              -> Unary (Elem.Sqrt, x) 
        | UExpr.Ceiling x           -> Unary (Elem.Ceil, x)
        | UExpr.Floor x             -> Unary (Elem.Floor, x)
        | UExpr.Round x             -> Unary (Elem.Round, x)
        | UExpr.Truncate x          -> Unary (Elem.Truncate, x)
        | UExpr.Add (x, y)          -> Binary (Elem.Add, x, y)
        | UExpr.Subtract (x, y)     -> Binary (Elem.Substract, x, y)
        | UExpr.Multiply (x, y)     -> Binary (Elem.Multiply, x, y)
        | UExpr.Divide (x, y)       -> Binary (Elem.Divide, x, y)
        | UExpr.Modulo (x, y)       -> Binary (Elem.Modulo, x, y)
        | UExpr.Pow (x, y)          -> Binary (Elem.Power, x, y)
        | _                         -> NotCorresponding


/// Converts element-wise expressions to element expressions.
[<Optimizer>]
type TransformToElements () =

    let combineIntoElements (exprInfo: BaseExprGroup) (expr: UExpr) : UExpr =

        /// Gets the element expression for the argument, or starts a
        /// new element expression if the argument is not an element expression.
        let rec getArgElemExpr (argExpr: UExpr) =            
            let combinable () = 
                Seq.length (exprInfo.Dependants argExpr.BaseExprCh) = 1 ||
                Set.count argExpr.Vars = 0            

            let rec insertBcAxes substSize substStartDim srcShp rsShp elemExpr =
                match srcShp, rsShp with
                | [], [] -> elemExpr
                | _, rsSize::remRsShp when rsSize = substSize ->
                    let dimRng = [substStartDim .. argExpr.NDims-1]
                    let rplSym d = sprintf "__RPL%d__" d |> SizeSym
                    let insSubst1 =
                        dimRng
                        |> List.map (fun d -> Elem.Expr.idxSymbol d, Size.Atom (SizeAtom.Sym (rplSym d)))
                        |> Map.ofList
                    let insSubst2 =
                        dimRng
                        |> List.map (fun d -> rplSym d, Elem.Expr.idx (d+1))
                        |> Map.ofList
                    let substExpr = 
                        elemExpr |> Elem.Expr.substSymSizes insSubst1 |> Elem.Expr.substSymSizes insSubst2
                    insertBcAxes substSize (substStartDim+1) srcShp remRsShp substExpr
                | srcSize::remSrcShp, rsSize::remRsShp when srcSize = rsSize ->
                    insertBcAxes substSize (substStartDim+1) remSrcShp remRsShp elemExpr
                | _ -> failwith "invalid reshape for broadcast axes insertion"

            match argExpr with
            | UExpr.Elements (_, argElemExpr, argArgs) when combinable() -> argElemExpr, argArgs
            | UExpr.DoBroadcast (shp, a) when combinable() ->
                // set broadcasted dimensions to zero in element expression
                let bcSubst = 
                    a.Shape
                    |> List.indexed
                    |> List.collect (fun (d, ss) ->
                        if ss = Size.broadcastable then [Elem.Expr.idxSymbol d, Size.zero]
                        else [])
                    |> Map.ofSeq
                let bcElemExpr, bcArgs = getArgElemExpr a
                bcElemExpr |> Elem.Expr.substSymSizes bcSubst, bcArgs
            | UExpr.Reshape (rsShp, src) when combinable() &&
                    (rsShp |> List.withoutValue Size.broadcastable) = src.Shape ->
                // replace insertion of broadcast axes using Reshape op by insertion of
                // axes into element expression
                let rsElemExpr, rsArgs = getArgElemExpr src
                insertBcAxes Size.broadcastable 0 src.Shape rsShp rsElemExpr, rsArgs
            | UExpr.Reshape (rsShp, src) when combinable() && 
                    (rsShp |> List.withoutValue Size.one) = src.Shape ->
                // replace insertion of singleton axes using Reshape op by insertion of
                // axes into element expression
                let rsElemExpr, rsArgs = getArgElemExpr src
                insertBcAxes Size.one 0 src.Shape rsShp rsElemExpr, rsArgs
            | combArgExpr -> 
                let idxs = [0 .. combArgExpr.NDims-1] |> List.map Elem.Expr.idx
                Elem.Expr.argElemWithType combArgExpr.DataType 0 idxs, [combArgExpr]  

        /// Joins the arguments of two element expressions and adjusts them accordingly.
        let joinArgsOfElemExprs (aElemExpr, aArgs) (bElemExpr, bArgs) =
            let rec adjust expr =
                match expr with
                | Elem.Expr.Leaf (Elem.ArgElement ((Elem.Arg arg, idx), tn)) ->
                    Elem.Expr.Leaf (Elem.ArgElement ((Elem.Arg (arg + List.length aArgs), idx), tn))
                | Elem.Expr.Leaf _ -> expr
                | Elem.Expr.Unary (op, a) -> Elem.Expr.Unary (op, adjust a)
                | Elem.Expr.Binary (op, a, b) -> Elem.Expr.Binary (op, adjust a, adjust b)
            aElemExpr, adjust bElemExpr, aArgs @ bArgs


        match CorrespondingElemOp.get expr with
        | Leaf elemOp ->
            UExpr.elements expr.Shape (Elem.Expr.Leaf elemOp) []
        
        | Unary (elemOp, aExpr) ->     
            let aElemExpr, aArgs = getArgElemExpr aExpr        
            let elemExpr = Elem.Expr.Unary (elemOp, aElemExpr)
            UExpr.elements expr.Shape elemExpr aArgs
                    
        | Binary (elemOp, aExpr, bExpr) ->
            let aElemExpr, aArgs = getArgElemExpr aExpr   
            let bElemExpr, bArgs = getArgElemExpr bExpr   
            let aElemExpr, bElemExpr, abArgs = 
                joinArgsOfElemExprs (aElemExpr, aArgs) (bElemExpr, bArgs)
            let elemExpr = Elem.Expr.Binary (elemOp, aElemExpr, bElemExpr) 
            UExpr.elements expr.Shape elemExpr abArgs

        //| Nary (Elements (_, elemExpr), args) ->
            // TODO: if we are an Elem.Expr, merge with children
            //printf "could combine two elemexprs"
            //expr

        | NotCorresponding ->
            expr


    interface IOptimizer with
        member __.Order = 30
        member __.Optimize opt expr = apply opt expr __.Optimize

    member __.Optimize data expr =
        combineIntoElements data.ExprGroup expr




/// Optimizes elements expressions.
[<Optimizer>]
type ElementsOptimizer() =

    /// optimizes an element expression
    static member internal optimizeElemExpr (elemExpr: Elem.Expr) =
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
            Elem.Expr.Unary(op, ElementsOptimizer.optimizeElemExpr a)
        | Elem.Expr.Binary (op, a, b) -> 
            Elem.Expr.Binary (op, 
                ElementsOptimizer.optimizeElemExpr a, 
                ElementsOptimizer.optimizeElemExpr b)

    interface IOptimizer with
        member __.Order = 31
        member __.Optimize opt expr = apply opt expr __.Optimize

    member __.Optimize opt expr =
        match expr with
        | UExpr.Elements (resShape, elemExpr, args) ->
            UExpr.elements resShape (ElementsOptimizer.optimizeElemExpr elemExpr) args
        | _ -> expr



/// Pulls summation out of element expressions.
[<Optimizer>]
type PullSumOutOfElements () =
    interface IOptimizer with
        member __.Order = 32
        member __.Optimize opt expr = apply opt expr __.Optimize

    member __.Optimize opt expr =
        match expr with
        | UExpr.Elements (resShape, elemExpr, args) ->
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

                    sumArg, [subOpt opt summedExpr]

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

        | _ -> expr



/// Replaces result dimensions that compute identical values for all elements by broadcasting.
[<Optimizer>]
type BroadcastInsignificantElementsAxes () =
    interface IOptimizer with
        member __.Order = 33
        member __.Optimize opt expr = apply opt expr __.Optimize

    member __.Optimize opt expr =
        match expr with
        | UExpr.Elements (resShape, elemExpr, args) as elements ->
            // determine significancy of all axes
            let axSigs = elementsAxesSignificancy elements
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
                subOpt opt bcElements
            | None -> elements
            
        | _ -> expr


