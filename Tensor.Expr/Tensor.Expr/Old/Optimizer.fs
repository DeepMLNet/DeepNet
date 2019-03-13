namespace Tensor.Expr

open DeepNet.Utils


module Optimizer =
   
    /// Cache of optimized expressions.
    let private optimized = Dictionary<Expr, Expr> (HashIdentity.Reference) 
    let private combined = Dictionary<Expr, Expr> (HashIdentity.Reference) 

    /// Broadcast information
    type BroadcastInfoT =
        /// axis is broadcasted to specific size
        | Broadcasted of SizeSpec
        /// axis is not broadcasted and has specific size
        | NotBroadcasted of SizeSpec
        /// true if axis is broadcasted
        member this.IsBC =
            match this with
            | Broadcasted _ -> true
            | NotBroadcasted _ -> false
        /// final size
        member this.Size = 
            match this with
            | Broadcasted s | NotBroadcasted s -> s

    /// Returns a list containing one element each axis of the expression.
    /// The element is true if the axis is broadcasted.
    let axesBroadcasted expr =
        match expr with
        | Unary (DoBroadcast bc, a) ->
            List.zip bc (Expr.shapeOf a)
            |> List.map (fun (bcDim, aDim) -> 
                if aDim = Size.broadcastable && bcDim <> aDim then Broadcasted bcDim
                else NotBroadcasted aDim)
        | _ ->
            Expr.shapeOf expr
            |> List.map (fun aDim -> NotBroadcasted aDim)


    /// Returns a list containing one element for each axis in the result of the elements expression.
    /// The element is true if the result can have different values over the corresponding axis.
    /// I.e. an axis is significant if values change depending on it.
    let elementsAxesSignificancy elements =
        match elements with 
        | Nary (Elements (resShape, elemExpr), args) ->

            let argsBroadcasted =
                List.indexed args
                |> List.map (fun (argPos, arg) -> Elem.Arg argPos, axesBroadcasted arg)
                |> Map.ofList

            let rec sigInDim dim elemExpr =
                let dimSym = Elem.Expr.idxSymbol dim
                match elemExpr with
                | Elem.Expr.Leaf (Elem.SizeValue (ss, _)) -> ss.ContainsSymbol dimSym 
                | Elem.Expr.Leaf (Elem.ArgElement ((arg, sa), _)) ->
                    List.zip sa argsBroadcasted.[arg]
                    |> List.exists (fun (dimSs, dimBc) ->
                        dimSs.ContainsSymbol dimSym && not dimBc.IsBC)
                | Elem.Expr.Leaf _ -> false

                | Elem.Expr.Unary (Elem.Sum (_, first, last), a) ->
                    first.ContainsSymbol dimSym 
                    || last.ContainsSymbol dimSym
                    || sigInDim dim a
                | Elem.Expr.Unary (Elem.KroneckerRng (ss, first, last), a) ->
                    ss.ContainsSymbol dimSym
                    || first.ContainsSymbol dimSym
                    || last.ContainsSymbol dimSym
                    || sigInDim dim a
                | Elem.Expr.Unary (_, a) ->
                    sigInDim dim a

                | Elem.Expr.Binary (Elem.IfThenElse (left, right), a, b) ->
                    left.ContainsSymbol dimSym
                    || right.ContainsSymbol dimSym
                    || sigInDim dim a || sigInDim dim b
                | Elem.Expr.Binary (_, a, b) ->
                    sigInDim dim a || sigInDim dim b

            let nDims = List.length resShape
            [0 .. nDims-1]
            |> Seq.map (fun dim -> sigInDim dim elemExpr)

        | _ -> failwith "not an elements expression"

    /// pulls summations out of elements expressions
    let rec pullSumOutOfElements elements =
        match elements with
        | Nary (Elements (resShape, elemExpr), args) ->
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
                    let summandExpr = Expr.elements sumandShape summandSubst args
                    let summedExpr = summandExpr |> Expr.sumAxis nDims

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
            Expr.elements resShape elemExprWithoutSum (args @ sumArgs)

        | _ -> failwith "not an elements expression"

    /// replaces result dimensions that compute identical values for all elements 
    /// by broadcasting
    and broadcastInsignificantElementsAxes elements =
        match elements with
        | Nary (Elements (resShape, elemExpr), args) ->
            // determine significancy of all axes
            let axSigs = elementsAxesSignificancy elements
            let insigAx = 
                Seq.indexed axSigs 
                |> Seq.tryFind (fun (ax, axSig) ->
                    not axSig && resShape.[ax] .<> Size.broadcastable)

            match insigAx with
            | Some (insigAx, _) ->
                //printfn "removing insignificant axis %d with shape %A of expr:\n%A"
                //    insigAx resShape.[insigAx] elemExpr

                // replace insignificant axis by axis with one broadcastable element
                let sigResShape = resShape |> Shape.set insigAx Size.broadcastable
                let sigElements = Expr.elements sigResShape elemExpr args

                // broadcast result to original shape
                let bcElements = sigElements |> Expr.broadcast resShape
                optRec bcElements
            | None -> elements

        | _ -> failwith "not an elements expression"

    /// optimizes an element expression
    and optimizeElemExpr (elemExpr: Elem.Expr) =
        match elemExpr with

        | Elem.Expr.Binary (Elem.Power, a, Elem.Expr.Leaf (Elem.Const cs)) ->
            // replace powers with integral exponents less than 5 with iterated multiplications
            let p = 
                match cs with
                | Const.Int x -> Some x
                | Const.Double (Util.Integral x) -> Some x
                | Const.Single (Util.Integral x) -> Some x
                | _ -> None

            let rec repMul cnt arg =
                match cnt with
                | 0 -> Elem.Expr.constSpec (Const.one elemExpr.Type)
                | 1 -> arg
                | _ when cnt > 0 ->
                    arg * repMul (cnt - 1) arg
                | _ when cnt < 0 ->
                    Elem.Expr.constSpec (Const.one elemExpr.Type) / repMul (-cnt) arg
                | _ -> failwith "impossible"

            match p with
            | Some p when abs p < 5 -> repMul p a
            | _ -> elemExpr                

        | Elem.Expr.Leaf _ -> elemExpr
        | Elem.Expr.Unary (op, a) -> Elem.Expr.Unary(op, optimizeElemExpr a)
        | Elem.Expr.Binary (op, a, b) -> Elem.Expr.Binary (op, optimizeElemExpr a, optimizeElemExpr b)

    /// optimizes elements expression in an expression
    and optimizeElements elements =
        match elements with
        | Nary (Elements (resShape, elemExpr), args) ->           
            Nary (Elements (resShape, optimizeElemExpr elemExpr), args)
        | _ -> failwith "not an elements expression"

    and leafOpToElemOp op =
        match op with
        | ScalarConst cs        -> Some (Elem.Const cs)
        | SizeValue (value, tn) -> Some (Elem.SizeValue (value, tn))
        | _                     -> None

    and unaryOpToElemOp op =
        match op with
        | Negate    -> Some Elem.Negate
        | Abs       -> Some Elem.Abs
        | SignT     -> Some Elem.SignT
        | Log       -> Some Elem.Log
        | Log10     -> Some Elem.Log10  
        | Exp       -> Some Elem.Exp
        | Sin       -> Some Elem.Sin 
        | Cos       -> Some Elem.Cos 
        | Tan       -> Some Elem.Tan  
        | Asin      -> Some Elem.Asin
        | Acos      -> Some Elem.Acos
        | Atan      -> Some Elem.Atan
        | Sinh      -> Some Elem.Sinh
        | Cosh      -> Some Elem.Cosh
        | Tanh      -> Some Elem.Tanh
        | Sqrt      -> Some Elem.Sqrt 
        | Ceil      -> Some Elem.Ceil
        | Floor     -> Some Elem.Floor
        | Round     -> Some Elem.Round
        | Truncate  -> Some Elem.Truncate
        | _         -> None

    and binaryOpToElemOp op =
        match op with
        | Add       -> Some Elem.Add
        | Substract -> Some Elem.Substract
        | Multiply  -> Some Elem.Multiply
        | Divide    -> Some Elem.Divide
        | Modulo    -> Some Elem.Modulo
        | Power     -> Some Elem.Power
        | _         -> None

    /// combines elemwise and elements operations into one elements operation
    and combineIntoElementsRec (exprInfo: ExprInfoT) (expr: Expr) : Expr =
        let subComb = combineIntoElementsRec exprInfo

        /// Gets the element expression for the argument, or starts a
        /// new element expression if the argument is not an element expression.
        let rec getArgElemExpr argExpr =            
            let combinable () = 
                (exprInfo.DependantsStructural argExpr).Count = 1 ||
                Set.count (Expr.extractVars argExpr) = 0            

            let rec insertBcAxes substSize substStartDim srcShp rsShp elemExpr =
                match srcShp, rsShp with
                | [], [] -> elemExpr
                | _, rsSize::remRsShp when rsSize = substSize ->
                    let dimRng = [substStartDim .. argExpr.NDims-1]
                    let rplSym d = sprintf "__RPL%d__" d |> SizeSymbol.ofName
                    let insSubst1 =
                        dimRng
                        |> List.map (fun d -> Elem.Expr.idxSymbol d, Size.Base (BaseSize.Sym (rplSym d)))
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

            match subComb argExpr with
            | Nary (Elements (_, argElemExpr), argArgs) when combinable() -> argElemExpr, argArgs
            | Unary (DoBroadcast shp, a) when combinable() ->
                // set broadcasted dimensions to zero in element expression
                let bcSubst = 
                    Expr.shapeOf a
                    |> List.indexed
                    |> List.collect (fun (d, ss) ->
                        if ss = Size.broadcastable then [Elem.Expr.idxSymbol d, Size.zero]
                        else [])
                    |> Map.ofSeq
                let bcElemExpr, bcArgs = getArgElemExpr a
                bcElemExpr |> Elem.Expr.substSymSizes bcSubst, bcArgs
            | Unary (Reshape rsShp, src) when combinable() &&
                    (rsShp |> List.withoutValue Size.broadcastable) = src.Shape ->
                // replace insertion of broadcast axes using Reshape op by insertion of
                // axes into element expression
                let rsElemExpr, rsArgs = getArgElemExpr src
                insertBcAxes Size.broadcastable 0 src.Shape rsShp rsElemExpr, rsArgs
            | Unary (Reshape rsShp, src) when combinable() && 
                    (rsShp |> List.withoutValue Size.one) = src.Shape ->
                // replace insertion of singleton axes using Reshape op by insertion of
                // axes into element expression
                let rsElemExpr, rsArgs = getArgElemExpr src
                insertBcAxes Size.one 0 src.Shape rsShp rsElemExpr, rsArgs
            | combArgExpr -> 
                let idxs = [0 .. combArgExpr.NDims-1] |> List.map Elem.Expr.idx
                Elem.Expr.argElemWithType combArgExpr.Type 0 idxs, [combArgExpr]  

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

        match combined.LockedTryFind expr with
        | Some comb -> comb
        | None -> 
            let comb =
                match expr with
                | Leaf op ->
                    match leafOpToElemOp op with
                    | Some elemOp -> 
                        Expr.elements (Expr.shapeOf expr) (Elem.Expr.Leaf elemOp) []
                        |> optimizeElements
                    | None -> expr
        
                | Unary (Gather indices, a) ->
                    Unary (Gather (List.map (Option.map subComb) indices), subComb a)
                | Unary (Scatter (indices, shp), a) ->
                    Unary (Scatter (List.map (Option.map subComb) indices, shp), subComb a)
                | Unary (op, aExpr) ->
                    match unaryOpToElemOp op with
                    | Some elemOp ->       
                        let aElemExpr, aArgs = getArgElemExpr aExpr        
                        let elemExpr = Elem.Expr.Unary (elemOp, aElemExpr)
                        Expr.elements (Expr.shapeOf expr) elemExpr aArgs
                        |> optimizeElements
                    | None -> Unary (op, subComb aExpr)
                    
                | Binary (IfThenElse cond, a, b) ->
                    Binary (IfThenElse (subComb cond), subComb a, subComb b)
                | Binary (op, aExpr, bExpr) ->
                    match binaryOpToElemOp op with
                    | Some elemOp ->
                        let aElemExpr, aArgs = getArgElemExpr aExpr   
                        let bElemExpr, bArgs = getArgElemExpr bExpr   
                        let aElemExpr, bElemExpr, abArgs = 
                            joinArgsOfElemExprs (aElemExpr, aArgs) (bElemExpr, bArgs)
                        let elemExpr = Elem.Expr.Binary (elemOp, aElemExpr, bElemExpr) 
                        Expr.elements (Expr.shapeOf expr) elemExpr abArgs
                        |> optimizeElements
                    | None -> Binary (op, subComb aExpr, subComb bExpr)

                //| Nary (Elements (_, elemExpr), args) ->
                    // TODO: if we are an Elem.Expr, merge with children
                    //printf "could combine two elemexprs"
                    //expr

                | Nary (op, es) ->
                    Nary (op, es |> List.map subComb)

            combined.LockedSet (expr, comb)
            combined.LockedSet (comb, comb)
            comb

    /// Optimizes an expression.
    and private optRec (expr: Expr) : Expr =
        match optimized.LockedTryFind expr with
        | Some opt -> opt 
        | None ->
            let opt = 
                match expr with

                // remove unnecessary axes permutations
                | Unary (PermuteAxes perm, a) when Permutation.isIdentity perm ->
                    optRec a

                // remove unnecessary reshapes
                | Unary (Reshape ss, a) when Shape.equalWithBroadcastability ss (Expr.shapeOf a) ->
                    optRec a            

                // remove unnecessary broadcasts
                | Unary (DoBroadcast ss, a) when Shape.equalWithBroadcastability ss (Expr.shapeOf a) ->
                    optRec a

                // combine subsequent axes permutations
                | Unary (PermuteAxes perm1, Unary (PermuteAxes perm2, a)) ->
                    let perm = Permutation.chain perm1 perm2
                    Unary (PermuteAxes perm, a) |> optRec

                // remove unneccessary permutation of size-one axes before reshape
                | Unary (Reshape ss, Unary (PermuteAxes (Permutation.Swap (ax1, ax2)), a)) when
                        (a.Shape.[ax1] .= Size.one || a.Shape.[ax2] .= Size.one) &&
                        a.Shape.[ax1+1 .. ax2-1] |> List.forall (fun ss -> ss .= Size.one) ->
                    Unary (Reshape ss, a) |> optRec

                // combine subsequent reshapes
                | Unary (Reshape ss, Unary (Reshape _, a)) ->
                    Unary (Reshape ss, a) |> optRec

                // combine subsequent broadcasts
                | Unary (DoBroadcast bc, Unary (DoBroadcast _, a)) ->
                    Unary (DoBroadcast bc, a) |> optRec

                // remove unnecessary broadcasts after reshape
                | Unary (DoBroadcast bcShp, Unary (Reshape reShp, a)) when 
                        Shape.equalWithoutBroadcastability bcShp reShp ->
                    Unary (Reshape bcShp, a) |> optRec

                // pull permute through broadcast
                | Unary (DoBroadcast bc, Unary (PermuteAxes perm, a)) ->
                    let bcPerm = bc |> Permutation.apply (Permutation.invert perm)
                    Unary (PermuteAxes perm, Unary (DoBroadcast bcPerm, a)) |> optRec

                // pull permute, broadcast and reshape through unary elementwise ops
                | Unary (Expr.UnaryElemwiseOp as op, Unary (PermuteAxes _ as lop, a)) 
                | Unary (Expr.UnaryElemwiseOp as op, Unary (Reshape _ as lop, a)) 
                | Unary (Expr.UnaryElemwiseOp as op, Unary (DoBroadcast _ as lop, a)) ->
                    Unary (lop, Unary (op, a)) |> optRec

                // pull broadcast over batched dimensions through Diag
                | Unary ((Diag (ax1, ax2) as op), (Unary (DoBroadcast _, a) as ba))
                            when List.indexed (axesBroadcasted ba)
                                 |> List.exists (fun (d, bc) -> d <> ax1 && d <> ax2 && bc.IsBC) ->
                    let aOptBc =
                        List.indexed (axesBroadcasted ba)   
                        |> List.map (function | d, bc when d = ax1 || d = ax2 -> bc.Size
                                              | _, Broadcasted _ -> Size.broadcastable
                                              | _, NotBroadcasted s -> s)
                    let baOpt = Unary (DoBroadcast aOptBc, a) |> optRec
                    Unary (DoBroadcast (Expr.shapeOf expr), Unary (op, baOpt)) |> optRec

                // pull broadcast over batched dimensions through DiagMat 
                | Unary ((DiagMat (ax1, _) as op), (Unary (DoBroadcast _, a) as ba))
                            when List.indexed (axesBroadcasted ba)
                                 |> List.exists (fun (d, bc) -> d <> ax1 && bc.IsBC) ->
                    let aOptBc =
                        List.indexed (axesBroadcasted ba)   
                        |> List.map (function | d, bc when d = ax1 -> bc.Size
                                              | _, Broadcasted _ -> Size.broadcastable
                                              | _, NotBroadcasted s -> s)
                    let baOpt = Unary (DoBroadcast aOptBc, a) |> optRec
                    Unary (DoBroadcast (Expr.shapeOf expr), Unary (op, baOpt)) |> optRec

                // pull matching permute, broadcast and reshape through binary elementwise ops
                | Binary (Expr.BinaryElemwiseOp as op, Unary (PermuteAxes _ as lopa, a),
                                                  Unary (PermuteAxes _ as lopb, b))
                | Binary (Expr.BinaryElemwiseOp as op, Unary (Reshape _ as lopa, a),
                                                  Unary (Reshape _ as lopb, b))
                | Binary (Expr.BinaryElemwiseOp as op, Unary (DoBroadcast _ as lopa, a),
                                                  Unary (DoBroadcast _ as lopb, b))
                            when lopa = lopb && Expr.shapeOf a = Expr.shapeOf b ->
                    Unary (lopa, Binary (op, a, b)) |> optRec

                // pull matching broadcasts over batched dimensions through dot op
                | Binary (Dot, (Unary (DoBroadcast _, a) as ba), (Unary (DoBroadcast _, b) as bb))
                        when List.zip (axesBroadcasted ba) (axesBroadcasted bb)
                             |> List.indexed
                             |> List.exists (fun (d, (aBc, bBc)) -> d < ba.NDims - 2 && aBc.IsBC 
                                                                                     && bBc.IsBC) ->
                    let aOptBc, bOptBc =
                        List.zip (axesBroadcasted ba) (axesBroadcasted bb)
                        |> List.indexed
                        |> List.map (function | d, (aBc, bBc) when d >= ba.NDims-2 -> aBc.Size, bBc.Size
                                              | _, (Broadcasted _, Broadcasted _) -> Size.broadcastable, Size.broadcastable
                                              | _, (aBc, bBc) -> aBc.Size, bBc.Size)
                        |> List.unzip
                    let baOpt = Unary (DoBroadcast aOptBc, a) |> optRec
                    let bbOpt = Unary (DoBroadcast bOptBc, b) |> optRec
                    Unary (DoBroadcast (Expr.shapeOf expr), Binary (Dot, baOpt, bbOpt)) |> optRec

                // optimize gather and scatter index arguments
                | Unary (Gather indices, a) ->
                    Unary (Gather (indices |> List.map (Option.map optRec)), optRec a)
                | Unary (Scatter (indices, shp), a) ->
                    Unary (Scatter (indices |> List.map (Option.map optRec), shp), optRec a)

                // optimize IfThenElse condition
                | Binary (IfThenElse cond, a, b) ->
                    Binary (IfThenElse (optRec cond), optRec a, optRec b)

                // tranform SetSubtensor(Zero, X) into BuildTensor(X)
                | Binary (SetSubtensor (SimpleRanges.Static as rngs), Expr.ZeroExpr, part) ->
                    let shp = Expr.shapeOf expr
                    Expr.buildTensor shp [SimpleRanges.toBaseRanges shp rngs] [optRec part]

                // combine Add(BuildTensor, BuildTensor) into BuildTensor if ranges are not overlapping
                | Binary (Add, Nary (BuildTensor (aShp, aRngs), aParts),
                               Nary (BuildTensor (bShp, bRngs), bParts)) when 
                        aShp=bShp && not (BaseRanges.areOverlapping (aRngs @ bRngs)) ->
                    let aParts = aParts |> List.map optRec
                    let bParts = bParts |> List.map optRec
                    Expr.buildTensor aShp (aRngs @ bRngs) (aParts @ bParts)

                // optimize elements expressions
                | Nary (Elements (resShape, elemExpr), args) ->
                    let args = args |> List.map optRec
                    Nary (Elements (resShape, elemExpr), args)
                    |> optimizeElements
                    |> pullSumOutOfElements
                    |> broadcastInsignificantElementsAxes

                // optmize loops
                | Nary (Channel (Loop loopSpec, ch), args) ->
                    let args = args |> List.map optRec
                    let optChExprs = 
                        Map.toList loopSpec.Channels                        
                        |> List.map (fun (ch, lv) -> lv.Expr)
                        |> optimize
                    let channels =
                        Map.toList loopSpec.Channels
                        |> List.zip optChExprs
                        |> List.map (fun (optExpr, (ch, lv)) -> ch, {lv with Expr=optExpr})
                        |> Map.ofList
                    let loopSpec = {loopSpec with Channels=channels}
                    Nary (NaryOp.Channel (Loop loopSpec, ch), args)

                // pass through
                | Leaf _ -> expr
                | Unary(op, a) -> 
                    let opt = Unary (op, optRec a) 
                    if opt <> expr then optRec opt else opt
                | Binary(op, a, b) -> 
                    let opt = Binary (op, optRec a, optRec b)
                    if opt <> expr then optRec opt else opt
                | Nary(op, es) -> 
                    let opt = Nary (op, List.map optRec es)
                    if opt <> expr then optRec opt else opt

            optimized.LockedSet (expr, opt)
            optimized.LockedSet (opt, opt)
            opt

    /// Optimizes a group of expressions.
    and optimize (exprs: Expr list) : Expr list =
        for expr in exprs do
            Expr.checkExpr expr
        let exprs = exprs |> List.map (optRec >> Expr.check)
        let exprs = 
            if not Debug.DisableCombineIntoElementsOptimization then
                let exprsInfo = ExprInfoT exprs
                exprs |> List.map (combineIntoElementsRec exprsInfo >> Expr.check)
            else exprs
        exprs




