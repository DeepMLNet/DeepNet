namespace Tensor.Expr.Elem

open Tensor.Expr
open DeepNet.Utils


/// Functions for calculating derivatives of element expressions.
module Deriv =

    /// map containing the derivative for each argument
    type DerivT = Map<ArgElementSpec, Expr>
        
    /// merges to derivative maps
    let private merge (aGrads: DerivT) (bGrads: DerivT) : DerivT =
        (aGrads, bGrads)
        ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                    | Some ovg -> m |> Map.add v (vg + ovg)
                                    | None -> m |> Map.add v vg) 

    let rec reverseDiffStep (expr: Expr) (eg: Expr) : DerivT =    
        let rds = reverseDiffStep
        let zero = 0 |> convTo expr.Type |> Expr.scalar
        let one = 1 |> convTo expr.Type |> Expr.scalar
        let two = 2 |> convTo expr.Type |> Expr.scalar

        match expr with
        | Leaf (op) ->
            match op with
            | Const _ -> Map.empty
            | SizeValue _ -> Map.empty
            | ArgElement (argSpec, _) -> Map [argSpec, eg]

        | Unary (op, a) ->
            match op with
            | Negate -> -eg |> rds a
            | Abs -> eg * Expr.signt a |> rds a
            | SignT -> Map.empty
            | Log -> eg * (a ** -(one)) |> rds a
            | Log10 -> eg |> rds (log a / log (Expr.scalar 10))
            | Exp -> eg * exp a |> rds a
            | Sin -> eg * cos a |> rds a
            | Cos -> eg * (-sin a) |> rds a
            | Tan -> eg * (one + (tan a)**two) |> rds a
            | Asin -> eg * (one / Expr.sqrtt (one - a**two)) |> rds a
            | Acos -> eg * (-one / Expr.sqrtt (one - a**two)) |> rds a
            | Atan -> eg * (one / (one + a**two)) |> rds a
            | Sinh -> eg * cosh a |> rds a
            | Cosh -> eg * sinh a |> rds a
            | Tanh -> eg * (one - (tanh a)**two) |> rds a
            | Sqrt -> eg * (one / (two * Expr.sqrtt a)) |> rds a
            | Ceil -> Map.empty
            | Floor -> Map.empty
            | Round -> Map.empty
            | Truncate -> Map.empty
            | Sum (sym, first, last) -> eg |> Expr.kroneckerRng (SizeSpec.Base (BaseSize.Sym sym)) first last |> rds a
            | KroneckerRng (sym, first, last) -> eg |> Expr.kroneckerRng sym first last |> rds a                

        | Binary (op, a, b) ->
            let inline (.+) da db = 
                merge (rds a da) (rds b db)

            match op with
            | Add -> eg .+ eg 
            | Substract -> eg .+ (-eg)
            | Multiply -> (eg * b) .+ (a * eg)
            | Divide -> eg |> rds (a * b ** (-one))
            | Modulo -> eg .+ (-truncate (a / b))    // TODO: FIXME
            | Power -> (eg * b * a**(b - one)) .+ (eg * a**b * log a)
            | IfThenElse (left, right) -> 
                Expr.ifThenElse left right eg zero .+ Expr.ifThenElse left right zero eg 

    let compute (expr: Expr) : DerivT =
        let one = 1 |> convTo expr.Type |> Expr.scalar
        reverseDiffStep expr one

    let ofArgElem (argElem: Expr) (deriv: DerivT) =
        let zero = 0 |> convTo argElem.Type |> Expr.scalar
        match deriv |> Map.tryFind (Expr.extractArg argElem) with
        | Some da -> da
        | None -> zero

    type private DerivDim =
        | SummingDim of SizeSymbol * SizeSpec * SizeSpec * SizeSymbol
        | FixedDim of SizeSpec * SizeSymbol

    let buildDerivElemExpr (expr: Expr) (exprShp: ShapeSpec) nArgs =
        let nDims = ShapeSpec.nDim exprShp
        let allDerives = compute expr
        let egArgNo = nArgs
        let egElem = Expr.argElemWithType (Expr.typeName expr).Type egArgNo
        let zero = 0 |> convTo expr.Type |> Expr.scalar

        let mutable sumSymbolCnt = 0
        let newSumSymbol () =
            sumSymbolCnt <- sumSymbolCnt + 1
            sprintf "__DERIV_%d" sumSymbolCnt |> Expr.sumSymbol

        let argDerivExprs = [
            for arg=0 to nArgs-1 do
            
                let argDerivs = allDerives |> Map.filter (fun (Arg n, _) _ -> n=arg)
                let argExprs = [
                    for KeyValue ((_, argIdx), deriv) in argDerivs do
                        // solve for target indices given derivative indices
                        let nArgDims = argIdx.Length
                        let idxSyms = [for d=0 to nArgDims-1 do yield sprintf "D%d" d |> SizeSymbol.ofName]
                        let sol = ShapeSpec.solve argIdx idxSyms

                        // extract sum information
                        let egIdxDimInfo = [
                            for exprDim=0 to nDims-1 do
                                let exprDimSym = Expr.idxSymbol exprDim
                                match sol.LeftValues |> Map.tryFind exprDimSym with
                                | Some ss -> yield FixedDim (ss, exprDimSym)
                                | None -> yield SummingDim (newSumSymbol(),
                                                            SizeSpec.zero, exprShp.[exprDim]-1L,
                                                            exprDimSym)
                        ]              
                        // build indices for eg
                        let egIdxSyms = 
                            egIdxDimInfo 
                            |> List.map (function
                                         | SummingDim (sym, _, _, _) -> SizeSpec.Base (BaseSize.Sym sym)
                                         | FixedDim (ss, _) -> ss)
                        let funcDimSym = SizeSymbol.ofName "F"
                        let egIdxSyms = (SizeSpec.Base (BaseSize.Sym funcDimSym)) :: egIdxSyms

                        // sum over dimensions for which it is necessary
                        let argExprSumed = 
                            (egIdxDimInfo, deriv * egElem egIdxSyms)
                            ||> List.foldBack (fun dimInfo derivSumSoFar ->
                                match dimInfo with
                                | SummingDim (sym, first, last, oldSym) ->
                                    let substSum = 
                                        derivSumSoFar 
                                        |> Expr.substSymSizes (Map [oldSym, SizeSpec.Base (BaseSize.Sym sym)]) 
                                    Unary (Sum (sym, first, last), substSum) 
                                | FixedDim (ss, oldSym) -> 
                                    derivSumSoFar
                                    |> Expr.substSymSizes (Map [oldSym, ss]))

                        // apply constraints if necessary
                        let argExpr =
                            (argExprSumed, sol.RightValues)
                            ||> Map.fold (fun kroneckersSoFar idxSym reqVal ->
                                Expr.ifThenElse (SizeSpec.Base (BaseSize.Sym idxSym)) reqVal kroneckersSoFar zero)

                        // substitute index symbols "Dnnn" with result index symbols "R(nnn+1)"
                        let resSyms = [for d=1 to nArgDims do yield Expr.idx d]
                        let idxToResSyms = 
                            List.zip idxSyms resSyms 
                            |> Map.ofList
                            |> Map.add funcDimSym (Expr.idx 0)
                        yield Expr.substSymSizes idxToResSyms argExpr
                ]

                let argExprSum = 
                    if List.isEmpty argExprs then zero 
                    else argExprs |> List.reduce (+)
                yield argExprSum
        ]

        argDerivExprs

