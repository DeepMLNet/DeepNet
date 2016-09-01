namespace SymTensor

open Basics
open ArrayNDNS
open ShapeSpec
open VarSpec
open ElemExpr


module ElemExprDeriv =

    /// map containing the derivative for each argument
    type DerivT<'T> = Map<ArgElementSpecT, ElemExprT<'T>>
        
    /// merges to derivative maps
    let private merge (aGrads: DerivT<_>) (bGrads: DerivT<_>) : DerivT<_> =
        (aGrads, bGrads)
        ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                    | Some ovg -> m |> Map.add v (vg + ovg)
                                    | None -> m |> Map.add v vg) 

    let rec reverseDiffStep (expr: ElemExprT<'T>) (eg: ElemExprT<'T>) : DerivT<'T> =    
        let rds = reverseDiffStep
        
        match expr with
        | Leaf (op) ->
            match op with
            | Const _ -> Map.empty
            | SizeValue _ -> Map.empty
            | ArgElement argSpec -> Map [argSpec, eg]

        | Unary (op, a) ->
            match op with
            | Negate -> -eg |> rds a
            | Abs -> eg * signt a |> rds a
            | SignT -> Map.empty
            | Log -> eg * (a ** -(one())) |> rds a
            | Log10 -> eg |> rds (log a / log (scalart 10))
            | Exp -> eg * exp a |> rds a
            | Sin -> eg * cos a |> rds a
            | Cos -> eg * (-sin a) |> rds a
            | Tan -> eg * (one<'T>() + (tan a)**two<'T>()) |> rds a
            | Asin -> eg * (one<'T>() / sqrtt (one<'T>() - a**two<'T>())) |> rds a
            | Acos -> eg * (-one<'T>() / sqrtt (one<'T>() - a**two<'T>())) |> rds a
            | Atan -> eg * (one<'T>() / (one<'T>() + a**two<'T>())) |> rds a
            | Sinh -> eg * cosh a |> rds a
            | Cosh -> eg * sinh a |> rds a
            | Tanh -> eg * (one<'T>() - (tanh a)**two<'T>()) |> rds a
            | Sqrt -> eg * (one<'T>() / (two<'T>() * sqrtt a)) |> rds a
            | Ceil -> Map.empty
            | Floor -> Map.empty
            | Round -> Map.empty
            | Truncate -> Map.empty
            | Sum (sym, first, last) -> eg |> kroneckerRng (Base (Sym sym)) first last |> rds a
            | KroneckerIf (left, right) -> eg |> kroneckerIf left right |> rds a
            | KroneckerRng (sym, first, last) -> eg |> kroneckerRng sym first last |> rds a                

        | Binary (op, a, b) ->
            let inline (.+) da db = 
                merge (rds a da) (rds b db)

            match op with
            | Add -> eg .+ eg 
            | Substract -> eg .+ (-eg)
            | Multiply -> (eg * b) .+ (a * eg)
            | Divide -> eg |> rds (a * b ** (-one()))
            | Modulo -> eg .+ (-truncate (a / b))    // TODO: FIXME
            | Power -> (eg * b * a**(b - one())) .+ (eg * a**b * log a)


    let compute (expr: ElemExprT<'T>) : DerivT<'T> =
        reverseDiffStep expr (one())


    type private DerivDim =
        | SummingDim of SizeSymbolT * SizeSpecT * SizeSpecT * SizeSymbolT
        | FixedDim of SizeSpecT * SizeSymbolT

    let buildDerivElemExpr (expr: ElemExprT<'T>) (exprShp: ShapeSpecT) nArgs =
        let nDims = ShapeSpec.nDim exprShp
        let allDerives = compute expr
        let egArgNo = nArgs
        let egElem = ElemExpr.argElem egArgNo

        let mutable sumSymbolCnt = 0
        let newSumSymbol () =
            sumSymbolCnt <- sumSymbolCnt + 1
            sprintf "__DERIV_%d" sumSymbolCnt |> ElemExpr.sumSymbol

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
                                let exprDimSym = ElemExpr.idxSymbol exprDim
                                match sol.LeftValues |> Map.tryFind exprDimSym with
                                | Some ss -> yield FixedDim (ss, exprDimSym)
                                | None -> yield SummingDim (newSumSymbol(),
                                                            SizeSpec.zero, exprShp.[exprDim]-1,
                                                            exprDimSym)
                        ]              
                        // build indices for eg
                        let egIdxSyms = 
                            egIdxDimInfo 
                            |> List.map (function
                                         | SummingDim (sym, _, _, _) -> Base (Sym sym)
                                         | FixedDim (ss, _) -> ss)

                        // sum over dimensions for which it is necessary
                        let argExprSumed = 
                            (egIdxDimInfo, deriv * egElem egIdxSyms)
                            ||> List.foldBack (fun dimInfo derivSumSoFar ->
                                match dimInfo with
                                | SummingDim (sym, first, last, oldSym) ->
                                    let substSum = 
                                        derivSumSoFar 
                                        |> ElemExpr.substSymSizes (Map [oldSym, Base (Sym sym)]) 
                                    Unary (Sum (sym, first, last), substSum) 
                                | FixedDim (ss, oldSym) -> 
                                    derivSumSoFar
                                    |> ElemExpr.substSymSizes (Map [oldSym, ss]))

                        // apply constraints if necessary
                        let argExpr =
                            (argExprSumed, sol.RightValues)
                            ||> Map.fold (fun kroneckersSoFar idxSym reqVal ->
                                ElemExpr.kroneckerIf (Base (Sym idxSym)) reqVal kroneckersSoFar
                            )

                        // substitute index symbols "Dnnn" with result index symbols "Rnnn"
                        let resSyms = [for d=0 to nArgDims-1 do yield idx d]
                        let idxToResSyms = List.zip idxSyms resSyms |> Map.ofList
                        yield ElemExpr.substSymSizes idxToResSyms argExpr
                ]

                let argExprSum = 
                    if List.isEmpty argExprs then zero ()
                    else argExprs |> List.reduce (+)
                yield argExprSum
        ]

        argDerivExprs


