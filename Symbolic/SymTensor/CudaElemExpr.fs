namespace SymTensor.Compiler.Cuda

open System

open Tensor
open Tensor.Utils
open DeepNet.Utils

open SymTensor
open SymTensor.Compiler
open ElemExpr
open UElemExpr


module CudaElemExpr =
    
    type VarNameT = string
    type CodeT = string


    let private constCode tn (c: ConstSpecT) =
        try
            match tn with
            | _ when tn = TypeName.ofType<int> -> sprintf "%d" (c.GetConvertedValue<int>())
            | _ when tn = TypeName.ofType<int64> -> sprintf "%d" (c.GetConvertedValue<int64>())
            | _ when tn = TypeName.ofType<single> -> sprintf "%e" (c.GetConvertedValue<single>())
            | _ when tn = TypeName.ofType<double> -> sprintf "%e" (c.GetConvertedValue<double>())
            | _ -> failwithf "unsupported type: %A" tn
        with 
        | :? InvalidCastException | :? FormatException | :? OverflowException ->
            failwithf "cannot convert constant %A of type %A to expression type %A"
                c (c.GetType()) (TypeName.getType tn)

    let private cppType tn =
        match tn with
        | _ when tn = TypeName.ofType<int> -> "int32_t"
        | _ when tn = TypeName.ofType<int64> -> "int64_t"
        | _ when tn = TypeName.ofType<single> -> "float"
        | _ when tn = TypeName.ofType<double> -> "double"
        | _ -> failwithf "unsupported type: %A" tn


    let generateSizeSpecCode (sizeSymVars: Map<SizeSymbolT, VarNameT>) (ss: SizeSpecT) =
        match SizeSpec.tryEval ss with
        | Some v -> sprintf "%dLL" v
        | None ->
            match SizeSpec.simplify ss with 
            | Base (Fixed c) -> sprintf "%dLL" c.IntValue
            | Base (Sym sym) -> sizeSymVars.[sym]
            | Broadcast -> "1LL"
            | Multinom m ->
                m.Products
                |> Map.toSeq
                |> Seq.map (fun (sp, fac) ->
                    sp.Symbols
                    |> Map.toSeq
                    |> Seq.map (fun (sym, pow) ->
                        [for p in 0L .. pow-1L do yield sizeSymVars.[sym]]
                        |> String.concat "*")
                    |> Seq.append (Seq.singleton (sprintf "%dLL" fac.IntValue))
                    |> String.concat " * ")
                |> String.concat " + "


    let private valueCode op tn (subExprVars: VarNameT list) 
            (argVars: Map<ElemExpr.ArgT, VarNameT>) (sizeSymVars: Map<SizeSymbolT, VarNameT>)  =
        let ssCode = generateSizeSpecCode sizeSymVars

        match op with
        | ULeafOp leafOp ->
            match leafOp with
            | Const c -> constCode tn c
            | SizeValue (ss, _) -> ssCode ss
            | ArgElement ((arg, idxs), _) ->
                let argVar = argVars.[arg]
                let idxStr =
                    idxs
                    |> List.map ssCode
                    |> String.concat ", "
                sprintf "%s.element(%s)" argVars.[arg] idxStr
                      
        | UUnaryOp unaryOp ->
            let av = subExprVars.[0]
            match unaryOp with
            | Negate -> sprintf "-%s" av
            | Abs -> sprintf "fabs(%s)" av
            | SignT -> sprintf "signt(%s)" av
            | Log -> sprintf "log(%s)" av
            | Log10 -> sprintf "log10(%s)" av
            | Exp -> sprintf "exp(%s)" av
            | Sin -> sprintf "sin(%s)" av
            | Cos -> sprintf "cos(%s)" av
            | Tan -> sprintf "tan(%s)" av
            | Asin -> sprintf "asin(%s)" av
            | Acos -> sprintf "acos(%s)" av
            | Atan -> sprintf "atan(%s)" av
            | Sinh -> sprintf "sinh(%s)" av
            | Cosh -> sprintf "cosh(%s)" av
            | Tanh -> sprintf "tanh(%s)" av
            | Sqrt -> sprintf "sqrt(%s)" av
            | Ceil -> sprintf "ceil(%s)" av
            | Floor -> sprintf "floor(%s)" av
            | Round -> sprintf "round(%s)" av
            | Truncate -> sprintf "trunc(%s)" av
            | Sum _ -> failwith "not a simple value"
            | KroneckerRng (s, first, last) ->
                sprintf "((%s) <= (%s) && (%s) <= (%s)) ? %s : 0"
                    (ssCode first) (ssCode s) (ssCode s) (ssCode last) av

        | UBinaryOp binaryOp ->
            let av, bv = subExprVars.[0], subExprVars.[1]
            match binaryOp with
            | Add -> sprintf "%s + %s" av bv
            | Substract -> sprintf "%s - %s" av bv
            | Multiply -> sprintf "%s * %s" av bv
            | Divide -> sprintf "%s / %s" av bv
            | Modulo -> sprintf "%s %% %s" av bv
            | Power -> sprintf "pow(%s, %s)" av bv
            | IfThenElse (left, right) ->
                sprintf "((%s) == (%s)) ? %s : %s" (ssCode left) (ssCode right) av bv



    /// generates a functor that evaluates the UElemFuncT
    let generateFunctor name {Expr=expr; NDims=nTrgtDims; NArgs=nArgs} posOrder =

        if not (Permutation.is posOrder && List.length posOrder = nTrgtDims) then
            invalidArg "posOrder" "posOrder must be a permutation of the target dimensions"

        let mutable varCount = 0
        let newVar () : VarNameT =
            varCount <- varCount + 1
            sprintf "v%d" varCount    

        let mutable sumIdxCount = 0
        let newSumIdxVar () : VarNameT =
            sumIdxCount <- sumIdxCount + 1
            sprintf "s%d" sumIdxCount

        let rec genExpr (UElemExpr (op, subExprs, tn) as expr) (exprVars: Map<UElemExprT, VarNameT>) 
                (argVars: Map<ElemExpr.ArgT, VarNameT>) (sizeSymVars: Map<SizeSymbolT, VarNameT>) (indent: int)
                : VarNameT * Map<UElemExprT, VarNameT> * CodeT =

            let spc = String.replicate indent " "

            match exprVars |> Map.tryFind expr with
            | Some myVarName -> myVarName, exprVars, ""
            | None ->
                let myVarName = newVar ()
                let myVarType = cppType tn

                match op with
                | UUnaryOp (Sum (sumSym, first, last)) -> 
                    // evaluates range
                    let firstVal = first |> generateSizeSpecCode sizeSymVars
                    let lastVal = last |> generateSizeSpecCode sizeSymVars

                    // create index variable and add to size symbols
                    let sumIdxVar = newSumIdxVar ()
                    let sizeSymVars = sizeSymVars |> Map.add sumSym sumIdxVar                    

                    // generate code for summand
                    let sumandExpr = subExprs.[0]
                    let iterResVarName, _, iterCalcCode = 
                        genExpr sumandExpr exprVars argVars sizeSymVars (indent + 2)

                    // generate loop code
                    let sumCode = 
                        sprintf "%s%s %s = 0;\n" spc myVarType myVarName +
                        sprintf "%sfor (idx_t %s = %s; %s <= %s; %s++) {\n"
                            spc sumIdxVar firstVal sumIdxVar lastVal sumIdxVar +
                        iterCalcCode +
                        sprintf "%s  %s += %s;\n" spc myVarName iterResVarName +
                        sprintf "%s}\n" spc

                    // update known expressions and return
                    let exprVars = exprVars |> Map.add expr myVarName                             
                    myVarName, exprVars, sumCode
                | _ ->
                    // evaluate arguments
                    let subExprVars, (exprVars, code) =
                        ((exprVars, ""), subExprs)
                        ||> List.mapFold (fun (exprVars, code) subExpr ->
                            let subExprVar, exprVars, argCode =
                                genExpr subExpr exprVars argVars sizeSymVars indent
                            subExprVar, (exprVars, code + argCode))

                    // generate code for this op
                    let myValueCode = valueCode op tn subExprVars argVars sizeSymVars
                    let myCode =
                        sprintf "%sconst %s %s = %s;\n" spc myVarType myVarName myValueCode

                    // update known expressions and return
                    let exprVars = exprVars |> Map.add expr myVarName                             
                    myVarName, exprVars, code + myCode

        // build arguments vars
        let argVars = 
            seq {for a=0 to nArgs-1 do yield (ElemExpr.Arg a), sprintf "a%d" a}
            |> Map.ofSeq
        let posVars =
            seq {for d=0 to nTrgtDims-1 do yield ElemExpr.idxSymbol d, sprintf "p%d" d}
            |> Map.ofSeq
        
        // generate calculation code
        let resVar, _, calcCode = genExpr expr Map.empty argVars posVars 4

        // generate functor code
        let tmplArgs = [
            yield "typename TTrgt"
            for a=0 to nArgs-1 do yield sprintf "typename Ta%d" a
        ]
        let funcArgs = [
            yield "TTrgt &trgt"
            for d in posOrder do yield sprintf "const idx_t p%d" d
            for a=0 to nArgs-1 do yield sprintf "const Ta%d &a%d" a a
        ]
        let trgtIdxStr =
            seq {for d=0 to nTrgtDims-1 do yield generateSizeSpecCode posVars (ElemExpr.idx d)}
            |> String.concat ", "
        let functorCode =    
            let tmpl =         
                if List.isEmpty tmplArgs then ""
                else sprintf "template <%s>\n" (tmplArgs |> String.concat ", ") 
            tmpl +
            sprintf "struct %s {\n" name +
            sprintf "  _dev void operator() (%s) const {\n" (funcArgs |> String.concat ", ") +
            calcCode +
            sprintf "    trgt.element(%s) = %s;\n" trgtIdxStr resVar +
            sprintf "  }\n" +
            sprintf "};\n\n"
            
        functorCode


    /// Returns a map from dimension to the number of stride one reads/writes that would occur
    /// if this dimension was the X work dimension of the CUDA kernel calculating the elements
    /// expression.
    let strideStats (trgt: TensorManikin) (srcs: TensorManikin list) 
                    {Expr=expr; NDims=nTrgtDims; NArgs=nArgs} =
        
        let rec srcStats expr =
            match expr with
            | UElemExpr (ULeafOp (ArgElement ((Arg argIdx, idxs), _)), _, _) -> 
                let arg = srcs.[argIdx]
                List.init nTrgtDims (fun warpDim ->
                    idxs
                    |> List.indexed
                    |> List.filter (fun (_, pos) -> pos = ElemExpr.idx warpDim)
                    |> List.sumBy (fun (argDim, _) -> arg.Layout.Stride.[argDim])
                    |> fun strIncr -> if strIncr = 1L then 1L else 0L)
            | UElemExpr (UUnaryOp (Sum (sumSym, first, last)), [summand], _) ->
                let iters = SizeSpec.eval last - SizeSpec.eval first + 1L 
                srcStats summand |> List.map (fun oneStrs -> oneStrs * iters)
            | UElemExpr (_, args, _) ->
                args |> List.map srcStats |> List.sumElemwise

        let trgtStats =         
            List.init nTrgtDims (fun warpDim ->
                if trgt.Layout.Stride.[warpDim] = 1L then 1L else 0L)

        List.addElemwise (srcStats expr) trgtStats
        |> List.mapi (fun warpDim oneStrs -> oneStrs * trgt.Shape.[warpDim])


    /// Returns the best position ordering so that coalesced memory access is maximized.
    let bestPosOrder trgt srcs uElemFunc =
        strideStats trgt srcs uElemFunc
        |> List.indexed
        |> List.sortBy snd
        |> List.map fst
    
