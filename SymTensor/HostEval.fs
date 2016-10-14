namespace SymTensor

open System
open System.Reflection

open Basics
open UExprTypes
open ArrayNDNS
open ArrayNDNS.ArrayND


module HostEval =
    open Expr

    /// if true, intermediate results are printed during evaluation.
    let mutable debug = false

    /// evaluation functions
    type private EvalT =       

        static member Eval<'R> (evalEnv: EvalEnvT, expr: ExprT) : ArrayNDHostT<'R> =
            let retType = expr.Type
            if retType <> typeof<'R> then
                failwithf "expression of type %A does not match eval function of type %A"
                    retType typeof<'R>

            let argType = 
                match expr with
                | Leaf _ -> retType
                | Unary (_, a) -> a.Type
                | Binary (_, a, b) ->
                    if a.Type <> b.Type then
                        failwithf "currently arguments of binary ops must have same data type 
                                   but we got types %A and %A" a.Type b.Type
                    a.Type
                | Nary (_, es) ->
                    match es with
                    | [] -> retType
                    | [e] -> e.Type
                    | e::res ->
                        if res |> List.exists (fun re -> re.Type <> e.Type) then
                            failwithf "currently arguments of n-ary ops must all have the same 
                                       data type but we got types %A" 
                                       (es |> List.map (fun e -> e.Type))
                        e.Type

            callGeneric<EvalT, ArrayNDHostT<'R>> "DoEval" [argType; retType] (evalEnv, expr)

        static member DoEval<'T, 'R> (evalEnv: EvalEnvT, expr: ExprT) : ArrayNDHostT<'R> =
            if expr.Type <> typeof<'R> then
                failwithf "expression of type %A does not match eval function of type %A"
                    expr.Type typeof<'R>

            let varEval vs = evalEnv.VarEnv |> VarEnv.getVarSpec vs |> box :?> ArrayNDHostT<'T>
            let shapeEval symShape = ShapeSpec.eval symShape
            let sizeEval symSize = SizeSpec.eval symSize
            let subEval subExpr : ArrayNDHostT<'T> = EvalT.Eval<'T> (evalEnv, subExpr) 
            let rngEval = 
                SimpleRangesSpec.eval 
                    (fun intExpr -> EvalT.Eval<int> (evalEnv, intExpr) |> ArrayND.value)
            let toBool (v : ArrayNDHostT<'V>) : ArrayNDHostT<bool> = v |> box |> unbox
            let toT (v: ArrayNDHostT<'V>) : ArrayNDHostT<'T> = v |> box |> unbox
            let toR (v: ArrayNDHostT<'V>) : ArrayNDHostT<'R> = v |> box |> unbox
        
            let res : ArrayNDHostT<'R> = 
                match expr with
                | Leaf(op) ->
                    match op with
                    | Identity (ss, tn) -> ArrayNDHost.identity (sizeEval ss) 
                    | SizeValue (sv, tn) -> sizeEval sv |> conv<'T> |> ArrayNDHost.scalar
                    | Arange (ss, tn) -> 
                        ArrayNDHost.arange (sizeEval ss) 
                        |> ArrayND.convert :> ArrayNDT<'T> :?> ArrayNDHostT<'T>
                    | ScalarConst sc -> ArrayNDHost.scalar (sc.GetValue())
                    | Var(vs) -> varEval vs 
                    |> box |> unbox
                | Unary(op, a) ->
                    let av = subEval a
                    match op with
                    | Negate -> -av
                    | Abs -> abs av
                    | SignT -> ArrayND.signt av
                    | Log -> log av
                    | Log10 -> log10 av
                    | Exp -> exp av
                    | Sin -> sin av
                    | Cos -> cos av
                    | Tan -> tan av
                    | Asin -> asin av
                    | Acos -> acos av
                    | Atan -> atan av
                    | Sinh -> sinh av
                    | Cosh -> cosh av
                    | Tanh -> tanh av
                    | Sqrt -> sqrt av
                    | Ceil -> ceil av
                    | Floor -> floor av
                    | Round -> round av
                    | Truncate -> truncate av
                    | Not -> ~~~~(toBool av) |> toT
                    | Diag(ax1, ax2) -> ArrayND.diagAxis ax1 ax2 av
                    | DiagMat(ax1, ax2) -> ArrayND.diagMatAxis ax1 ax2 av
                    | Invert -> ArrayND.invert av
                    | Sum -> ArrayND.sum av
                    | SumAxis ax -> ArrayND.sumAxis ax av
                    | Reshape ss -> ArrayND.reshape (shapeEval ss) av
                    | DoBroadcast ss -> ArrayND.broadcastToShape (shapeEval ss) av
                    | PermuteAxes perm -> ArrayND.permuteAxes perm av
                    | ReverseAxis ax -> ArrayND.reverseAxis ax av
                    | Gather indices ->
                        let vIndices = 
                            indices 
                            |> List.map (Option.map (fun idx -> EvalT.Eval<int> (evalEnv, idx)))
                        ArrayND.gather vIndices av
                    | Scatter (indices, trgtShp) ->
                        let vIndices = 
                            indices 
                            |> List.map (Option.map (fun idx -> EvalT.Eval<int> (evalEnv, idx)))
                        ArrayND.scatter vIndices (shapeEval trgtShp) av                        
                    | Subtensor sr -> av.[rngEval sr]
                    | StoreToVar vs -> 
                        // TODO: stage variable write to avoid overwrite of used variables
                        ArrayND.copyTo av (VarEnv.getVarSpec vs evalEnv.VarEnv)
                        ArrayND.relayout ArrayNDLayout.emptyVector av
                    | NullifyJacobian -> av
                    | AssumeJacobian _ -> av
                    | Print msg ->
                        printfn "%s=\n%A\n" msg av
                        av
                    | Dump name ->
                        Dump.dumpValue name av
                        av
                    | CheckFinite name ->
                        if not (ArrayND.allFinite av |> ArrayND.value) then
                            printfn "Infinity or NaN encountered in %s with value:\n%A" name av
                            failwithf "Infinity or NaN encountered in %s" name
                        av
                    | Annotated _-> av  
                    | Held (_, heldOp) ->
                        failwithf "the held op %A must be expanded before evaluation" heldOp
                    |> box |> unbox              
                | Binary(op, a, b) ->
                    let av, bv = subEval a, subEval b
                    match op with
                    | Equal -> av ==== bv |> toR
                    | Less -> av <<<< bv |> toR
                    | LessEqual -> av <<== bv |> toR
                    | Greater -> av >>>> bv |> toR
                    | GreaterEqual -> av >>== bv |> toR
                    | NotEqual -> av <<>> bv |> toR
                    | _ ->
                        match op with
                        | Equal | Less | LessEqual 
                        | Greater | GreaterEqual | NotEqual
                            -> failwith "implemented above"
                        | Add -> av + bv
                        | Substract -> av - bv
                        | Multiply -> av * bv
                        | Divide -> av / bv
                        | Modulo -> av % bv
                        | Power -> av ** bv
                        | MaxElemwise -> ArrayND.maxElemwise av bv 
                        | MinElemwise -> ArrayND.minElemwise av bv 
                        | Dot -> av .* bv
                        | TensorProduct -> av %* bv
                        | And -> (toBool av) &&&& (toBool bv) |> toT
                        | Or -> (toBool av) |||| (toBool bv) |> toT
                        | IfThenElse cond ->
                            let condVal = EvalT.Eval<bool> (evalEnv, cond) 
                            ArrayND.ifThenElse condVal av bv
                        | SetSubtensor sr -> 
                            let v = ArrayND.copy av
                            v.[rngEval sr] <- bv
                            v                        
                        |> box |> unbox

                | Nary(op, es) ->
                    let esv = es |> List.map subEval
                    match op with 
                    | Discard -> ArrayNDHost.zeros [0]
                    | Elements (resShape, elemExpr) -> 
                        let esv = esv |> List.map (fun v -> v :> ArrayNDT<'T>)
                        let nResShape = shapeEval resShape
                        ElemExprHostEval.eval elemExpr esv nResShape    
                    | Interpolate ip -> esv |> Interpolator.interpolate ip 
                    | Channel (Loop spec, channel) -> 
                        if Trace.isActive () then Trace.enteringLoop (expr |> UExpr.toUExpr |> Trace.extractLoop)
                        let channelValues = EvalT.LoopEval (evalEnv, spec, esv)
                        if Trace.isActive () then Trace.leavingLoop (expr |> UExpr.toUExpr |> Trace.extractLoop)
                        channelValues.[channel]                       
                    | ExtensionOp eop -> eop.EvalSimple esv 
                    |> box |> unbox

            if Trace.isActive () then
                Trace.exprEvaled (expr |> UExpr.toUExpr) (lazy (res :> IArrayNDT))
            res

        /// evaluates all channels of a loop
        static member LoopEval<'T, 'R> (evalEnv: EvalEnvT, spec: LoopSpecT, args: ArrayNDHostT<'T> list) 
                                       : Map<ChannelT, ArrayNDHostT<'R>> =

            let args = args |> List.map (fun arg -> arg :> IArrayNDT)

            // iteration index variables
            let nIters = SizeSpec.eval spec.Length
            let iterAry = ArrayNDHost.zeros<int> []
            let itersRemAry = ArrayNDHost.zeros<int> []

            // create channel information
            let channelInfos =
                spec.Channels
                |> Map.map (fun ch lv ->
                    let sliceShp = lv.Expr.Shape |> ShapeSpec.eval
                    let targetShp = sliceShp |> List.insert lv.SliceDim nIters
                    {
                        LoopEval.Shape    = sliceShp
                        LoopEval.SliceDim = lv.SliceDim
                        LoopEval.Target   = ArrayNDHost.zeros<'R> targetShp :> IArrayNDT
                    })

            // perform loop
            for iter=0 to nIters-1 do            
                if Trace.isActive () then Trace.setLoopIter iter
                   
                // set iteration indices
                iterAry.[[]] <- iter
                itersRemAry.[[]] <- nIters - iter - 1

                // calculate and store channel values
                let iterVarEnv, iterChannelEnv =
                    LoopEval.buildInOut iter iterAry itersRemAry spec.Vars args channelInfos
                let iterEvalEnv = {evalEnv with VarEnv=iterVarEnv}
                for KeyValue(ch, lv) in spec.Channels do
                    (iterChannelEnv.[ch] :?> ArrayNDHostT<'R>).[Fill] <- 
                        EvalT.Eval (iterEvalEnv, lv.Expr)

            // return outputs
            channelInfos |> Map.map (fun ch ci -> ci.Target :?> ArrayNDHostT<'R>)                          
            
    /// Evaluates a unified expression.
    /// This is done by evaluating the generating expression.
    let evalUExpr (evalEnv: EvalEnvT) uExpr =
        let expr = UExpr.toExpr uExpr
        callGeneric<EvalT, IArrayNDT> "Eval" [expr.Type] (evalEnv, expr)

    /// Evaluates the specified unified expressions.
    /// This is done by evaluating the generating expressions.
    let evalUExprs (evalEnv: EvalEnvT) (uexprs: UExprT list) =
        List.map (evalUExpr evalEnv) uexprs


[<AutoOpen>]
module HostEvalTypes =
    /// evaluates expression on host using interpreter
    let onHost (compileEnv: CompileEnvT) (uexprs: UExprT list) = 
        if compileEnv.ResultLoc <> LocHost then
            failwith "host evaluator needs host result location"
        for KeyValue(vs, loc) in compileEnv.VarLocs do
            if loc <> LocHost then
                failwithf "host evaluator cannot evaluate expression with variable %A located in %A" vs loc

        fun evalEnv -> HostEval.evalUExprs evalEnv uexprs


        
