namespace Tensor.Expr

open System
open System.Reflection

open Tensor
open Tensor.Backend
open DeepNet.Utils

open UExprTypes
open Tensor.Expr.Compiler


module HostEval =
    open Expr

    /// if true, intermediate results are printed during evaluation.
    let mutable debug = false

    /// evaluation functions
    type private EvalT =       

        static member EvalTypeNeutral (evalEnv: EvalEnv, expr: Expr) : ITensor =
             callGeneric<EvalT, ITensor> "Eval" [expr.Type] (evalEnv, expr)


        static member Eval<'R> (evalEnv: EvalEnv, expr: Expr) : Tensor<'R> =
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
                    | _ ->
                        let ts = es |> List.map (fun e -> e.TypeName) |> Set.ofList
                        if ts.Count = 1 then es.Head.Type
                        else typeof<obj>
            callGeneric<EvalT, Tensor<'R>> "DoEval" [argType; retType] (evalEnv, expr)


        static member DoEval<'T, 'R> (evalEnv: EvalEnv, expr: Expr) : Tensor<'R> =
            if expr.Type <> typeof<'R> then
                failwithf "expression of type %A does not match eval function of type %A"
                    expr.Type typeof<'R>

            let varEval vs = evalEnv.VarEnv |> VarEnv.getVarSpec vs |> fun v -> v.Copy() |> box :?> Tensor<'T>
            let shapeEval symShape = ShapeSpec.eval symShape
            let sizeEval symSize = SizeSpec.eval symSize
            let subEval subExpr : Tensor<'T> = EvalT.Eval<'T> (evalEnv, subExpr) 
            let subEvalTypeNeutral (subExpr: Expr) : ITensor = 
                EvalT.EvalTypeNeutral (evalEnv, subExpr)
            let rngEval = 
                SimpleRangesSpec.eval 
                    (fun intExpr -> EvalT.Eval<int64> (evalEnv, intExpr) |> Tensor.value)
            let toBool (v : Tensor<'V>) : Tensor<bool> = v |> box |> unbox
            let toT (v: Tensor<'V>) : Tensor<'T> = v |> box |> unbox
            let toR (v: Tensor<'V>) : Tensor<'R> = v |> box |> unbox
        
            let res : Tensor<'R> = 
                match expr with
                | Leaf(op) ->
                    match op with
                    | Identity (ss, tn) -> HostTensor.identity (sizeEval ss) 
                    | SizeValue (sv, tn) -> sizeEval sv |> conv<'T> |> HostTensor.scalar
                    | Arange (ss, tn) -> 
                        HostTensor.counting (sizeEval ss) |> Tensor<'T>.convert
                    | ScalarConst sc -> HostTensor.scalar (sc.GetValue())
                    | Var(vs) -> varEval vs 
                    |> box |> unbox
                | Unary(op, a) ->
                    let av = subEval a
                    match op with
                    | ArgMaxAxis ax -> Tensor.argMaxAxis ax av |> box |> unbox
                    | ArgMinAxis ax -> Tensor.argMinAxis ax av |> box |> unbox
                    | _ ->
                        match op with
                        | Negate -> -av
                        | Abs -> abs av
                        | SignT -> sgn av
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
                        | Diag(ax1, ax2) -> Tensor.diagAxis ax1 ax2 av
                        | DiagMat(ax1, ax2) -> Tensor.diagMatAxis ax1 ax2 av
                        | Invert -> Tensor.invert av
                        | Sum -> Tensor.sumTensor av
                        | SumAxis ax -> Tensor.sumAxis ax av
                        | Product -> Tensor.productTensor av
                        | ProductAxis ax -> Tensor.productAxis ax av
                        | MaxAxis ax -> Tensor.maxAxis ax av
                        | MinAxis ax -> Tensor.minAxis ax av
                        | ArgMaxAxis _
                        | ArgMinAxis _ -> failwith "implemented above"
                        | Reshape ss -> Tensor.reshape (shapeEval ss) av
                        | DoBroadcast ss -> Tensor.broadcastTo (shapeEval ss) av
                        | PermuteAxes perm -> Tensor.permuteAxes perm av
                        | ReverseAxis ax -> Tensor.reverseAxis ax av
                        | UnaryOp.Gather indices ->
                            let vIndices = 
                                indices 
                                |> List.map (Option.map (fun idx -> EvalT.Eval<int64> (evalEnv, idx)))
                            Tensor.gather vIndices av
                        | UnaryOp.Scatter (indices, trgtShp) ->
                            let vIndices = 
                                indices 
                                |> List.map (Option.map (fun idx -> EvalT.Eval<int64> (evalEnv, idx)))
                            Tensor.scatter vIndices (shapeEval trgtShp) av                        
                        | UnaryOp.Subtensor sr -> av.[rngEval sr]
                        | StoreToVar vs -> 
                            // TODO: stage variable write to avoid overwrite of used variables
                            let tv : Tensor<'T> = VarEnv.getVarSpec vs evalEnv.VarEnv
                            tv.CopyFrom av
                            Tensor.relayout TensorLayout.emptyVector av
                        | NullifyJacobian -> av
                        | AssumeJacobian _ -> av
                        | Print msg ->
                            printfn "%s=\n%A\n" msg av
                            av
                        | Dump name ->
                            Dump.dumpValue name av
                            av
                        | CheckFinite name ->
                            if not (Tensor.allFinite av) then
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
                        | MaxElemwise -> Tensor.maxElemwise av bv 
                        | MinElemwise -> Tensor.minElemwise av bv 
                        | Dot -> av .* bv
                        | TensorProduct -> Tensor.tensorProduct av bv
                        | And -> (toBool av) &&&& (toBool bv) |> toT
                        | Or -> (toBool av) |||| (toBool bv) |> toT
                        | BinaryOp.IfThenElse cond ->
                            let condVal = EvalT.Eval<bool> (evalEnv, cond) 
                            Tensor.ifThenElse condVal av bv
                        | BinaryOp.SetSubtensor sr -> 
                            let v = Tensor.copy av
                            v.[rngEval sr] <- bv
                            v                        
                        |> box |> unbox

                | Nary(op, es) ->
                    match op with 
                    | Discard -> 
                        es |> List.iter (subEvalTypeNeutral >> ignore)
                        HostTensor.zeros<'R> [0L] |> box
                    | BuildTensor (shp, rngs) ->
                        let trgt = HostTensor.zeros<'R> (shapeEval shp)
                        for rng, e in List.zip rngs es do                            
                            let aryRng = rng |> List.map (fun (first, last) -> 
                                Rng.Rng (Some (sizeEval first), Some (sizeEval last)))
                            trgt.[aryRng] <- subEval e |> toR
                        trgt |> box
                    | NaryOp.Elements (resShape, elemExpr) -> 
                        let esv = es |> List.map subEval 
                        let nResShape = shapeEval resShape
                        Elem.Interpreter.eval elemExpr esv nResShape |> box
                    | Interpolate ip ->  
                        es |> List.map subEval |> Interpolator.interpolate ip |> box
                    | NaryOp.Channel (MultiChannelOp.Loop spec, channel) -> 
                        let esv = es |> List.map subEvalTypeNeutral
                        if Trace.isActive () then Trace.enteringLoop (expr |> UExpr.toUExpr |> Trace.extractLoop)
                        let channelValues = EvalT.LoopEval (evalEnv, spec, esv)
                        if Trace.isActive () then Trace.leavingLoop (expr |> UExpr.toUExpr |> Trace.extractLoop)
                        channelValues.[channel] |> box
                    | ExtensionOp eop -> 
                        eop.EvalSimple (es |> List.map subEval) |> box
                    |> unbox

            if Trace.isActive () then
                Trace.exprEvaled (expr |> UExpr.toUExpr) (lazy (res :> ITensor))
            res

        /// evaluates all channels of a loop
        static member LoopEval (evalEnv: EvalEnv, spec: LoopSpec, args: ITensor list) 
                               : Map<Channel, ITensor> =

            // iteration index variables
            let nIters = SizeSpec.eval spec.Length
            let iterAry = HostTensor.zeros<int64> []
            let itersRemAry = HostTensor.zeros<int64> []

            // create channel information
            let channelInfos =
                spec.Channels
                |> Map.map (fun ch lv ->
                    let sliceShp = lv.Expr.Shape |> ShapeSpec.eval
                    let targetShp = sliceShp |> List.insert lv.SliceDim nIters
                    let target = Tensor.NewOfType (targetShp, lv.Expr.Type, HostTensor.Dev, order=RowMajor)
                    {
                        LoopEval.Shape      = sliceShp
                        LoopEval.SliceDim   = lv.SliceDim
                        LoopEval.Target     = target
                    })

            // perform loop
            for iter in 0L .. nIters-1L do            
                if Trace.isActive () then Trace.setLoopIter iter
                   
                // set iteration indices
                iterAry.[[]] <- iter
                itersRemAry.[[]] <- nIters - iter - 1L

                // calculate and store channel values
                let iterVarEnv, iterChannelEnv =
                    LoopEval.buildInOut nIters iter iterAry itersRemAry spec.Vars args channelInfos
                let iterEvalEnv = {evalEnv with VarEnv=iterVarEnv}
                for KeyValue(ch, lv) in spec.Channels do
                    iterChannelEnv.[ch].[Fill] <- EvalT.EvalTypeNeutral (iterEvalEnv, lv.Expr)

            // return outputs
            channelInfos |> Map.map (fun ch ci -> ci.Target)                          

            
    /// Evaluates a unified expression.
    /// This is done by evaluating the generating expression.
    let evalUExpr (evalEnv: EvalEnv) uExpr =
        let expr = UExpr.toExpr uExpr
        callGeneric<EvalT, ITensor> "Eval" [expr.Type] (evalEnv, expr)

    /// Evaluates the specified unified expressions.
    /// This is done by evaluating the generating expressions.
    let evalUExprs (evalEnv: EvalEnv) (uexprs: UExprT list) =
        List.map (evalUExpr evalEnv) uexprs


[<AutoOpen>]
module HostEvalTypes =

    /// evaluates expression on host using interpreter
    let onHost (compileEnv: CompileEnvT) (uexprs: UExprT list) = 

        // check requirements
        if compileEnv.ResultLoc <> HostTensor.Dev then
            failwith "host evaluator needs host result location"
        for KeyValue(vs, loc) in compileEnv.VarLocs do
            if loc <> HostTensor.Dev then
                failwithf "host evaluator cannot evaluate expression with 
                           variable %A located in %A" vs loc

        // evaluation function
        let evalFn = fun evalEnv -> HostEval.evalUExprs evalEnv uexprs

        // build diagnostics information
        let joinedExpr = 
            UExpr (UNaryOp NaryOp.Discard, uexprs, {ChannelType=Map.empty; ChannelShape=Map.empty; Expr=None})
        let diag : CompileDiagnosticsT = {
            UExpr          = joinedExpr
            ExecUnits      = []
            SubDiagnostics = Map.empty
        }

        evalFn, Some diag

        


        
