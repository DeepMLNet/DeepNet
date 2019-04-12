namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Expr



type CompileEnv = {
    VarOffsetStrides: Map<VarName, int64 * int64 list> 
}

type CompileData = {
    Alloc: AllocReq -> AllocStub
    Env: CompileEnv
}


type ICompilableOp =
    /// Should compute the output stubs given the input stubs.
    abstract ChStubs: CompileData -> Map<Arg, TensorStub> -> Map<Ch, TensorStub>


module BaseExprCompiler =

    /// Evaluates an expression tree using the specified function for evaluation
    /// of each expression given its arguments.
    let eval (fn: BaseExpr -> Map<Arg, 'D> -> Map<Ch, 'D>) (group: BaseExprGroup) =
        /// Evaluated values for each BaseExpr channel.
        let exprChValues = Dictionary<BaseExprCh, 'D> ()
        /// BaseExprs that have all their arguments evaluated and thus can be evaluated themselves.
        let evalQueue = Queue<BaseExpr> ()
        /// Values that are still missing but needed for evaluation of a BaseExpr.
        let missingValuesForExpr = Dictionary<BaseExpr, HashSet<BaseExprCh>> ()
        /// Dependants of an expression channel that are not yet evaluated.
        let unevaledDeps = Dictionary<BaseExprCh, HashSet<BaseExpr>> ()
                  
        // Build missing values for each expression and enqueue leafs.
        for expr in group.AllExprs do
            let dependingOn = HashSet<_> ()
            for KeyValue(_, arg) in expr.Args do
                dependingOn.Add arg |> ignore
            missingValuesForExpr.[expr] <- dependingOn

            if dependingOn.Count = 0 then
                evalQueue.Enqueue expr

            for ch in expr.Channels do
                unevaledDeps.[expr.[ch]] <- HashSet (group.Dependants expr.[ch])

        // Loop until all expressions are evaluated.
        while evalQueue.Count > 0 do
            // Get expression to evaluate and its argument values.
            let expr = evalQueue.Dequeue ()
            let argVals = expr.Args |> Map.map (fun _ argExpr -> exprChValues.[argExpr])

            // Evaluate.
            let chVals = fn expr argVals

            // Store the result value of the expression and update its dependants.
            for KeyValue(ch, value) in chVals do
                // Store channel value.
                exprChValues.[expr.[ch]] <- value

                // Update dependants.
                for dep in group.Dependants expr.[ch] do
                    let mv = missingValuesForExpr.[dep]
                    mv.Remove expr.[ch] |> ignore

                    // Enqueue, if all arguments have evaluated values.
                    if mv.Count = 0 then
                        evalQueue.Enqueue dep

                // Update unevaled deps.
                for KeyValue(_, arg) in expr.Args do    
                    let ueDeps = unevaledDeps.[arg]
                    ueDeps.Remove expr |> ignore

                    // Remove value, if all dependants are evaluated.
                    if ueDeps.Count = 0 then
                        exprChValues.Remove arg |> ignore

        exprChValues |> Map.ofDictionary


    /// Walks over an expression tree in such a way that all arguments of an
    /// expression are evaluated first before that expression itself is evaluated.
    let iter (fn: BaseExpr -> unit) (group: BaseExprGroup) =
        let evalFn (expr: BaseExpr) (_arg: Map<Arg, unit>) =
            fn expr
            expr.Channels |> Set.toSeq |> Seq.map (fun ch -> ch, ()) |> Map.ofSeq
        eval evalFn group |> ignore


    // step 1: perform allocations
    let performLayouting (env: CompileEnv) (rootExpr: BaseExpr) =
        let group = BaseExprGroup [rootExpr]

        /// All allocation stubs.
        let allocStubs = ResizeArray<AllocStub> ()
        let alloc expr req =
            let alloc = {
                Req = req
                Users = HashSet<BaseExpr> ()
            }
            alloc.Users.Add expr |> ignore
            allocStubs.Add alloc
            alloc

        /// All tensor stubs.
        let tensorStubs = Dictionary<BaseExprCh, TensorStub> ()

        // Walk over expression tree.
        group |> iter (fun expr ->    
            let compileFns = {
                Alloc = alloc expr
                Env = env
            }
            let argStubs = expr.Args |> Map.map (fun _ argExpr -> tensorStubs.[argExpr])

            // Add expression to users of its argument stubs.
            for KeyValue(_, argStub) in argStubs do
                match argStub.Storage with
                | StorageStub.Allocated allocStub -> allocStub.Users.Add expr |> ignore
                | StorageStub.Dynamic -> ()
                
            // Compute channel stubs given argument stubs.
            let chStubs =
                match expr.Op with
                | :? ICompilableOp as cop -> 
                    // Let op compute its output stubs given argument stubs.
                    cop.ChStubs compileFns argStubs
                | _ ->
                    // Op cannot compute stubs, generate dynamic stubs for all
                    // its output channels. 
                    expr.Channels
                    |> Set.toSeq
                    |> Seq.map (fun ch -> 
                        ch, {
                            Shape = expr.Shapes.[ch] |> Shape.eval
                            TypeName = expr.TypeNames.[ch]
                            Dev = expr.Devs.[ch]
                            OffsetStride = None
                            Storage = StorageStub.Dynamic
                        })
                    |> Map.ofSeq

            // Add expression to users of its channel stubs.
            for KeyValue(_, chStub) in chStubs do
                match chStub.Storage with
                | StorageStub.Allocated allocStub -> allocStub.Users.Add expr |> ignore
                | StorageStub.Dynamic -> ()

            // Store channel stubs.
            for KeyValue(ch, chStub) in chStubs do
                tensorStubs.[expr.[ch]] <- chStub
            )


        // next step? 
        // - Try to write the allocation stub function for some ops.
        // - Before that: switch to op extenders here?

        ()


module CompileTools =

    let channelStubs (fns: CompileData) (op: IOp) =
        op.Channels
        |> Set.toSeq
        |> Seq.map (fun ch ->
            let typeName = op.TypeNames.[ch]
            let shape = Shape.eval op.Shapes.[ch]
            let dev = op.Devs.[ch]
            ch, TensorStub.alloc (fns.Alloc, typeName, shape, dev))
        |> Map.ofSeq
