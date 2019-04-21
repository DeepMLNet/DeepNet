namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Expr



type CompileEnv = {
    VarOffsetStrides: Map<VarName, int64 * int64 list> 
}


//type ActionData = {
//    Alloc:              AllocReq -> AllocStub
//    Env:                CompileEnv
//    ArgStubs:           Map<Arg, TensorStub> 
//    ChStubs:            Map<Ch, TensorStub>
//    Expr:               BaseExpr
//}

type ExecuteData = {
    StubValues:         IReadOnlyDictionary<TensorStub, ITensor>
    ArgValues:          Map<Arg, ITensor>
    ChValues:           Map<Ch, ITensor>
}

type IAction =
    /// Execute actions and return resulting tensor for dynamic stubs.
    /// A returned tensor for a dynamic tensor stub must not use a storage
    /// with a static allocation.
    /// TODO: verify and enforce.
    abstract Execute: ExecuteData -> Map<Ch, ITensor>


[<ReferenceEquality>]
type ActionGroup = {
    DependsOn:          HashSet<ActionGroup>
    Actions:            IAction list
}


type CompileData = {
    Alloc:              AllocReq -> AllocStub
    Env:                CompileEnv
    Expr:               BaseExpr
    ArgStubs:           Map<Arg, TensorStub> 
    OverwritableArgs:   Set<Arg>
    ChStubWishes:       Map<Ch, TensorStub>
    ChStubs:            Map<Ch, TensorStub>
    ArgStubWishes:      Map<Arg, TensorStub>
}


type CompileOutput = {
    ChStubs:            Map<Ch, TensorStub>
    Actions:            IAction list
}

type CompileResult = {
    Allocs:             AllocStub list
    ChStubs:            Map<BaseExprCh, TensorStub>
    ActionGroups:       ActionGroup list
}


type UpPropData = {
    Alloc:              AllocReq -> AllocStub
    Env:                CompileEnv    
    Expr:               BaseExpr
    ChStubWishes:       Map<Ch, TensorStub>
}

type UpPropOutput = {
    ChStubs:            Map<Ch, TensorStub>
    ArgStubWishes:      Map<Arg, TensorStub>
}


type UpPropResult = {
    Allocs:             AllocStub list
    ChStubs:            Map<BaseExprCh, TensorStub>
    ChStubWishes:       Map<BaseExprCh, TensorStub>
    ArgStubWishes:      Map<BaseExpr, Map<Arg, TensorStub>>
}


type ICompilableOp =
    /// Should compute the channel stubs given the argument stubs.
    abstract Compile: CompileData -> CompileOutput

    //abstract Actions: ActionData -> IAction list

type ITensorStubWishPropagatingOp =
    /// Should compute argument stub wishes given channel stub wishes.
    abstract PropagateWishes: UpPropData -> UpPropOutput


type NonCompilableOpAction (expr: BaseExpr, argStubs: Map<Arg, TensorStub>) =
    interface IAction with
        member this.Execute data =
            failwith "TODO"


module BaseExprCompiler =

    /// Propagates tensor stub wishes from the roots of the expression tree towards its nodes.
    /// Only wishes for non-aliased tensor stubs are honored.
    let propagateStubWishes (env: CompileEnv) (chStubWishes: Map<BaseExprCh, TensorStub>) (group: BaseExprGroup) =

        /// All allocation stubs.
        let allocStubs = ResizeArray<AllocStub> ()

        /// Allocate memory for the specified expression.
        let alloc expr req =
            let alloc = {Req = req}
            allocStubs.Add alloc
            alloc

        let filterWishes stubWishes =
            stubWishes
            |> Map.filter (fun _ ts -> TensorStub.isNotAliased ts)            

        /// Tensor stub wishes for an expression channel.
        let chStubWishes = chStubWishes |> filterWishes |> Dictionary<BaseExprCh, TensorStub> 

        /// Tensor stubs for an expression channel.
        let chStubs = Dictionary<BaseExprCh, TensorStub> ()

        /// Tensor stub wishes made by an expression for its arguments.
        let allArgWishes = Dictionary<BaseExpr, Map<Arg, TensorStub>> ()

        /// Propagates wishes from an expression to its arguments.
        let propagate (expr: BaseExpr) =
            match expr.Op with
            | :? ITensorStubWishPropagatingOp as op ->
                // Op supports wish propagation.

                // Assemble all channel wishes of an expression.
                let chWishes =
                    expr.Channels
                    |> Seq.choose (fun ch ->
                        chStubWishes.TryFind expr.[ch]
                        |> Option.map (fun wish -> ch, wish))
                    |> Map.ofSeq

                // Let op calculate wishes for its arguments.
                let comp = op.PropagateWishes  {
                    Alloc = alloc expr
                    Env = env
                    Expr = expr
                    ChStubWishes = chWishes
                }

                // Store argument stub wishes.
                allArgWishes.[expr] <- comp.ArgStubWishes

                // Store channel stubs.
                for KeyValue(ch, chStub) in comp.ChStubs do
                    let exprCh = expr.[ch]
                    if not (chStub.Dev = exprCh.Dev && 
                            chStub.TypeName = exprCh.TypeName &&
                            chStub.Shape = Shape.eval exprCh.Shape) then
                        failwithf "Tensor stub %A for channel %A is not compatiable with expression %A."
                            chStub ch exprCh    
                    chStubs.[exprCh] <- chStub

                // Propagate wishes to argument expressions.
                for KeyValue(arg, argWish) in filterWishes comp.ArgStubWishes do
                    let argExprCh = expr.Args.[arg]

                    // Check wish for plausibility.
                    if not (argWish.Dev = argExprCh.Dev && 
                            argWish.TypeName = argExprCh.TypeName &&
                            argWish.Shape = Shape.eval argExprCh.Shape) then
                        failwithf "Tensor stub wish %A for argument %A is not compatiable with %A."
                            argWish arg argExprCh

                    // The first wish for an expression channel wins and subsequent 
                    // wishes are ignored.
                    if not (chStubWishes.ContainsKey argExprCh) then
                        chStubWishes.[argExprCh] <- argWish

            | _ -> 
                // Op does not support wish propagation. Do nothing.
                ()

        // Walk over expression tree from roots to leafs and propagate wishes.
        group.RevIter propagate

        {
            Allocs = List.ofSeq allocStubs
            ChStubs =  Map.ofDictionary chStubs 
            ChStubWishes =  Map.ofDictionary chStubWishes 
            ArgStubWishes =  Map.ofDictionary allArgWishes 
        }


    /// Assigns a tensor stub to each expression channel in the expression tree.
    let assignStubs (env: CompileEnv) (upProp: UpPropResult) (group: BaseExprGroup) =

        /// All allocation stubs.
        let allocStubs = ResizeArray<AllocStub> upProp.Allocs
        /// Users of an allocation, either as argument, internally or as channel.
        let allocUsers = Dictionary<AllocStub, Set<BaseExpr>> ()
        /// Expression channels that use an allocation.
        let allocUserChs = Dictionary<AllocStub, Set<BaseExprCh>> ()

        /// Allocate memory for the specified expression.
        let alloc expr req =
            let alloc = {Req = req}
            allocUsers.Add (alloc, Set<BaseExpr> [expr])
            allocUserChs.Add (alloc, Set.empty)
            allocStubs.Add alloc
            alloc

        let addAllocUserCh allocStub exprCh =
            let users = allocUserChs.GetOrDefault allocStub Set.empty
            allocUserChs.[allocStub] <- users |> Set.add exprCh 

        /// Assigned tensor stub for each expression channel.
        let tensorStubs = Dictionary<BaseExprCh, TensorStub> ()

        /// Action group for each expression.
        let actionGroupForExpr = Dictionary<BaseExpr, ActionGroup> ()

        /// All expressions that an expression depends on transitively.
        let depending = Dictionary<BaseExpr, Set<BaseExpr>> ()

        // Store channel stubs from stub propagation towards leafs. 
        for KeyValue(exprCh, stub) in upProp.ChStubs do
            tensorStubs.[exprCh] <- stub
            match stub.Storage with
            | StorageStub.Allocated allocStub -> addAllocUserCh allocStub exprCh
            | _ -> ()

        /// Processes an expression.
        let processExpr (expr: BaseExpr) =    
            /// Stubs for the arguments of the expression.
            let argStubs = expr.Args |> Map.map (fun _ argExpr -> tensorStubs.[argExpr])

            /// All expression channels this expression transitively depends on.
            /// Due to the walk order over the expression tree all these expressions
            /// have already been processed.
            let exprDepending =
                if Map.isEmpty expr.Args then 
                    Set.empty
                else
                    expr.Args
                    |> Map.toSeq
                    |> Seq.map (fun (_arg, argExpr) -> Set.add argExpr.Expr depending.[argExpr.Expr])
                    |> Set.unionMany
            depending.[expr] <- exprDepending

            // Add expression to users of its argument stubs.
            for KeyValue(_, argStub) in argStubs do
                match argStub.Storage with
                | StorageStub.Allocated allocStub -> 
                    allocUsers.[allocStub] <- allocUsers.[allocStub] |> Set.add expr 
                | _ -> ()

            // Compute set of overwritable arguments for in-place operations.
            // An argument value can be overwritten with the result of this expression,
            // if it is guaranteed that all other expressions using the argument value are
            // finished evaluating before the evaluation of this expression starts.
            // Additionally, the argument must not share the same storage with another
            // argument.
            let overwritableArgs =
                expr.Args
                |> Map.toSeq
                |> Seq.filter (fun (arg, argExprCh) ->
                    // Arguments that share the same storage must not be overwritten.
                    expr.Args |> Map.toSeq |> Seq.exists (fun (otherArg, otherArgExprCh) ->
                        otherArg <> arg && tensorStubs.[otherArgExprCh] = tensorStubs.[argExprCh])
                    |> not)
                |> Seq.filter (fun (_arg, argExprCh) ->
                    // Only overwrite arguments that do not have aliased elements,
                    // for example due to broadcasting.
                    TensorStub.isNotAliased tensorStubs.[argExprCh])
                |> Seq.filter (fun (_arg, argExprCh) ->
                    match tensorStubs.[argExprCh].Storage with
                    | StorageStub.Allocated argAlloc ->
                        // Arguments uses an allocated storage: overwriting it might be possible.
                        // Get all expression channels that use the allocated storage of that argument (so far).
                        let exprChsUsingArgStorage = allocUserChs.[argAlloc] 
                        // Get their (complete) dependants.
                        let deps =
                            exprChsUsingArgStorage
                            |> Set.toSeq
                            |> Seq.collect group.Dependants
                            |> Seq.filter ((<>) expr)
                        // Check if we depend on all their dependants *and* that they have been processed,
                        // i.e. if they exposed the allocation via an output channel it would already have been
                        // added to allocUserChs.
                        deps |> Seq.forall exprDepending.Contains
                    | _ -> 
                        // Argument uses other storage. Do not overwrite it.
                        false)
                |> Seq.map fst
                |> Set.ofSeq
                
            // Assemble all pre-assigned channels for the expression.
            let preassignedChStubs =
                expr.Channels
                |> Seq.choose (fun ch ->
                    upProp.ChStubs.TryFind expr.[ch]
                    |> Option.map (fun wish -> ch, wish))
                |> Map.ofSeq

            // Assemble all channel wishes for the expression.
            let chWishes =
                expr.Channels
                |> Seq.choose (fun ch ->
                    upProp.ChStubWishes.TryFind expr.[ch]
                    |> Option.map (fun wish -> ch, wish))
                |> Map.ofSeq

            // Assemble all argument wishes that the expression has made.
            let argWishes =
                expr.Args
                |> Map.keys
                |> Seq.choose (fun arg ->
                    upProp.ArgStubWishes
                    |> Map.tryFind expr
                    |> Option.bind (Map.tryFind arg)
                    |> Option.map (fun wish -> arg, wish))
                |> Map.ofSeq

            // Compute channel stubs given argument stubs.
            let comp =
                match expr.Op with
                | :? ICompilableOp as cop -> 
                    // Let op compute its output stubs given argument stubs.
                    cop.Compile {
                        Alloc = alloc expr
                        Env = env
                        ArgStubs = argStubs
                        OverwritableArgs = overwritableArgs
                        ChStubs = preassignedChStubs
                        ChStubWishes = chWishes
                        ArgStubWishes = argWishes
                        Expr = expr
                    } 
                | _ ->
                    // Op cannot compute stubs, generate dynamic stubs for all
                    // its output channels. 
                    let chStubs =
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
                    let actions = [NonCompilableOpAction (expr, argStubs) :> IAction] 
                    {ChStubs=chStubs; Actions=actions}

            // Check that pre-assigned channel stubs were accepted.
            for KeyValue(ch, preassignedChStub) in preassignedChStubs do
                match comp.ChStubs |> Map.tryFind ch with
                | Some stub when stub = preassignedChStub -> ()
                | _ ->
                    failwithf "Expression %A did not accept pre-assigned channel stub %A." 
                        expr preassignedChStub

            // Check channel stubs for plausibility.
            for KeyValue(ch, chStub) in comp.ChStubs do
                let exprCh = expr.[ch]
                if not (chStub.Dev = exprCh.Dev && 
                        chStub.TypeName = exprCh.TypeName &&
                        chStub.Shape = Shape.eval exprCh.Shape) then
                    failwithf "Tensor stub %A for channel %A is not compatiable with expression %A."
                        chStub ch exprCh    

            // Store action group.
            actionGroupForExpr.[expr] <- {
                Actions = comp.Actions
                DependsOn = 
                    group.Depending expr
                    |> Seq.map (fun dep -> actionGroupForExpr.[dep])  
                    |> HashSet
            }

            // Add expression to users of its channel stubs.
            for KeyValue(ch, chStub) in comp.ChStubs do
                match chStub.Storage with
                | StorageStub.Allocated allocStub -> 
                    allocUsers.[allocStub] <- allocUsers.[allocStub] |> Set.add expr 
                    allocUserChs.[allocStub] <- allocUserChs.[allocStub] |> Set.add expr.[ch]
                | _ -> ()

            // Store channel stubs.
            for KeyValue(ch, chStub) in comp.ChStubs do
                tensorStubs.[expr.[ch]] <- chStub

        /// Called after all dependencies of an expression have been processed.
        let clearDepending (expr: BaseExpr) =
            // Clear dependency set of expression to save memory.
            depending.Remove expr |> ignore

        // Process expression tree.
        group.Iter (processExpr, allDepsOfExprEvaled=clearDepending)

        {
            Allocs = List.ofSeq allocStubs
            ChStubs = Map.ofDictionary tensorStubs
            ActionGroups = List.ofSeq actionGroupForExpr.Values
        }


[<RequireQualifiedAccess>]
type TryInplace =
    | All
    | Limited of Set<Arg>
    | None


type CompileTools () =

    /// Allocates tensor stubs for all output channels of the op.
    /// Argument stubs will be reused, if tryInplace is true.
    /// Channel wishes will be honored, if honorWishes is true.
    static member chStubs (data: CompileData, ?tryInplace: TryInplace, ?honorWishes: bool) =
        let tryInplace = defaultArg tryInplace TryInplace.None
        let honorWishes = defaultArg honorWishes true

        let op = data.Expr.Op
        let mutable overwritableArgs = data.OverwritableArgs
        
        /// Returns an overwritable argument TensorStub that matches the
        /// given specifications, if available.
        let tryUseArg typeName dev shape =
            overwritableArgs
            |> Set.toSeq
            |> Seq.filter (fun arg ->
                // check that argument may be overridden
                match tryInplace with
                | TryInplace.All -> true
                | TryInplace.Limited allowed when allowed.Contains arg -> true
                | TryInplace.Limited _allowed -> false
                | TryInplace.None -> false)
            |> Seq.tryFind (fun arg ->
                // match type, device and shape
                let argExpr = op.Args.[arg]
                argExpr.TypeName = typeName &&
                argExpr.Dev = dev &&
                Shape.eval argExpr.Shape = shape)
            |> Option.map (fun arg -> 
                // remove from set of overwritable arguments
                overwritableArgs <- overwritableArgs |> Set.remove arg
                data.ArgStubs.[arg])

        op.Channels
        |> Set.toSeq
        |> Seq.map (fun ch ->
            let typeName = op.TypeNames.[ch]
            let shape = Shape.eval op.Shapes.[ch]
            let dev = op.Devs.[ch]
            let stub = 
                match data.ChStubWishes |> Map.tryFind ch with
                | Some chStubWish when honorWishes -> chStubWish
                | _ ->
                    match tryUseArg typeName dev shape with
                    | Some inplaceStub -> inplaceStub
                    | None -> TensorStub.alloc (data.Alloc, typeName, shape, dev)
            ch, stub)
        |> Map.ofSeq

    /// Passes through tensor stub of unary argument.
    static member passthroughStub (data: CompileData) =
        Map [Ch.Default, data.ArgStubs.[Arg.Only]]

    /// Propagates a tensor stub wish for an unary operation.
    static member propUnaryWish (data: UpPropData) (fn: TensorStub -> TensorStub option)  =
        let chWishOpt = data.ChStubWishes |> Map.tryFind Ch.Default
        let argWishOpt = chWishOpt |> Option.bind fn
        match chWishOpt, argWishOpt with
        | Some chWish, Some argWish -> 
            // We had a wish for our channel stub and propagated it to our argument.
            // Thus we must commit to the wish and assign it to our channel.
            {
                ChStubs = Map [Ch.Default, chWish]
                ArgStubWishes = Map [Arg.Only, argWish]
            }
        | _ -> 
            // We had not wish or did not propagate it.
            {
                ChStubs = Map.empty
                ArgStubWishes = Map.empty
            }


    static member simpleAction (actFn: Map<Ch, ITensor> -> Map<Arg, ITensor> -> unit) =
        let action = 
            { new IAction with
                member __.Execute data =
                    actFn data.ChValues data.ArgValues 
                    Map.empty
            }
        [action]


    static member noAction () : IAction list =
        []


    static member tryStatic (data: CompileData) (staticFn: TensorStub -> TensorStub option) (dynFn: ITensor -> ITensor) =
        let op = data.Expr.Op
        let argStub = ArgValue.unaryX data.ArgStubs 

        match data.ArgStubWishes |> Map.tryFind Arg.Only with
        | Some argStubWish when argStubWish = argStub ->
            // We propagated a tensor stub as a wish to our argument and it was accepted.
            // Thus our channel stub is already assigned and no actions need to be performed.
            {
                ChStubs = data.ChStubs
                Actions = []
            }
        | Some argStubWish ->
            // We propagated a tensor stub as a wish to our argument, but it was not accepeted.
            // Thus we need to copy the arguments output to our assigned channel stub.
            {
                ChStubs = data.ChStubs
                Actions = CompileTools.simpleAction (fun chVals argVals ->
                    (ChValue.onlyX chVals).CopyFrom (ArgValue.unaryX argVals))
            }
        | None ->
            // No wish was propagated, we have to compute a channel stub.
            match staticFn argStub with
            | Some chStub -> 
                // Channel stub could be calculated at compile-time.
                {
                    ChStubs = Ch.only chStub
                    Actions = []
                }
            | None ->
                // Channel stub must be calculated at run-time.
                {
                    ChStubs = Ch.only {
                        Shape = Shape.eval op.Shapes.[Ch.Default]
                        TypeName = op.TypeNames.[Ch.Default]
                        Dev = op.Devs.[Ch.Default]
                        OffsetStride = None
                        Storage = argStub.Storage    
                    }
                    Actions = [{ new IAction with
                        member __.Execute execData =                            
                            ArgValue.unaryX execData.ArgValues
                            |> dynFn 
                            |> Ch.only
                    }]     
                }        




