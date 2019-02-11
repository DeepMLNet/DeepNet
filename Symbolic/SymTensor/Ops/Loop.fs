namespace SymTensor.Ops

open SymTensor


module LoopOps = 

    /// a slice of an argument to the loop
    type SequenceArgSlice = {
        /// the index of the argument
        ArgIdx:     int
        /// the dimension the loop is performed over
        SliceDim:   int
    }

    /// references a loop channel of a previous iteration
    type PreviousChannel = {
        /// the channel to use
        Channel:       string
        /// the delay, must be at least one
        Delay:         SizeSpec
        /// the index of the argument specifying the initial values
        InitialArg:    int
    }

    /// a loop variable value specification
    type LoopInput = 
        /// provides the loop argument to all loop iterations
        | ConstArg of argIdx:int
        /// provides a slice of the loop argument to each loop iteration
        | SequenceArgSlice of SequenceArgSlice
        /// provides the value of a loop channel from a previous loop iteration
        | PreviousChannel of PreviousChannel
        /// provides the index of the current loop iteration (zero-based)
        | IterationIndex
        /// provides the number of remaining loop iterations after this iteration
        | IterationsRemaining

    /// the value of a loop channel
    type LoopValue = {
        /// the expression to compute the loop channel;
        /// it may only use variables defined in LoopSpecT.Vars
        Expr:       Expr2
        /// the dimension to concatenate the results along to produce the loop output
        SliceDim:   int
    }


    /// Elementwise interpolation using a value table.
    type Loop = {
        /// number of loop iterations
        Length:     SizeSpec
        /// specifies the values of the variables used in the channel value expressions,
        /// i.e. LoopValueT.Expr
        Vars:       Map<Var, LoopInput>   
        /// specifies the values of the loop channels
        Channels:   Map<string, LoopValue>
        /// inputs
        Xs:         Expr2 list
    } with
        interface IMultiChannelOp with       
            member this.Check () =
                // check that all variables are defined
                let usedVars =
                    Map.toSeq this.Channels
                    |> Seq.map (fun (_, lv) -> Expr2.vars lv.Expr)
                    |> Set.unionMany
                let specifiedVars = 
                    Map.toSeq this.Vars
                    |> Seq.map (fun (var, _) -> var)
                    |> Set.ofSeq
                if not (Set.isEmpty (usedVars - specifiedVars)) then
                    failwithf "The variables %A were used in the loop but not defined."
                              (usedVars - specifiedVars)

                // check that shapes of loop variables are correct and referenced arguments exist
                let checkArg idx =
                    if not (0 <= idx && idx < this.Xs.Length) then
                        failwithf "The zero-based index %d does not exist for %d specified arguments." idx this.Xs.Length
                for KeyValue(vs, li) in this.Vars do
                    match li with
                    | ConstArg idx -> 
                        checkArg idx
                        if this.Xs.[idx].TypeName <> vs.TypeName then
                            failwithf "Constant argument variable %A was given argument of type %A." vs this.Xs.[idx].DataType
                        if not (ShapeSpec.equalWithoutBroadcastability vs.Shape this.Xs.[idx].Shape) then
                            failwithf "Constant argument variable %A was given argument of shape %A." vs this.Xs.[idx].Shape
                    | SequenceArgSlice {ArgIdx=idx; SliceDim=dim} ->
                        checkArg idx
                        if this.Xs.[idx].TypeName <> vs.TypeName then
                            failwithf "Sequence argument variable %A was given argument of type %A." vs this.Xs.[idx].DataType
                        let reqShp = vs.Shape |> ShapeSpec.insertAxis dim this.Length
                        if not (ShapeSpec.equalWithoutBroadcastability reqShp this.Xs.[idx].Shape) then
                            failwithf "Sequence argument variable %A requires argument shape %A but was given %A." 
                                      vs reqShp this.Xs.[idx].Shape
                    | PreviousChannel {Channel=prvCh; Delay=delay; InitialArg=ivIdx} ->
                        // check previous channel
                        match this.Channels |> Map.tryFind prvCh with
                        | Some chVal -> 
                            if vs.TypeName <> chVal.Expr.TypeName then
                                failwithf "Previous channel variable %A was given channel of type %A." vs chVal.Expr.DataType
                            if not (ShapeSpec.equalWithoutBroadcastability chVal.Expr.Shape vs.Shape) then
                                failwithf "Previous channel variable %A was given channel of shape %A." vs chVal.Expr.Shape                                
                        | None -> 
                            failwithf "Previous channel %A for variable %A does not exist." prvCh vs
                            
                        // check initial value arg
                        checkArg ivIdx
                        if this.Xs.[ivIdx].TypeName <> vs.TypeName then
                            failwithf "Previous channel variable %A was given initial value of type %A" 
                                      vs this.Xs.[ivIdx].DataType
                        let sliceDim = this.Channels.[prvCh].SliceDim
                        let reqShp = vs.Shape |> ShapeSpec.insertAxis sliceDim delay
                        if not (ShapeSpec.equalWithoutBroadcastability reqShp this.Xs.[ivIdx].Shape) then
                            failwithf "Previous channel variable %A needs initial value of shape %A but was given %A." 
                                      vs reqShp this.Xs.[ivIdx].Shape
                    | IterationIndex 
                    | IterationsRemaining -> 
                        if vs.TypeName <> TypeName.ofType<int> then
                            failwithf "Iteration index variable %A must be of type int." vs
                        if not (ShapeSpec.equalWithoutBroadcastability vs.Shape []) then
                            failwithf "Iteration index variable %A must be scalar." vs
            member this.Channels = 
                this.Channels |> Map.toList |> List.map fst
            member this.TypeNames = 
                this.Channels |> Map.map (fun _ lv -> lv.Expr.TypeName)
            member this.Shapes = 
                this.Channels |> Map.map (fun ch lv ->
                    lv.Expr.Shape |> ShapeSpec.insertAxis lv.SliceDim this.Length)
            member this.Args = Args.nary this.Xs
            member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
            member this.SubstSymSizes env = 
                {this with
                    Length = SizeSpec.substSymbols env this.Length
                    Vars = this.Vars
                            |> Map.toSeq
                            |> Seq.map (fun (vs, li) ->
                                let vs = {vs with Shape = ShapeSpec.substSymbols env vs.Shape}
                                let li = match li with
                                         | PreviousChannel pc -> 
                                            PreviousChannel {pc with Delay = SizeSpec.substSymbols env pc.Delay}
                                         | _ -> li
                                vs, li)
                            |> Map.ofSeq
                    Channels = this.Channels
                                |> Map.map (fun ch lv -> {lv with Expr = Expr2.substSymSizes env lv.Expr})
                } :> _
            member this.CanEvalAllSymSizes = 
                (SizeSpec.canEval this.Length) &&
                (this.Vars |> Map.toSeq |> Seq.forall (fun (vs, li) ->
                    ShapeSpec.canEval vs.Shape &&
                    match li with
                    | PreviousChannel pc -> SizeSpec.canEval pc.Delay
                    | _ -> true)) &&
                (this.Channels |> Map.toSeq |> Seq.forall (fun (ch, lv) -> Expr2.canEvalAllSymSizes lv.Expr))    
            member this.Deriv dOp = failwith "TODO" // TODO
            member this.Eval env = 
                failwith "TODO"
    let (|Loop|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Loop as this -> Some this
        | _ -> None

    /// A loop provides iterative evaluation of one or multiple expresisons.
    /// All variables occurs in the loop channel expressions must be defined as loop variables.
    /// The function `loop` performs automatic lifting of constants and thus allows for easy
    /// usage of variables external to the loop.
    let loopNoLift length vars channels xs =
        {Loop.Length=length; Vars=vars; Channels=channels; Xs=xs} |> MultiChannelExpr

    /// A loop provides iterative evaluation of one or multiple expresisons.
    let loop length vars channels xs =       
        let mutable args = xs
        let mutable vars = vars

        /// adds an argument and returns its index
        let addArg (expr: Expr2) =
            match args |> List.tryFindIndex ((=) expr) with
            | Some argIdx -> argIdx
            | None ->
                let argIdx = args.Length
                args <- args @ [expr]
                argIdx

        /// adds a constant variable, its required argument and returns the associated VarSpecT
        let addConstVar (expr: Expr2) =
            match vars |> Map.tryFindKey (fun vs lv ->
                                           match lv with
                                           | ConstArg argIdx when args.[argIdx] = expr -> true
                                           | _ -> false) with
            | Some vs -> vs
            | None ->
                let rec genName i =
                    let name = sprintf "CONST%d" i
                    match vars |> Map.tryFindKey (fun vs _ -> vs.Name = name) with
                    | Some _ -> genName (i + 1)
                    | None -> name
                let vs = Var.ofNameShapeAndTypeName (genName 0) expr.Shape expr.TypeName
                let lv = ConstArg (addArg expr)
                vars <- vars |> Map.add vs lv
                vs

        let loopVarSet = vars |> Map.toSeq |> Seq.map (fun (vs, _) -> vs) |> Set.ofSeq
        let lifted = Dictionary<Expr2, Expr2> ()

        let rec lift expr =
            match lifted.TryFind expr with
            | Some rep -> rep
            | None ->
                let exprVars = Expr2.vars expr
                let dependsOnVars = not (Set.isEmpty exprVars)
                let dependsOnLoopVars = Set.intersect exprVars loopVarSet |> Set.isEmpty |> not
                let rep =
                    if dependsOnVars && not dependsOnLoopVars then
                        //if not (dependsOnLoopVars expr) then
                        let vs = addConstVar expr
                        {VarArg.Var=vs} |> Expr2
                    else
                        failwith "TODO"
                        //match expr with                   
                        //| Unary (Gather indices, a) ->
                        //    Unary (Gather (indices |> List.map (Option.map lift)), lift a)
                        //| Unary (Scatter (indices, trgtShp), a) ->
                        //    Unary (Scatter (indices |> List.map (Option.map lift), trgtShp), lift a)
                        //| Binary (IfThenElse cond, a, b) ->
                        //    Binary (IfThenElse (lift cond), lift a, lift b)

                        //| Leaf _ -> expr
                        //| Unary (op, a) -> Unary (op, lift a)
                        //| Binary (op, a, b) -> Binary (op, lift a, lift b)
                        //| Nary (op, es) -> Nary (op, es |> List.map lift)
                lifted.[expr] <- rep
                rep
                
        // lift constants out of loop
        let liftedChannels = channels |> Map.map (fun ch lv -> {lv with Expr = lift lv.Expr})
        {Loop.Length=length; Vars=vars; Channels=liftedChannels; Xs=args} |> MultiChannelExpr            


