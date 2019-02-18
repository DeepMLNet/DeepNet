namespace SymTensor.Ops

open DeepNet.Utils
open SymTensor



/// Elementwise interpolation using a value table.
type Loop = {
    /// number of loop iterations
    Length:     SizeSpec
    /// specifies the values of the variables used in the channel value expressions,
    /// i.e. LoopValueT.Expr
    Vars:       Map<Var, Loop.Input>   
    /// specifies the values of the loop channels
    Channels:   Map<string, Loop.Value>
    /// inputs
    Xs:         BaseExpr list
} with
    interface IMultiChannelOp with       
        member this.Check () =
            // check that all variables are defined
            let usedVars =
                Map.toSeq this.Channels
                |> Seq.map (fun (_, lv) -> lv.Expr.Vars)
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
                | Loop.ConstArg idx -> 
                    checkArg idx
                    if this.Xs.[idx].TypeName <> vs.TypeName then
                        failwithf "Constant argument variable %A was given argument of type %A." vs this.Xs.[idx].DataType
                    if not (ShapeSpec.equalWithoutBroadcastability vs.Shape this.Xs.[idx].Shape) then
                        failwithf "Constant argument variable %A was given argument of shape %A." vs this.Xs.[idx].Shape
                | Loop.SequenceArgSlice {ArgIdx=idx; SliceDim=dim} ->
                    checkArg idx
                    if this.Xs.[idx].TypeName <> vs.TypeName then
                        failwithf "Sequence argument variable %A was given argument of type %A." vs this.Xs.[idx].DataType
                    let reqShp = vs.Shape |> ShapeSpec.insertAxis dim this.Length
                    if not (ShapeSpec.equalWithoutBroadcastability reqShp this.Xs.[idx].Shape) then
                        failwithf "Sequence argument variable %A requires argument shape %A but was given %A." 
                                    vs reqShp this.Xs.[idx].Shape
                | Loop.PreviousChannel {Channel=prvCh; Delay=delay; InitialArg=ivIdx} ->
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
                | Loop.IterationIndex 
                | Loop.IterationsRemaining -> 
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
                                     | Loop.PreviousChannel pc -> 
                                        Loop.PreviousChannel {pc with Delay = SizeSpec.substSymbols env pc.Delay}
                                     | _ -> li
                            vs, li)
                        |> Map.ofSeq
                Channels = this.Channels
                            |> Map.map (fun ch lv -> {lv with Expr = lv.Expr |> BaseExpr.substSymSizes env})
            } :> _
        member this.CanEvalAllSymSizes = 
            (SizeSpec.canEval this.Length) &&
            (this.Vars |> Map.toSeq |> Seq.forall (fun (vs, li) ->
                ShapeSpec.canEval vs.Shape &&
                match li with
                | Loop.PreviousChannel pc -> SizeSpec.canEval pc.Delay
                | _ -> true)) &&
            (this.Channels |> Map.toSeq |> Seq.forall (fun (ch, lv) -> lv.Expr.CanEvalAllSymSizes))    
        member this.Eval env = 
            failwith "TODO"

    static member internal noLift length vars channels xs =
        BaseMultiChannelExpr.ofOp {Loop.Length=length; Vars=vars; Channels=channels; Xs=xs} 

    static member internal withLift (length: SizeSpec) (vars: Map<Var, Loop.Input>) 
                                    (channels: Map<string, Loop.Value>) (xs: BaseExpr list) =       
        let mutable args = xs
        let mutable vars = vars

        /// adds an argument and returns its index
        let addArg (expr: BaseExpr) =
            match args |> List.tryFindIndex ((=) expr) with
            | Some argIdx -> argIdx
            | None ->
                let argIdx = args.Length
                args <- args @ [expr]
                argIdx

        /// adds a constant variable, its required argument and returns the associated VarSpecT
        let addConstVar (expr: BaseExpr) =
            let var = 
                vars |> Map.tryFindKey (fun vs lv ->
                    match lv with
                    | Loop.ConstArg argIdx when args.[argIdx] = expr -> true
                    | _ -> false) 
            match var with
            | Some vs -> vs
            | None ->
                let rec genName i =
                    let name = sprintf "CONST%d" i
                    match vars |> Map.tryFindKey (fun vs _ -> vs.Name = name) with
                    | Some _ -> genName (i + 1)
                    | None -> name
                let vs = Var.ofNameShapeAndTypeName (genName 0) expr.Shape expr.TypeName
                let lv = Loop.ConstArg (addArg expr)
                vars <- vars |> Map.add vs lv
                vs

        let loopVarSet = vars |> Map.toSeq |> Seq.map (fun (vs, _) -> vs) |> Set.ofSeq
        let lifted = Dictionary<BaseExpr, BaseExpr> ()

        let rec lift (xChExpr: BaseXChExpr) : BaseXChExpr =
            match xChExpr with
            | BaseXChExpr.SingleCh expr ->
                match lifted.TryFind expr with
                | Some rep -> rep |> BaseXChExpr.SingleCh
                | None ->
                    let exprVars = expr.Vars
                    let dependsOnVars = not (Set.isEmpty exprVars)
                    let dependsOnLoopVars = Set.intersect exprVars loopVarSet |> Set.isEmpty |> not
                    let rep =
                        if dependsOnVars && not dependsOnLoopVars then
                            //if not (dependsOnLoopVars expr) then
                            let vs = addConstVar expr
                            BaseExpr.ofOp {VarArg.Var=vs} 
                        else
                            expr |> BaseExpr.mapArgs lift
                    lifted.[expr] <- rep
                    rep |> BaseXChExpr.SingleCh
            | BaseXChExpr.MultiCh mChExpr ->
                mChExpr |> BaseMultiChannelExpr.mapArgs lift |> BaseXChExpr.MultiCh            
                
        // lift constants out of loop
        let liftedChannels = 
            channels 
            |> Map.map (fun ch lv -> {
                lv with Loop.Value.Expr = lv.Expr |> BaseXChExpr.SingleCh |> lift |> BaseXChExpr.singleCh
            })
        BaseMultiChannelExpr.ofOp {Loop.Length=length; Vars=vars; Channels=liftedChannels; Xs=args} 
       

 