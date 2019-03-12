namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr

/// specification of variable strides
type internal VarStrides = Map<Var, int64 list>

/// specification of channel strides
type internal ChannelStridesT = Map<Ch, int64 list>

/// channel information for loop execution
type internal LoopChannelInfoT = {
    Shape:          NShapeSpec
    SliceDim:       int
    Target:         ITensor
} 

/// channel layout information for building loop strides
type internal LoopChannelLayoutInfoT = {
    Shape:          NShapeSpec
    SliceDim:       int
    TargetLayout:   TensorLayout
} 


/// Elementwise interpolation using a value table.
type Loop = {
    /// number of loop iterations
    Length:     Size
    /// specifies the values of the variables used in the channel value expressions,
    /// i.e. LoopValueT.Expr
    Vars:       Map<Var, Loop.Input>   
    /// specifies the values of the loop channels
    Channels:   Map<Ch, Loop.Value>
    /// inputs
    Xs:         BaseExprCh list
} with

    /// Device on which IterationIndex and IterationsRemaining is expected.
    /// Defaults to host device, if value is not used in loop expression.
    member private this.IterIndexDev =
        this.Vars
        |> Map.tryFindKey (fun var li ->
            match li with
            | Loop.Input.IterationIndex
            | Loop.Input.IterationsRemaining -> true
            | _ -> false)
        |> Option.map Var.dev
        |> Option.defaultValue HostTensor.Dev

    /// build strides information for loop sources and targets
    static member internal buildStrides (vars: Map<Var, Loop.Input>) (args: TensorLayout list) 
                                        (channels: Map<Ch, LoopChannelLayoutInfoT>) 
                                        : VarStrides * ChannelStridesT * int list option list =

        let mutable argRequiredStrideOrder = List.replicate args.Length None

        let varStrides = 
            vars |> Map.map (fun vs li ->
                match li with
                | Loop.ConstArg idx -> 
                    args.[idx].Stride
                | Loop.SequenceArgSlice {ArgIdx=idx; SliceDim=dim} ->
                    args.[idx].Stride |> List.without dim
                | Loop.PreviousChannel {Channel=ch; InitialArg=ivIdx} ->
                    let sliceDim = channels.[ch].SliceDim
                    let chStride = channels.[ch].TargetLayout.Stride |> List.without sliceDim

                    // check that initial value has same stride as channel target
                    let ivStride = args.[ivIdx].Stride |> List.without sliceDim
                    if chStride <> ivStride then
                        // Stride mismatch. 
                        // Check that copying argument to temporary array would
                        // result in matching strides.
                        let shp = args.[ivIdx].Shape
                        let strideOrder = 
                            [0 .. shp.Length-1] |> List.swap 0 sliceDim |> List.rev
                        let ivCopyStride = 
                            TensorLayout.orderedStride args.[ivIdx].Shape strideOrder
                            |> List.without sliceDim
                        if chStride <> ivCopyStride then 
                            eprintfn "Loop stride problem:"
                            eprintfn "Channel %A:\n%A" ch channels.[ch]
                            eprintfn "Initial value layout:\n%A" args.[ivIdx]
                            eprintfn "Copy stride:    %A" ivCopyStride
                            eprintfn "Channel stride: %A" chStride
                            failwithf "channel %A slice strides %A are different from initial \
                                       value slice strides %A for loop variable %A and copying \
                                       to a temporary array would not help" 
                                      ch chStride ivStride vs
                        // request copy to C-strided temporary array
                        argRequiredStrideOrder <- 
                            argRequiredStrideOrder |> List.set ivIdx (Some strideOrder)
                    chStride
                | Loop.IterationIndex
                | Loop.IterationsRemaining -> [])

        let channelStrides =
            channels |> Map.map (fun ch lv -> lv.TargetLayout.Stride |> List.without lv.SliceDim)

        varStrides, channelStrides, argRequiredStrideOrder

    /// builds inputs and outputs for one loop iteration 
    static member internal buildInOut (nIters: int64) (iter: int64) (iterAry: ITensor) 
                                      (itersRemainingAry: ITensor) (vars: Map<Var, Loop.Input>)
                                      (args: ITensor list) (channels: Map<Ch, LoopChannelInfoT>)
                                      : VarEnv * Map<Ch, ITensor> =

        /// RngAll in all dimensions but specified one
        let rngAllBut ary dim dimSlice = 
            List.replicate (ITensor.nDims ary) Rng.All
            |> List.set dim dimSlice

        /// The slice of the channel's target for the specified iteration.
        let targetSlice ch iter =
            let dim = channels.[ch].SliceDim
            let trgtSize = channels.[ch].Target.Shape.[dim]
            // calculate offset so that last loop iteration is written into last element of
            // channel's target
            let offset = trgtSize - 1L - ((nIters - 1L) % trgtSize)
            assert ((offset + nIters - 1L) % trgtSize = trgtSize - 1L)
            let pos = (offset + iter) % trgtSize
            let slice = rngAllBut channels.[ch].Target dim (Rng.Elem pos)
            channels.[ch].Target.[slice]

        // build variable environment for value sources
        let srcVarEnv = 
            vars
            |> Map.mapKeyValue (fun vs li ->
                // get value for variable
                let value = 
                    match li with
                    | Loop.ConstArg idx -> 
                        args.[idx] 
                    | Loop.SequenceArgSlice {ArgIdx=idx; SliceDim=dim} -> 
                        let slice = rngAllBut args.[idx] dim (Rng.Elem iter)
                        args.[idx].[slice] 
                    | Loop.PreviousChannel {Channel=ch; Delay=delay; InitialArg=ivIdx} ->
                        let delay = Size.eval delay
                        let dim = channels.[ch].SliceDim
                        if channels.[ch].Target.Shape.[dim] < delay then
                            failwithf "target for channel %A has insufficient size %d for delay %d"
                                       ch channels.[ch].Target.Shape.[dim] delay
                        let prvIter = iter - delay
                        if prvIter >= 0L then targetSlice ch prvIter
                        else
                            let initialIter = args.[ivIdx].Shape.[dim] + prvIter
                            let slice = rngAllBut args.[ivIdx] dim (Rng.Elem initialIter)
                            args.[ivIdx].[slice] 
                    | Loop.IterationIndex -> iterAry
                    | Loop.IterationsRemaining -> itersRemainingAry

                // check type and shape
                if ShapeSpec.eval vs.Shape <> value.Shape then
                    failwithf "loop variable %A got value with shape %A" vs value.Shape
                if vs.DataType <> value.DataType then
                    failwithf "loop variable %A got value with data type %A" vs value.DataType
                    
                vs.Name, value)

        // slice outputs into channel targets
        let targets =
            channels |> Map.map (fun ch _ -> targetSlice ch iter)

        VarEnv srcVarEnv, targets


    interface IOp with       
        member this.Check () =
            // check that all variables are defined
            let usedVars =
                Map.toSeq this.Channels
                |> Seq.map (fun (_, lv) -> lv.Expr.Expr.Vars)
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
                    if vs.Dev <> this.IterIndexDev then
                        failwithf "Iteration index variable %A is inconsistent with another usage on device %A."
                                  vs this.IterIndexDev

        member this.Channels = 
            this.Channels |> Map.toSeq |> Seq.map fst |> Set.ofSeq
        member this.TypeNames = 
            this.Channels |> Map.map (fun _ lv -> lv.Expr.TypeName)
        member this.Devs =
            this.Channels |> Map.map (fun _ lv -> lv.Expr.Dev)
        member this.Shapes = 
            this.Channels |> Map.map (fun ch lv ->
                lv.Expr.Shape |> ShapeSpec.insertAxis lv.SliceDim this.Length)

        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _

        member this.SubstSymSizes env = 
            {this with
                Length = Size.substSymbols env this.Length
                Vars = this.Vars
                        |> Map.toSeq
                        |> Seq.map (fun (vs, li) ->
                            let vs = {vs with Shape = ShapeSpec.substSymbols env vs.Shape}
                            let li = match li with
                                     | Loop.PreviousChannel pc -> 
                                        Loop.PreviousChannel {pc with Delay = Size.substSymbols env pc.Delay}
                                     | _ -> li
                            vs, li)
                        |> Map.ofSeq
                Channels = this.Channels
                            |> Map.map (fun ch lv -> {lv with Expr = lv.Expr |> BaseExprCh.map (BaseExpr.substSymSizes env)})
            } :> _

        member this.CanEvalAllSymSizes = 
            (Size.canEval this.Length) &&
            (this.Vars |> Map.toSeq |> Seq.forall (fun (vs, li) ->
                ShapeSpec.canEval vs.Shape &&
                match li with
                | Loop.PreviousChannel pc -> Size.canEval pc.Delay
                | _ -> true)) &&
            (this.Channels |> Map.toSeq |> Seq.forall (fun (ch, lv) -> lv.Expr.Expr.CanEvalAllSymSizes))   
            
        member this.Eval evalEnv argVals = 
            let args = ArgValue.naryXs argVals
            let thisExpr = BaseExpr.ofOp this

            // iteration index variables
            let nIters = Size.eval this.Length
            let iterAry = Tensor<int64>.zeros this.IterIndexDev []
            let itersRemAry = Tensor<int64>.zeros this.IterIndexDev []

            // create channel information
            let channelInfos =
                this.Channels
                |> Map.map (fun ch lv ->
                    let sliceShp = lv.Expr.Shape |> ShapeSpec.eval
                    let targetShp = sliceShp |> List.insert lv.SliceDim nIters
                    let target = Tensor.NewOfType (targetShp, lv.Expr.DataType, HostTensor.Dev, order=RowMajor)
                    {
                        LoopChannelInfoT.Shape      = sliceShp
                        LoopChannelInfoT.SliceDim   = lv.SliceDim
                        LoopChannelInfoT.Target     = target
                    })

            // perform loop
            for iter in 0L .. nIters-1L do     
                evalEnv.Tracer.Log (TraceEvent.ForExpr (thisExpr, TraceEvent.LoopIter iter))
                   
                // set iteration indices
                iterAry.[[]] <- iter
                itersRemAry.[[]] <- nIters - iter - 1L

                // calculate and store channel values
                let iterVarEnv, iterChannelEnv =
                    Loop.buildInOut nIters iter iterAry itersRemAry this.Vars args channelInfos
                for KeyValue(ch, lv) in this.Channels do
                    let iterEvalEnv = {evalEnv with VarEnv=iterVarEnv; Tracer=evalEnv.Tracer.GetSubTracer()}
                    iterEvalEnv.Tracer.Log (TraceEvent.ParentExpr thisExpr)
                    let lvVal = BaseExprEval.eval iterEvalEnv lv.Expr.Expr
                    iterChannelEnv.[ch].[Fill] <- lvVal.[Ch.Default]

            // return outputs
            channelInfos |> Map.map (fun ch ci -> ci.Target)   

    static member internal noLift length vars channels xs =
        BaseExpr.ofOp {Loop.Length=length; Vars=vars; Channels=channels; Xs=xs} 

    static member internal withLift (length: Size) (vars: Map<Var, Loop.Input>) 
                                    (channels: Map<Ch, Loop.Value>) (xs: BaseExprCh list) =       
        let mutable args = xs
        let mutable vars = vars

        /// adds an argument and returns its index
        let addArg (expr: BaseExprCh) =
            match args |> List.tryFindIndex ((=) expr) with
            | Some argIdx -> argIdx
            | None ->
                let argIdx = args.Length
                args <- args @ [expr]
                argIdx

        /// adds a constant variable, its required argument and returns the associated VarSpecT
        let addConstVar (expr: BaseExprCh) =
            let var = 
                vars |> Map.tryFindKey (fun vs lv ->
                    match lv with
                    | Loop.ConstArg argIdx when args.[argIdx] = expr -> true
                    | _ -> false) 
            match var with
            | Some vs -> vs
            | None ->
                let rec genName i =
                    let name = VarName (ContextPath.root / "LoopConst" / sprintf "%d" i)
                    match vars |> Map.tryFindKey (fun vs _ -> vs.Name = name) with
                    | Some _ -> genName (i + 1)
                    | None -> name
                let vs = Var.make (genName 0, expr.DataType, expr.Dev, expr.Shape)
                let lv = Loop.ConstArg (addArg expr)
                vars <- vars |> Map.add vs lv
                vs

        let loopVarSet = vars |> Map.toSeq |> Seq.map (fun (vs, _) -> vs) |> Set.ofSeq
        let lifted = Dictionary<BaseExprCh, BaseExprCh> ()

        let rec lift (expr: BaseExprCh) : BaseExprCh =
            match lifted.TryFind expr with
            | Some rep -> rep 
            | None ->
                let exprVars = expr.Expr.Vars 
                let dependsOnVars = not (Set.isEmpty exprVars)
                let dependsOnLoopVars = Set.intersect exprVars loopVarSet |> Set.isEmpty |> not
                let rep =
                    if dependsOnVars && not dependsOnLoopVars then
                        //if not (dependsOnLoopVars expr) then
                        let vs = addConstVar expr
                        let repExpr = BaseExpr.ofOp {VarArg.Var=vs} 
                        repExpr.[Ch.Default]
                    else
                        expr |> BaseExprCh.map (BaseExpr.mapArgs lift)
                lifted.[expr] <- rep
                rep 
                
        // lift constants out of loop
        let liftedChannels = 
            channels 
            |> Map.map (fun ch lv -> {
                lv with Loop.Value.Expr = lift lv.Expr 
            })
        BaseExpr.ofOp {Loop.Length=length; Vars=vars; Channels=liftedChannels; Xs=args} 
       

 