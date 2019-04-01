namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr


/// Types for internal loop representation. 
module Loop =

    /// a slice of an argument to the loop
    type InputSlice = {
        /// the index of the argument
        ArgIdx:     int
        /// the dimension the loop is performed over
        SliceDim:   int
    }

    /// references a loop channel of a previous iteration
    type PrevCh = {
        /// the channel to use
        Channel:       Ch
        /// the delay, must be at least one
        Delay:         Size
        /// the index of the argument specifying the initial values
        InitialArg:    int
    }

    /// a loop variable value specification
    [<RequireQualifiedAccess>]
    type LoopVar = 
        /// provides the loop argument to all loop iterations
        | Input of argIdx:int
        /// provides a slice of the loop argument to each loop iteration
        | InputSlice of InputSlice
        /// provides the value of a loop channel from a previous loop iteration
        | PrevCh of PrevCh
        /// provides the index of the current loop iteration (zero-based)
        | IterIdx
        /// provides the number of remaining loop iterations after this iteration
        | IterRem

    /// the value of a loop channel
    type ChValue = {
        /// the expression to compute the loop channel;
        /// it may only use variables defined in LoopSpecT.Vars
        Expr:       BaseExprCh
        /// the dimension to concatenate the results along to produce the loop output
        SliceDim:   int
    }

    /// specification of variable strides
    type internal VarStrides = Map<Var, int64 list>

    /// specification of channel strides
    type internal ChStrides = Map<Ch, int64 list>

    /// channel information for loop execution
    type internal ChInfo = {
        Shape:          int64 list
        SliceDim:       int
        Target:         ITensor
    } 

    /// channel layout information for building loop strides
    type internal ChLayoutInfo = {
        Shape:          int64 list
        SliceDim:       int
        TargetLayout:   TensorLayout
    } 



/// An argument within a loop.
[<RequireQualifiedAccess>]
type LoopArg =
    /// same value for all loop iterations
    | Input of expr:BaseExprCh
    /// slice of the expression to each loop iteration
    | InputSlice of expr:BaseExprCh * sliceDim:int 
    /// the value of a loop channel from a previous loop iteration
    | PrevCh of ch:Ch * initial:BaseExprCh * sliceDim:int
    /// the index of the current loop iteration (zero-based)
    | IterIdx of dev:ITensorDevice
    /// the number of remaining loop iterations after this iteration
    | IterRem of dev:ITensorDevice

    interface IOp with
        member this.Check () = 
            match this with
            | Input expr -> ()
            | InputSlice (expr, sliceDim) -> 
                if not (0 <= sliceDim && sliceDim < expr.NDims) then
                    failwithf "Slice dimension %d is out of range for input slice expression with %d dimensions."
                        sliceDim expr.NDims
            | PrevCh (ch, initial, sliceDim) -> 
                if not (0 <= sliceDim && sliceDim < initial.NDims) then
                    failwithf "Slice dimension %d is out of range for initial expression with %d dimensions."
                        sliceDim initial.NDims
            | IterIdx dev -> ()
            | IterRem dev -> ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = 
            match this with
            | Input expr -> expr.TypeName
            | InputSlice (expr, sliceDim) -> expr.TypeName
            | PrevCh (ch, initial, sliceDim) -> initial.TypeName
            | IterIdx dev -> TypeName.ofType<int64>
            | IterRem dev -> TypeName.ofType<int64>
            |> Ch.only
        member this.Devs = 
            match this with
            | Input expr -> expr.Dev
            | InputSlice (expr, sliceDim) -> expr.Dev
            | PrevCh (ch, initial, sliceDim) -> initial.Dev
            | IterIdx dev -> dev
            | IterRem dev -> dev
            |> Ch.only
        member this.Shapes = 
           match this with
            | Input expr -> expr.Shape
            | InputSlice (expr, sliceDim) -> expr.Shape |> Shape.withoutAxis sliceDim
            | PrevCh (ch, initial, sliceDim) -> initial.Shape |> Shape.withoutAxis sliceDim
            | IterIdx dev -> Shape.scalar
            | IterRem dev -> Shape.scalar
            |> Ch.only
        member this.Args = 
          match this with
            | Input expr -> Args.unary expr
            | InputSlice (expr, sliceDim) -> Args.unary expr
            | PrevCh (ch, initial, sliceDim)-> Args.unary initial
            | IterIdx dev -> Args.leaf
            | IterRem dev -> Args.leaf
        member this.ReplaceArgs args = 
          match this with
            | Input expr -> Input (Args.unaryX args)
            | InputSlice (expr, sliceDim) ->  InputSlice (Args.unaryX args, sliceDim)
            | PrevCh (ch, initial, sliceDim)-> PrevCh (ch, Args.unaryX args, sliceDim)
            | IterIdx dev -> IterIdx dev
            | IterRem dev -> IterRem dev
            :> _
        member this.SubstSymSizes env = 
          match this with
            | Input expr -> Input expr
            | InputSlice (expr, sliceDim) ->  InputSlice (expr, sliceDim)
            | PrevCh (ch, initial, sliceDim) -> PrevCh (ch, initial, sliceDim)
            | IterIdx dev -> IterIdx dev
            | IterRem dev -> IterRem dev
            :> _
        member this.CanEvalAllSymSizes = 
          match this with
            | Input expr -> true
            | InputSlice (expr, sliceDim) ->  true
            | PrevCh (ch, initial, sliceDim) -> true
            | IterIdx dev -> true
            | IterRem dev -> true
        member this.Eval env argVals = 
            failwith "LoopArg must be used within a loop."

    interface IOpFormat with
        member this.Text =
            sprintf "%A" this


/// A loop.
type Loop = {
    /// number of loop iterations
    Length:     Size
    /// specifies the values of the variables used in the channel value expressions,
    /// i.e. LoopValueT.Expr
    Vars:       Map<Var, Loop.LoopVar>   
    /// specifies the values of the loop channels
    Channels:   Map<Ch, Loop.ChValue>
    /// inputs
    Xs:         BaseExprCh list
} with

    /// Device on which IterationIndex and IterationsRemaining is expected.
    /// Defaults to host device, if value is not used in loop expression.
    member private this.IterIndexDev =
        this.Vars
        |> Map.tryFindKey (fun var li ->
            match li with
            | Loop.LoopVar.IterIdx
            | Loop.LoopVar.IterRem -> true
            | _ -> false)
        |> Option.map Var.dev
        |> Option.defaultValue HostTensor.Dev

    /// build strides information for loop sources and targets
    static member internal buildStrides (vars: Map<Var, Loop.LoopVar>) (args: TensorLayout list) 
                                        (channels: Map<Ch, Loop.ChLayoutInfo>) 
                                        : Loop.VarStrides * Loop.ChStrides * int list option list =

        let mutable argRequiredStrideOrder = List.replicate args.Length None

        let varStrides = 
            vars |> Map.map (fun vs li ->
                match li with
                | Loop.LoopVar.Input idx -> 
                    args.[idx].Stride
                | Loop.LoopVar.InputSlice {ArgIdx=idx; SliceDim=dim} ->
                    args.[idx].Stride |> List.without dim
                | Loop.LoopVar.PrevCh {Channel=ch; InitialArg=ivIdx} ->
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
                | Loop.LoopVar.IterIdx
                | Loop.LoopVar.IterRem -> [])

        let channelStrides =
            channels |> Map.map (fun ch lv -> lv.TargetLayout.Stride |> List.without lv.SliceDim)

        varStrides, channelStrides, argRequiredStrideOrder

    /// builds inputs and outputs for one loop iteration 
    static member internal buildInOut (nIters: int64) (iter: int64) (iterAry: ITensor) 
                                      (itersRemainingAry: ITensor) (vars: Map<Var, Loop.LoopVar>)
                                      (args: ITensor list) (channels: Map<Ch, Loop.ChInfo>)
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
                    | Loop.LoopVar.Input idx -> 
                        args.[idx] 
                    | Loop.LoopVar.InputSlice {ArgIdx=idx; SliceDim=dim} -> 
                        let slice = rngAllBut args.[idx] dim (Rng.Elem iter)
                        args.[idx].[slice] 
                    | Loop.LoopVar.PrevCh {Channel=ch; Delay=delay; InitialArg=ivIdx} ->
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
                    | Loop.LoopVar.IterIdx -> iterAry
                    | Loop.LoopVar.IterRem -> itersRemainingAry

                // check type and shape
                if Shape.eval vs.Shape <> value.Shape then
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
                | Loop.LoopVar.Input idx -> 
                    checkArg idx
                    if this.Xs.[idx].TypeName <> vs.TypeName then
                        failwithf "Constant argument variable %A was given argument of type %A." vs this.Xs.[idx].DataType
                    if not (Shape.equalIgnoringBc vs.Shape this.Xs.[idx].Shape) then
                        failwithf "Constant argument variable %A was given argument of shape %A." vs this.Xs.[idx].Shape
                | Loop.LoopVar.InputSlice {ArgIdx=idx; SliceDim=dim} ->
                    checkArg idx
                    if this.Xs.[idx].TypeName <> vs.TypeName then
                        failwithf "Sequence argument variable %A was given argument of type %A." vs this.Xs.[idx].DataType
                    let reqShp = vs.Shape |> Shape.insertAxis dim this.Length
                    if not (Shape.equalIgnoringBc reqShp this.Xs.[idx].Shape) then
                        failwithf "Sequence argument variable %A requires argument shape %A but was given %A." 
                                    vs reqShp this.Xs.[idx].Shape
                | Loop.LoopVar.PrevCh {Channel=prvCh; Delay=delay; InitialArg=ivIdx} ->
                    // check previous channel
                    match this.Channels |> Map.tryFind prvCh with
                    | Some chVal -> 
                        if vs.TypeName <> chVal.Expr.TypeName then
                            failwithf "Previous channel variable %A was given channel of type %A." vs chVal.Expr.DataType
                        if not (Shape.equalIgnoringBc chVal.Expr.Shape vs.Shape) then
                            failwithf "Previous channel variable %A was given channel of shape %A." vs chVal.Expr.Shape                                
                    | None -> 
                        failwithf "Previous channel %A for variable %A does not exist." prvCh vs
                            
                    // check initial value arg
                    checkArg ivIdx
                    if this.Xs.[ivIdx].TypeName <> vs.TypeName then
                        failwithf "Previous channel variable %A was given initial value of type %A" 
                                    vs this.Xs.[ivIdx].DataType
                    let sliceDim = this.Channels.[prvCh].SliceDim
                    let reqShp = vs.Shape |> Shape.insertAxis sliceDim delay
                    if not (Shape.equalIgnoringBc reqShp this.Xs.[ivIdx].Shape) then
                        failwithf "Previous channel variable %A needs initial value of shape %A but was given %A." 
                                    vs reqShp this.Xs.[ivIdx].Shape
                | Loop.LoopVar.IterIdx 
                | Loop.LoopVar.IterRem -> 
                    if vs.TypeName <> TypeName.ofType<int> then
                        failwithf "Iteration index variable %A must be of type int." vs
                    if not (Shape.equalIgnoringBc vs.Shape []) then
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
                lv.Expr.Shape |> Shape.insertAxis lv.SliceDim this.Length)

        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _

        member this.SubstSymSizes env = 
            {this with
                Length = Size.subst env this.Length
                Vars = this.Vars
                        |> Map.toSeq
                        |> Seq.map (fun (vs, li) ->
                            let vs = {vs with Shape = Shape.subst env vs.Shape}
                            let li = 
                                match li with
                                | Loop.LoopVar.PrevCh pc -> 
                                    Loop.LoopVar.PrevCh {pc with Delay = Size.subst env pc.Delay}
                                | _ -> li
                            vs, li)
                        |> Map.ofSeq
                Channels = this.Channels
                            |> Map.map (fun ch lv -> {lv with Expr = lv.Expr |> BaseExprCh.map (BaseExpr.substSymSizes env)})
            } :> _

        member this.CanEvalAllSymSizes = 
            (Size.canEval this.Length) &&
            (this.Vars |> Map.toSeq |> Seq.forall (fun (vs, li) ->
                Shape.canEval vs.Shape &&
                match li with
                | Loop.LoopVar.PrevCh pc -> Size.canEval pc.Delay
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
                    let sliceShp = lv.Expr.Shape |> Shape.eval
                    let targetShp = sliceShp |> List.insert lv.SliceDim nIters
                    let target = Tensor.NewOfType (targetShp, lv.Expr.DataType, lv.Expr.Dev, order=RowMajor)
                    {
                        Loop.ChInfo.Shape      = sliceShp
                        Loop.ChInfo.SliceDim   = lv.SliceDim
                        Loop.ChInfo.Target     = target
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

    interface IMultiChannelOp

    interface IOpFormat with
        member this.Text =
            sprintf "Loop %A" this

    /// Constructs a loop using a variable substituion table.
    static member make length vars channels xs =
        BaseExpr.ofOp {Loop.Length=length; Vars=vars; Channels=channels; Xs=xs} 


    /// Constructs a loop from channel expressions containing LoopArgs.
    static member fromLoopArgExpr (length: Size) (channels: Map<Ch, BaseExprCh * int>) =  
        let argVars = Dictionary<LoopArg, Var> ()
        let loopArgs = ResizeArray<BaseExprCh> ()
        let loopInput = Dictionary<Var, Loop.LoopVar> ()

        /// Returns a variable for a LoopArg.
        let makeVar loopArg =
            let varIdx = loopInput.Count
            let varName = VarName (ContextPath.root / "_Loop" / (sprintf "%d" varIdx))
            let var, inp =
                match loopArg with
                | LoopArg.Input expr ->
                    let argIdx = loopArgs.Count
                    loopArgs.Add expr
                    let var = Var.make (varName, expr.DataType, expr.Dev, expr.Shape)
                    let inp = Loop.LoopVar.Input argIdx
                    var, inp
                | LoopArg.InputSlice (expr, sliceDim) ->
                    let argIdx = loopArgs.Count
                    loopArgs.Add expr
                    let shape = expr.Shape |> Shape.withoutAxis sliceDim
                    let var = Var.make (varName, expr.DataType, expr.Dev, shape)
                    let inp = Loop.LoopVar.InputSlice {ArgIdx=argIdx; SliceDim=sliceDim}
                    var, inp
                | LoopArg.PrevCh (ch, initial, sliceDim) ->
                    match channels.TryFind ch with
                    | Some (_chExpr, chSliceDim) when chSliceDim <> sliceDim ->
                        failwithf "Channel %A slice dimension %d is inconsistent with LoopArg.PrevCh 
                                   slice dimension %d." ch chSliceDim sliceDim
                    | None ->
                        failwithf "LoopArg.PrevCh %A does not exist within loop." ch
                    | _ -> ()
                    let argIdx = loopArgs.Count
                    loopArgs.Add initial
                    let delay = initial.Shape.[sliceDim]
                    let shape = initial.Shape |> Shape.withoutAxis sliceDim
                    let var = Var.make (varName, initial.DataType, initial.Dev, shape)
                    let inp = Loop.LoopVar.PrevCh {Channel=ch; Delay=delay; InitialArg=argIdx}
                    var, inp
                | LoopArg.IterIdx dev ->
                    let var = Var.make (varName, typeof<int64>, dev, Shape.scalar)
                    let inp = Loop.LoopVar.IterIdx
                    var, inp
                | LoopArg.IterRem dev ->
                    let var = Var.make (varName, typeof<int64>, dev, Shape.scalar)
                    let inp = Loop.LoopVar.IterRem
                    var, inp
            argVars.Add (loopArg, var)
            loopInput.Add (var, inp)  
            BaseExpr.ofOp {VarArg.Var=var}

        // Replace LoopArgs with variables in all channels.
        let processed = Dictionary<BaseExprCh, BaseExprCh> ()
        let rec processLoopArgs (exprCh: BaseExprCh) : BaseExprCh =
            processed.GetOrAdd exprCh (fun _ ->
                match exprCh.Expr.Op with
                | :? LoopArg as loopArg -> (makeVar loopArg).[Ch.Default]
                | _ -> exprCh |> BaseExprCh.map (BaseExpr.mapArgs processLoopArgs)           
            )
        let channels = 
            channels 
            |> Map.map (fun _ (expr, sliceDim) -> processLoopArgs expr, sliceDim)

        // Lift expression parts not depending on loop variables out of loop.
        let loopVarSet = Set loopInput.Keys
        let processed = Dictionary<BaseExprCh, BaseExprCh> ()
        let rec liftConstants (exprCh: BaseExprCh) : BaseExprCh =
            processed.GetOrAdd exprCh (fun _ ->
                let exprVars = exprCh.Expr.Vars 
                let indepOfLoopVars = Set.isEmpty (Set.intersect exprVars loopVarSet)
                if indepOfLoopVars then
                    let loopArg = LoopArg.Input exprCh
                    (makeVar loopArg).[Ch.Default]
                else
                    exprCh |> BaseExprCh.map (BaseExpr.mapArgs liftConstants)
            )
        let channels = 
            channels 
            |> Map.map (fun _ (expr, sliceDim) -> liftConstants expr, sliceDim)

        // Build channel specification.
        let channelSpec = 
            channels 
            |> Map.map (fun _ (expr, sliceDim) -> {
                Loop.ChValue.Expr = expr 
                Loop.ChValue.SliceDim = sliceDim
            })

        BaseExpr.ofOp {
            Loop.Length=length
            Loop.Vars=Map.ofDictionary loopInput
            Loop.Channels=channelSpec
            Loop.Xs=List.ofSeq loopArgs
        } 

       
  