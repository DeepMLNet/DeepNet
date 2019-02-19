namespace SymTensor.Deriv

open DeepNet.Utils
open SymTensor
open SymTensor.Ops
//open SymTensor.Loop


type internal LoopDerivT = {
    Port:        Ch
    Slice:       RangesSpec
    ReverseAxis: int option
}


type internal PortContentsT = {
    DerivWrt:   ResizeArray<Var>
    ValueOf:    Var option
    SliceDim:   int
}


[<OpExtender>]
type LoopDeriv(op: Loop) =
    interface IDerivableOp with
        member this.Deriv dOp =   
            let env = MultiChannelDerivTools.Env.make op dOp
            let dOutputs = env.DOp
            let originalArgs = env.Xs
            let spec = op

            /// number of elments of the function we take the derivative of
            let funElems = 
                match Map.toList dOutputs with
                | (ch0, dExpr0) :: rds ->
                    for ch, dExpr in rds do
                        if dExpr.Shape.[0] <> dExpr0.Shape.[0] then
                            failwith "inconsistent number of derivative function elements"
                    dExpr0.Shape.[0]
                | [] -> failwith "output derivatives invalid"

            /// argument of the derivative loop expression
            let args = ResizeArray<Expr> originalArgs

            /// adds an argument to the derivative loop expression and returns its index
            let addArg expr =
                match args |> Seq.tryFindIndex ((=) expr) with
                | Some idx -> idx
                | None ->
                    let idx = args.Count
                    args.Add expr
                    idx

            /// adds an argument with a value full of zeros for use with initial value of a PreviousChannel
            let addZeroInitialArg channelShp channelType sliceDim delay =
                let shp = channelShp |> ShapeSpec.insertAxis sliceDim delay
                let zeroExpr = Expr.zerosOfType channelType shp
                addArg zeroExpr

            /// Name of a channel.
            let chName (ch: Ch) =
                match ch with
                | Ch.Default -> "__DEFAULT__"
                | Ch.Custom name -> name

            /// map from variable representing a derivative to the loop input specification
            let varInputSpecs = Dictionary<Var, Loop.Input> ()

            /// map from a loop output to the variable representing its derivative
            let dOutputVars = Dictionary<Ch, Var> ()

            /// map from a loop PreviousPort to the variables representing its derivative sources
            let dPreviousVars = Dictionary<Loop.PreviousChannel, Var> ()

            /// map from a loop port to the value it must contain
            let portContents = Dictionary<Ch, PortContentsT> ()

            /// map from argument index to the loop ports containing its derivative summands
            let argIdxDerivs = Dictionary<int, HashSet<LoopDerivT>> ()
            for idx=0 to originalArgs.Length-1 do
                argIdxDerivs.[idx] <- HashSet<_> ()

            // expand and reverse all incoming Jacobians
            let dOutputs =
                dOutputs
                |> Map.map (fun ch dCh ->
                    let sliceDim = spec.Channels.[ch].SliceDim
                    let expShp = 
                        (funElems :: spec.Channels.[ch].Expr.Shape)
                        |> ShapeSpec.insertAxis (sliceDim + 1) spec.Length
                    dCh 
                    |> Expr.reshape expShp
                    |> Expr.reverseAxis (sliceDim + 1))

            // go through loop outputs and create variables representing their derivatives
            for KeyValue (outPort, dExpr) in dOutputs do
                // create variable for incoming Jacobian
                let value = spec.Channels.[outPort]
                let dName = sprintf "d_%s" (chName outPort)
                let dVar =
                    Var.create dName value.Expr.DataType (funElems :: value.Expr.Shape)
                dOutputVars.[outPort] <- dVar

                // create variable input specification:
                // source of incoming Jacobian is sequence of derivatives of the loop output
                let sas: Loop.SequenceArgSlice = {
                    ArgIdx = addArg dOutputs.[outPort]
                    SliceDim = value.SliceDim + 1
                }
                varInputSpecs.Add (dVar, Loop.SequenceArgSlice sas)         
               
            // go through loop variables and create corresponding derivative variables and ports
            for KeyValue (usingVar, li) in spec.Vars do
                let liType = usingVar.Type
                let liShape = usingVar.Shape
                let liElems = ShapeSpec.nElem usingVar.Shape
                let liDims = ShapeSpec.nDim usingVar.Shape

                match li with
                | Loop.ConstArg argIdx ->
                    // create a variable for the sum of the accumulated Jacobian so far
                    let dAccumName = sprintf "dSum_ConstArg%d[-1]" argIdx
                    let dAccumVar = Var.create dAccumName liType (funElems :: liShape)

                    // create loop port exposing the step Jacobian plus the accumulated Jacobian w.r.t. ConstArg argIdx
                    let dPort = Ch.Custom (sprintf "dSum_ConstArg%d" argIdx)
                    if not (portContents.ContainsKey dPort) then
                        portContents.[dPort] <- {DerivWrt=ResizeArray<_>(); ValueOf=Some dAccumVar; SliceDim=liDims+1}
                    portContents.[dPort].DerivWrt.Add usingVar

                    // create variable input specification:
                    // source is accumulated Jacobian w.r.t. ConstArg argIdx in previous derivative loop iteration
                    let dpp: Loop.PreviousChannel = {
                        Channel    = dPort
                        Delay      = SizeSpec.one
                        InitialArg = addZeroInitialArg (funElems :: usingVar.Shape) usingVar.Type (liDims+1) SizeSpec.one
                    }
                    varInputSpecs.Add (dAccumVar, Loop.PreviousChannel dpp)

                    // set Jacobian w.r.t. input argument argIdx specification
                    let slice = [
                        yield RangeSpec.All                         // function element axis
                        for d=0 to liDims-1 do yield RangeSpec.All  // derivative axes
                        yield RangeSpec.SymElem (spec.Length - 1L)  // sequence slice axis
                    ]
                    argIdxDerivs.[argIdx].Add {Port=dPort; Slice=slice; ReverseAxis=None} |> ignore

                | Loop.SequenceArgSlice {ArgIdx=argIdx; SliceDim=sliceDim} ->
                    // a sequence arg slice is an input variable and thus outputs a gradient
                    // it thus needs a loop port 

                    // create loop port exposing the step Jacobian w.r.t. the sequence slice
                    let dPort = Ch.Custom (sprintf "d_SeqArg%d_%d" argIdx sliceDim)
                    if not (portContents.ContainsKey dPort) then
                        portContents.[dPort] <- {DerivWrt=ResizeArray<_>(); ValueOf=None; SliceDim=sliceDim+1}
                    portContents.[dPort].DerivWrt.Add usingVar
                
                    // set Jacobian w.r.t. input argument argIdx specification
                    let slice = [
                        yield RangeSpec.All                                 // function element axis
                        for d=0 to sliceDim-1 do yield RangeSpec.All        // derivative axes
                        yield RangeSpec.All                                 // sequence slice axis
                        for d=sliceDim to liDims-1 do yield RangeSpec.All   // derivative axes
                    ]
                    argIdxDerivs.[argIdx].Add {Port=dPort; Slice=slice; ReverseAxis=Some (sliceDim+1)} |> ignore

                | Loop.PreviousChannel pp ->
                    // create loop port exposing the derivative w.r.t. the PreviousPort
                    let dPortName = sprintf "d_%s[%A]" (chName pp.Channel) pp.Delay
                    let dPort = Ch.Custom dPortName
                    let sliceDim = spec.Channels.[pp.Channel].SliceDim
                    if not (portContents.ContainsKey dPort) then                    
                        portContents.Add (dPort, {DerivWrt=ResizeArray<_>(); ValueOf=None; SliceDim=sliceDim+1})
                    portContents.[dPort].DerivWrt.Add usingVar

                    // create a variable for Jacobian coming from a PreviousPort in a (future) loop iteration
                    let dVar = Var.create dPortName liType (funElems :: liShape)
                    dPreviousVars.[pp] <- dVar

                    // create corresponding variable input specification:
                    // source is Jacobian calculated w.r.t. the PreviousPort in previous derivative loop iteration
                    let dpp: Loop.PreviousChannel = {
                        Channel    = dPort
                        Delay      = pp.Delay
                        InitialArg = addZeroInitialArg (funElems :: usingVar.Shape) usingVar.Type (sliceDim+1) pp.Delay
                    }
                    varInputSpecs.Add (dVar, Loop.PreviousChannel dpp)                                 

                    // We need to output the Jacboian w.r.t. to the initial sequence argument.
                    // It is available in the last "Delay" steps of the derivative loop port.
                    let sliceDim = spec.Channels.[pp.Channel].SliceDim
                    let slice = [
                        yield RangeSpec.All                                 // function element axis
                        for d=0 to sliceDim-1 do yield RangeSpec.All        // derivative axes
                        yield RangeSpec.SymStartSymEnd                      // sequence slice axis
                            (Some (spec.Length - pp.Delay), Some (spec.Length - 1L))                
                        for d=sliceDim to liDims-1 do yield RangeSpec.All   // derivative axes
                    ]
                    argIdxDerivs.[pp.InitialArg].Add {Port=dPort; Slice=slice; ReverseAxis=Some (sliceDim+1)} |> ignore

                                               
                | Loop.IterationIndex 
                | Loop.IterationsRemaining -> 
                    // iteration index is an intergral constant
                    ()        
            
            /// derivatives of all ports w.r.t. all variables
            let portDerivs =
                spec.Channels
                |> Map.toSeq
                |> Seq.map (fun (port, value) ->              
                    // build expression for incoming Jacobian, i.e. Jacobian w.r.t. this port
                    // shape is: [funElems; <shape of value.Expr>]
                    let incomingExpandedJacobian = 
                        seq { 
                            // derivative coming from external use of port's output slice
                            match dOutputVars.TryFind port with
                            | Some dVar -> yield Expr.makeVar dVar
                            | None -> ()

                            // derivatives coming from PreviousPort uses of this port 
                            for dpv in dPreviousVars do
                                let previousPort, dVar = dpv.Key, dpv.Value
                                if previousPort.Channel = port then yield Expr.makeVar dVar
                        } |> Seq.reduce (+)
                    
                    // collapse Jacobian
                    let incomingJacobian = incomingExpandedJacobian |> Expr.reshape [funElems; value.Expr.NElems]

                    // calculate Jacobians w.r.t. all variables
                    let chDeriv = Deriv.computeWithRootDeriv incomingJacobian (Expr value.Expr)
                    chDeriv
                    )    
                |> Seq.reduce Deriv.merge

            // go through portContents and create actual port contents
            let ports =
                portContents
                |> Map.ofDictionary
                |> Map.map (fun port {DerivWrt=derivWrts; ValueOf=valueOf; SliceDim=sliceDim} ->
                    let expr = 
                        seq {
                            // obtain Jacobians
                            for wrt in derivWrts do
                                let wrtJacobian = portDerivs |> ofVarSpec wrt
                                let wrtExpandedJacobian = wrtJacobian |> Expr.reshape (funElems :: wrt.Shape)
                                yield wrtExpandedJacobian
                   
                            // obtain value, if any
                            match valueOf with
                            | Some vs -> yield Expr.makeVar vs
                            | None -> ()
                        } |> Seq.reduce (+)
                    let value: Loop.Value = {Expr=expr.BaseExprCh; SliceDim=sliceDim}
                    value)

            // create variable specification
            let varsFromDeriv = 
                varInputSpecs
                |> Seq.map (fun vis -> vis.Key, vis.Value)
                |> Map.ofSeq

            // adapt original vars of loop
            let originalVars =
                spec.Vars
                |> Map.map (fun vs li ->
                    match li with
                    | Loop.ConstArg _ -> 
                        // constant arguments needs no adaption
                        li
                    | Loop.SequenceArgSlice {ArgIdx=argIdx; SliceDim=sliceDim} ->
                        // sequence arguments must be reversed
                        let revExpr = Expr.reverseAxis sliceDim args.[argIdx]
                        Loop.SequenceArgSlice {ArgIdx=addArg revExpr; SliceDim=sliceDim}
                    | Loop.PreviousChannel pp ->
                        // previous channel accesses the reversed output of the orignal loop
                        // with appropriate slicing to account for the delay                        
                        let portLoop = MultiChannelExpr.loop spec.Length spec.Vars spec.Channels originalArgs
                        let portOutput = portLoop.[pp.Channel]
                        let portExpr = spec.Channels.[pp.Channel].Expr
                        let sliceDim = spec.Channels.[pp.Channel].SliceDim

                        let initialValues = originalArgs.[pp.InitialArg]
                        let portSeq = Expr.concat sliceDim [initialValues; portOutput]
                        let revPortSeq = portSeq |> Expr.reverseAxis sliceDim

                        let delaySlice : RangesSpec = [
                            for d=0 to sliceDim-1 do yield RangeSpec.All 
                            yield RangeSpec.SymStartSymEnd (Some pp.Delay, None)
                            for d=sliceDim to portExpr.NDims-1 do yield RangeSpec.All
                        ]
                        let delayedPortSeq = revPortSeq.[delaySlice]

                        Loop.SequenceArgSlice {ArgIdx=addArg delayedPortSeq; SliceDim=sliceDim}
                    | Loop.IterationIndex -> 
                        // iteration index and iterations remaining are swapped
                        Loop.IterationsRemaining
                    | Loop.IterationsRemaining -> 
                        // iteration index and iterations remaining are swapped
                        Loop.IterationIndex)

            // build loop specification for derivative loop
            let dSpec: Loop = {
                Length    = spec.Length
                Vars      = Map.join originalVars varsFromDeriv
                Channels  = ports
                Xs        = args |> List.ofSeq |> List.map (Expr.baseExprCh)
            }
            let dLoopExpr = MultiChannelExpr dSpec
            //printfn "derivative loop spec is\n%A" dSpec

            // build derivatives w.r.t. our arguments
            let argIdxDerivExprs = 
                argIdxDerivs 
                |> Map.ofDictionary
                |> Map.map (fun argIdx loopDerivs ->                
                    // sum over ports producing derivative and reverse if necessary
                    let dExprExpanded =
                        loopDerivs
                        |> Seq.map (fun {Port=port; Slice=slice; ReverseAxis=reverseAxis} ->
                            let loopOutput = dLoopExpr.[port]
                            let sliced = loopOutput.[slice]
                            match reverseAxis with
                            | Some ax -> sliced |> Expr.reverseAxis ax
                            | None -> sliced)
                        |> Seq.reduce (+)

                    // collapse Jacobian
                    let wrtElems = ShapeSpec.nElem dExprExpanded.Shape.[1..] 
                    let dExpr = dExprExpanded |> Expr.reshape [funElems; wrtElems]
                    dExpr)

            // output mapping from original argument to its derivative
            argIdxDerivExprs |> Map.mapKeyValue (fun i d -> Arg.N i, d)


        
   