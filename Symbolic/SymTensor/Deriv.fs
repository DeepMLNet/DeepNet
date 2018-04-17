namespace SymTensor

open System.Diagnostics

open Tensor.Utils

[<AutoOpen>]
module DerivTypes =
    open Expr

    /// Jacobians for each variable
    type DerivT = {
        /// the number of elements of the function the derivative is taken of
        FunElems:   SizeSpecT
        /// the Jacobians w.r.t. the variables occuring in the expression
        Jacobians:  Map<VarSpecT, ExprT>
    }


    type internal LoopDerivT = {
        Port:        ChannelT
        Slice:       FullExprRngsSpecT
        ReverseAxis: int option
    }


    type internal PortContentsT = {
        DerivWrt:   ResizeArray<VarSpecT>
        ValueOf:    VarSpecT option
        SliceDim:   int
    }

/// derivative calculation
module Deriv =
    open Expr

    /// merges two derivative maps
    let private merge (aGrads: DerivT) (bGrads: DerivT) : DerivT =
        if aGrads.FunElems <> bGrads.FunElems then
            failwithf "cannot merge derivatives with different number of function elements: %A and %A"
                aGrads.FunElems bGrads.FunElems
        let jacs =
            (aGrads.Jacobians, bGrads.Jacobians)
            ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                        | Some ovg -> m |> Map.add v (vg + ovg)
                                        | None -> m |> Map.add v vg) 
        {FunElems=aGrads.FunElems; Jacobians=jacs}

    /// empty derivatives for expression
    let private empty expr =
        {FunElems=Expr.nElems expr; Jacobians=Map.empty}


    /// calculates the Jacobian of all arguments of an expression given the Jacobian of the expression
    let rec private reverseDiffStep (expr: ExprT) (eg: ExprT) : List<ExprT * ExprT> =    
        let exprShp = expr.Shape
        let funElems = eg.Shape.[0]  

        // check type and shape
        if expr.TypeName <> eg.TypeName then
            failwithf "Jacobian with type %A was specified for expression of type %A"
                eg.TypeName expr.TypeName
        if eg.Shape.[1] .<> expr.NElems then
            printfn "expr=\n%A" expr
            printfn "eg=\n%A" eg
            failwithf "Jacobian with %A wrt elements was specified for expression with %A elements"
                (shapeOf eg).[1] (ShapeSpec.nElem (shapeOf expr))

        /// expands the second dimension of the the Jacobian into the shape of this expression
        let egExp = eg |> reshape (funElems :: (shapeOf expr))

        /// flattens all but the first dimension into one dimension
        let collapse g =
            let wrtElems = (shapeOf g).[1..] |> ShapeSpec.nElem
            g |> reshape [funElems; wrtElems]

        // non differentiable op failures
        let failLogic op = failwithf "cannot calculate derivative of logic or comparison operation %A" op
        let failIndex op = failwithf "cannot calculate derivative of op %A that returns indices" op

        /// zero Jacobian
        let zeroJacobian wrt =
            Expr.zerosOfSameType wrt [funElems; wrt.NElems]

        // useful numbers
        let zero = Expr.zeroOfSameType expr
        let one = Expr.oneOfSameType expr
        let two = Expr.twoOfSameType expr
        let scalar = Expr.scalarOfSameType expr
        let zeros = Expr.zerosOfSameType expr

        match expr with
        | Leaf(op) -> List.empty            

        | Unary(op, a) ->
            match op with
            | Negate -> -eg 
            | Abs -> egExp * padLeft (signt a) |> collapse 
            | SignT -> zeroJacobian a
            | Log -> egExp * padLeft (a ** (-one)) |> collapse 
            | Log10 -> egExp * padLeft (a ** (-one) / log (scalar 10)) |> collapse
            | Exp -> egExp * padLeft (exp a) |> collapse 
            | Sin -> egExp * padLeft (cos a) |> collapse 
            | Cos -> egExp * padLeft (-sin a) |> collapse 
            | Tan -> egExp * padLeft (one + (tan a)**two) |> collapse 
            | Asin -> egExp * padLeft (one / sqrtt (one - a**two)) |> collapse 
            | Acos -> egExp * padLeft (-one / sqrtt (one - a**two)) |> collapse 
            | Atan -> egExp * padLeft (one / (one + a**two)) |> collapse 
            | Sinh -> egExp * padLeft (cosh a) |> collapse 
            | Cosh -> egExp * padLeft (sinh a) |> collapse 
            | Tanh -> egExp * padLeft (one - (tanh a)**two) |> collapse 
            | Sqrt -> egExp * padLeft (one / (two * sqrtt a)) |> collapse 
            | Ceil -> zeroJacobian a
            | Floor -> zeroJacobian a
            | Round -> zeroJacobian a
            | Truncate -> zeroJacobian a
            
            | Not -> zeroJacobian a

            | Diag (ax1, ax2) -> egExp |> diagMatAxis (ax1 + 1) (ax2 + 1) |> collapse 
            | DiagMat (ax1, ax2) -> egExp |> diagAxis (ax1 + 1) (ax2 + 1) |> collapse 
            | Invert -> -(padLeft expr.T) .* egExp .* (padLeft expr.T) |> collapse 
            | PermuteAxes perm -> 
                let backPerm = Permutation.invert perm
                let egePerm = 
                    0 :: List.map (fun p -> p + 1) backPerm
                egExp |> permuteAxes egePerm |> collapse 
            | Subtensor srs ->
                let agExpanded = zeros (funElems :: (shapeOf a))
                setSubtensor agExpanded.[SRSAll :: srs] egExp
                |> collapse 
            | Reshape ss -> eg 
            | DoBroadcast ss -> 
                let mutable egUnbroadcasted = egExp
                for ax, (eSize, aSize) in List.indexed (List.zip ss (shapeOf a)) do
                    match eSize, aSize with
                    | SizeSpecT.Broadcast, SizeSpecT.Broadcast -> ()
                    | _, SizeSpecT.Broadcast ->
                        egUnbroadcasted <- egUnbroadcasted |> sumKeepingAxis (ax + 1)
                    | _ -> ()
                egUnbroadcasted |> collapse 
            | ReverseAxis ax ->
                egExp |> reverseAxis (ax + 1) |> collapse
            | Gather indices ->
                let dIndices = indices |> List.map (Option.map padLeft)
                egExp |> scatter (None::dIndices) (funElems::shapeOf a) |> collapse
            | Scatter (indices, shp) ->
                let dIndices = indices |> List.map (Option.map (fun idx -> 
                    idx |> broadcastToShape (funElems::idx.Shape)))                   
                egExp |> gather (None::dIndices) |> collapse
            | Held (derivsShp, heldOp) -> 
                Unary(Held (shapeOf a :: derivsShp, heldOp), eg)       
                     
            | Sum -> eg |> enableBroadcast 1 |> broadcast (funElems :: ShapeSpec.flatten (shapeOf a)) 
                        |> collapse 
            | SumAxis ax -> 
                let bcEgExp = egExp |> reshape (shapeOf egExp |> ShapeSpec.insertBroadcastAxis (ax + 1))
                bcEgExp |> broadcast (shapeOf bcEgExp |> ShapeSpec.set (ax + 1) a.Shape.[ax]) |> collapse 
            | Product -> 
                // This division method incorrectly returns NaN for zero elements.
                // But currently I do not see any efficient alternative.
                let aBc = a |> reshape (SizeSpec.broadcastable :: ShapeSpec.flatten (shapeOf a))
                let pBc = expr |> reshape [SizeSpec.broadcastable; SizeSpec.broadcastable]
                (eg |> enableBroadcast 1) * (pBc / aBc)
            | ProductAxis ax ->
                let bcEgExp = egExp |> reshape (shapeOf egExp |> ShapeSpec.insertBroadcastAxis (ax + 1))
                let aBc = padLeft a
                let pBc = a |> productKeepingAxis ax |> padLeft
                bcEgExp * (pBc / aBc) |> collapse
            | MaxAxis ax 
            | MinAxis ax ->
                let bcExpr = expr |> reshape (expr.Shape |> ShapeSpec.insertBroadcastAxis ax)
                let bcEgExp = egExp |> reshape (egExp.Shape |> ShapeSpec.insertBroadcastAxis (ax + 1))
                Expr.ifThenElse (Expr.padLeft (a ==== bcExpr)) bcEgExp (zerosLike bcEgExp) |> collapse
            | ArgMaxAxis ax
            | ArgMinAxis ax -> zeroJacobian a

            | StoreToVar _ -> eg 

            | NullifyJacobian -> Expr.zerosLike eg 
            | AssumeJacobian jac ->
                match eg.Shape.[0], jac.Shape.[0] with
                | fl, jl when fl = jl -> jac
                | fl, jl when jl = SizeSpec.broadcastable -> jac |> Expr.broadcast [fl; jac.Shape.[1]]
                | _ -> failwithf "cannot broadcast specified Jacobian of shape %A to required 
                                  Jacobian shape %A" jac.Shape eg.Shape

            | Print _ -> eg 
            | Dump _ -> eg 
            | Annotated _ -> eg 
            | CheckFinite name ->
                eg |> checkFinite (sprintf "(partial) Jacobian wrt %s" name) 

            |> fun da -> [a, da]

        | Binary(op, a, b) ->

            let ifThenElseJac cond a b =
                let egZeros = zerosLike egExp
                let da = ifThenElse (padLeft cond) egExp egZeros |> collapse
                let db = ifThenElse (padLeft cond) egZeros egExp |> collapse
                [a, da; b, db]

            let inline (.+) da db = [a, da; b, db]

            match op with            
            | Add -> eg .+ eg
            | Substract -> eg .+ (-eg)
            | Multiply -> ((egExp * (padLeft b)) |> collapse) .+
                          ((egExp * (padLeft a)) |> collapse)
            | Divide -> ((egExp * padLeft (b**(-one))) |> collapse) .+
                        ((egExp * padLeft (-a * b**(-two))) |> collapse)
            | Modulo -> 
                failwith "Modulo gradient is broken"
                eg .+ (egExp * padLeft (-truncate (a / b)) |> collapse) 
            | Power -> (egExp * padLeft (b * a**(b - one)) |> collapse) .+ 
                       (egExp * padLeft (a**b * log a) |> collapse)
            
            | MaxElemwise -> ifThenElseJac (a >>>> b) a b
            | MinElemwise -> ifThenElseJac (a <<<< b) a b

            | Equal
            | Less
            | LessEqual
            | Greater
            | GreaterEqual
            | NotEqual
                -> zeroJacobian a .+ zeroJacobian b

            | And 
            | Or 
                -> zeroJacobian a .+ zeroJacobian b

            | IfThenElse cond -> ifThenElseJac cond a b

            | Dot -> 
                /// Jacobian of y = m .* x wrt x
                let mxWrtX (m: ExprT) x y dy =
                    let xShp, yShp, dyShp = shapeOf x, shapeOf y, shapeOf dy
                    let nd = ShapeSpec.nDim xShp
                    let batchShp = xShp.[0..nd-3]
                    let batchElems = ShapeSpec.nElem batchShp
                    let xSmplShp, ySmplShp = xShp.[nd-2..], yShp.[nd-2..]
                    let funElems = dyShp.[0]
                    let dyMat = dy |> swapDim 0 1 |> reshape (batchShp @ [ySmplShp.[0]; ySmplShp.[1] * funElems])
                    let dxMat = m.T .* dyMat
                    let dx = dxMat |> reshape [batchElems * xSmplShp.[0] * xSmplShp.[1]; funElems] |> swapDim 1 0
                    dx

                // Jacobian wrt b
                let db = mxWrtX a b expr eg

                // calculate Jacobian wrt a by transposing expression and resulting Jacobian
                let aShp = shapeOf a
                let nd = ShapeSpec.nDim aShp
                let batchShp = aShp.[0..nd-3]
                let egT = egExp.T |> collapse
                let daT = mxWrtX (b.T) (a.T) (expr.T) egT
                let da = daT |> reshape ([funElems] @ batchShp @ [aShp.[nd-1]; aShp.[nd-2]]) |> transpose |> collapse

                da .+ db
            | TensorProduct -> failwith "not implemented"
            | SetSubtensor sr ->
                let bgExpanded = egExp.[SRSAll::sr]
                let agExpanded = setSubtensor egExp.[SRSAll::sr] (zerosLike bgExpanded)
                (agExpanded |> collapse) .+ (bgExpanded |> collapse)

        | Nary(op, es) ->
            match op with
            | BuildTensor _ ->
                failwith "BuildTensor is used for optimization only and cannot be derived"
            | Elements (resShape, elemExpr) ->
                let desElemExprs = ElemExprDeriv.buildDerivElemExpr elemExpr resShape es.Length
                List.zip es desElemExprs
                |> List.map (fun (e, deElemExpr) -> 
                    let deShp = funElems :: (shapeOf e)
                    let deArgs = es @ [egExp]
                    e, Expr.elements deShp deElemExpr deArgs |> collapse)
            | Interpolate ip -> 
                match ip.Mode with
                | InterpolateLinearaly ->
                    List.indexed es
                    |> List.map (fun (d, e) ->
                        let ipd = ip |> Interpolator.getDerivative d 
                        e, egExp * padLeft (Expr.interpolate ipd es) |> collapse)
                | InterpolateToLeft -> 
                    es |> List.map (fun e -> e, zeroJacobian e)

            | Channel (Loop spec, output) ->              
                failwith "TODO"

            | ExtensionOp eop -> 
                let des = eop.Deriv eg es     
                List.zip es des          
            | Discard -> failwith "cannot propagate derivative thorugh Discard op"

    /// derivative of loop expression
    and private loopDeriv (dOutputs: Map<ChannelT, ExprT>) (originalArgs: ExprT list) (spec: LoopSpecT) =

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
        let args = ResizeArray<ExprT> originalArgs

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

        /// map from variable representing a derivative to the loop input specification
        let varInputSpecs = Dictionary<VarSpecT, LoopInputT> ()

        /// map from a loop output to the variable representing its derivative
        let dOutputVars = Dictionary<ChannelT, VarSpecT> ()

        /// map from a loop PreviousPort to the variables representing its derivative sources
        let dPreviousVars = Dictionary<PreviousChannelT, VarSpecT> ()

        /// map from a loop port to the value it must contain
        let portContents = Dictionary<ChannelT, PortContentsT> ()

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
            let dName = sprintf "d_%s" outPort
            let dVar =
                VarSpec.create dName value.Expr.Type (funElems :: value.Expr.Shape)
            dOutputVars.[outPort] <- dVar

            // create variable input specification:
            // source of incoming Jacobian is sequence of derivatives of the loop output
            let sas = {
                ArgIdx = addArg dOutputs.[outPort]
                SliceDim = value.SliceDim + 1
            }
            varInputSpecs.Add (dVar, SequenceArgSlice sas)         
               
        // go through loop variables and create corresponding derivative variables and ports
        for KeyValue (usingVar, li) in spec.Vars do
            let liType = usingVar.Type
            let liShape = usingVar.Shape
            let liElems = ShapeSpec.nElem usingVar.Shape
            let liDims = ShapeSpec.nDim usingVar.Shape

            match li with
            | ConstArg argIdx ->
                // create a variable for the sum of the accumulated Jacobian so far
                let dAccumName = sprintf "dSum_ConstArg%d[-1]" argIdx
                let dAccumVar = VarSpec.create dAccumName liType (funElems :: liShape)

                // create loop port exposing the step Jacobian plus the accumulated Jacobian w.r.t. ConstArg argIdx
                let dPortName = sprintf "dSum_ConstArg%d" argIdx
                if not (portContents.ContainsKey dPortName) then
                    portContents.[dPortName] <- {DerivWrt=ResizeArray<_>(); ValueOf=Some dAccumVar; SliceDim=liDims+1}
                portContents.[dPortName].DerivWrt.Add usingVar

                // create variable input specification:
                // source is accumulated Jacobian w.r.t. ConstArg argIdx in previous derivative loop iteration
                let dpp = {
                    Channel    = dPortName
                    Delay      = SizeSpec.one
                    InitialArg = addZeroInitialArg (funElems :: usingVar.Shape) usingVar.Type (liDims+1) SizeSpec.one
                }
                varInputSpecs.Add (dAccumVar, PreviousChannel dpp)

                // set Jacobian w.r.t. input argument argIdx specification
                let slice = [
                    yield RSAll                         // function element axis
                    for d=0 to liDims-1 do yield RSAll  // derivative axes
                    yield RSSymElem (spec.Length - 1L)  // sequence slice axis
                ]
                argIdxDerivs.[argIdx].Add {Port=dPortName; Slice=slice; ReverseAxis=None} |> ignore

            | SequenceArgSlice {ArgIdx=argIdx; SliceDim=sliceDim} ->
                // a sequence arg slice is an input variable and thus outputs a gradient
                // it thus needs a loop port 

                // create loop port exposing the step Jacobian w.r.t. the sequence slice
                let dPortName = sprintf "d_SeqArg%d_%d" argIdx sliceDim
                if not (portContents.ContainsKey dPortName) then
                    portContents.[dPortName] <- {DerivWrt=ResizeArray<_>(); ValueOf=None; SliceDim=sliceDim+1}
                portContents.[dPortName].DerivWrt.Add usingVar
                
                // set Jacobian w.r.t. input argument argIdx specification
                let slice = [
                    yield RSAll                                 // function element axis
                    for d=0 to sliceDim-1 do yield RSAll        // derivative axes
                    yield RSAll                                 // sequence slice axis
                    for d=sliceDim to liDims-1 do yield RSAll   // derivative axes
                ]
                argIdxDerivs.[argIdx].Add {Port=dPortName; Slice=slice; ReverseAxis=Some (sliceDim+1)} |> ignore

            | PreviousChannel pp ->
                // create loop port exposing the derivative w.r.t. the PreviousPort
                let dPortName = sprintf "d_%s[%A]" pp.Channel pp.Delay
                let sliceDim = spec.Channels.[pp.Channel].SliceDim
                if not (portContents.ContainsKey dPortName) then                    
                    portContents.Add (dPortName, {DerivWrt=ResizeArray<_>(); ValueOf=None; SliceDim=sliceDim+1})
                portContents.[dPortName].DerivWrt.Add usingVar

                // create a variable for Jacobian coming from a PreviousPort in a (future) loop iteration
                let dVar = VarSpec.create dPortName liType (funElems :: liShape)
                dPreviousVars.[pp] <- dVar

                // create corresponding variable input specification:
                // source is Jacobian calculated w.r.t. the PreviousPort in previous derivative loop iteration
                let dpp = {
                    Channel    = dPortName
                    Delay      = pp.Delay
                    InitialArg = addZeroInitialArg (funElems :: usingVar.Shape) usingVar.Type (sliceDim+1) pp.Delay
                }
                varInputSpecs.Add (dVar, PreviousChannel dpp)                                 

                // We need to output the Jacboian w.r.t. to the initial sequence argument.
                // It is available in the last "Delay" steps of the derivative loop port.
                let sliceDim = spec.Channels.[pp.Channel].SliceDim
                let slice = [
                    yield RSAll                                 // function element axis
                    for d=0 to sliceDim-1 do yield RSAll        // derivative axes
                    yield RSSymStartSymEnd                      // sequence slice axis
                        (Some (spec.Length - pp.Delay), Some (spec.Length - 1L))                
                    for d=sliceDim to liDims-1 do yield RSAll   // derivative axes
                ]
                argIdxDerivs.[pp.InitialArg].Add {Port=dPortName; Slice=slice; ReverseAxis=Some (sliceDim+1)} |> ignore

                                               
            | IterationIndex 
            | IterationsRemaining -> 
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
                let chDeriv = value.Expr |> computeWithRootJacobian incomingJacobian               
                chDeriv
                )    
            |> Seq.reduce merge

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
                {Expr=expr; SliceDim=sliceDim})

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
                | ConstArg _ -> 
                    // constant arguments needs no adaption
                    li
                | SequenceArgSlice {ArgIdx=argIdx; SliceDim=sliceDim} ->
                    // sequence arguments must be reversed
                    let revExpr = Expr.reverseAxis sliceDim args.[argIdx]
                    SequenceArgSlice {ArgIdx=addArg revExpr; SliceDim=sliceDim}
                | PreviousChannel pp ->
                    // previous channel accesses the reversed output of the orignal loop
                    // with appropriate slicing to account for the delay
                    let portOutput = Expr.loop spec pp.Channel originalArgs
                    let portExpr = spec.Channels.[pp.Channel].Expr
                    let sliceDim = spec.Channels.[pp.Channel].SliceDim

                    let initialValues = originalArgs.[pp.InitialArg]
                    let portSeq = Expr.concat sliceDim [initialValues; portOutput]
                    let revPortSeq = portSeq |> Expr.reverseAxis sliceDim

                    let delaySlice : FullExprRngsSpecT = [
                        for d=0 to sliceDim-1 do yield RSAll 
                        yield RSSymStartSymEnd (Some pp.Delay, None)
                        for d=sliceDim to portExpr.NDims-1 do yield RSAll
                    ]
                    let delayedPortSeq = revPortSeq.[delaySlice]

                    SequenceArgSlice {ArgIdx=addArg delayedPortSeq; SliceDim=sliceDim}
                | IterationIndex -> 
                    // iteration index and iterations remaining are swapped
                    IterationsRemaining
                | IterationsRemaining -> 
                    // iteration index and iterations remaining are swapped
                    IterationIndex)

        // build loop specification for derivative loop
        let dSpec = {
            Length    = spec.Length
            Vars      = Map.join originalVars varsFromDeriv
            Channels  = ports
        }
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
                        let loopOutput = Expr.loop dSpec port (List.ofSeq args)
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
        [for a=0 to originalArgs.Length-1 do
                yield originalArgs.[a], argIdxDerivExprs.[a]]
        
    /// computes the Jacobians of the arguments of a multi-channel op given the Jacobians
    /// w.r.t. all channels of the multi-channel op
    and private multiChannelDiffStep (mcOp: MultiChannelOpUsageT) (eg: Map<ChannelT, ExprT>) : List<ExprT * ExprT> =
        match mcOp with
        | Loop spec, args -> loopDeriv eg args spec

    /// computes the derivatives of the specified expression w.r.t. all variables occuring in it
    and computeWithRootJacobian (rootJacobian: ExprT) (rootExpr: ExprT) : DerivT =

        // build expression info and unify common subexpressions
        let exprInfo = ExprInfoT [rootExpr]
        let rootExpr = List.exactlyOne exprInfo.Exprs

        /// map from an expression to the sum of incoming Jacobians
        let incomingJacobian = Dictionary<ExprT, ExprT> (HashIdentity.Reference)
        /// map from an expression to the set of dependants that transmitted Jacobian to the expression
        let receivedJacobiansFrom = Dictionary<ExprT, HashSet<ExprT>> (HashIdentity.Reference)
        /// expressions that have received Jacobians from all their dependants
        let exprsWithFullJacobian = Queue<ExprT> ()

        let multiChannelOpJacobians = 
            Dictionary<MultiChannelOpUsageT, Dictionary<ChannelT, ExprT>> (HashIdentity.Structural) 
        let multiChannelOpsWithFullJacobians = Queue<MultiChannelOpUsageT> ()

        /// adds the specified Jacobian coming from `source` to `target`
        let transmitJacobian source target jacobian =
            let neededSources = exprInfo.Dependants target |> Set.ofSeq

            // add jacobian
            match incomingJacobian.TryFind target with
            | Some j -> incomingJacobian.[target] <- j + jacobian
            | None -> incomingJacobian.[target] <- jacobian

            // add to received set
            if not (receivedJacobiansFrom.ContainsKey target) then
                receivedJacobiansFrom.[target] <- HashSet<ExprT> (HashIdentity.Structural)
            match source with
            | Choice1Of2 exprSource -> 
                if receivedJacobiansFrom.[target].Contains exprSource then
                    failwithf "Jacobian from %A to %A was already transmitted" exprSource target
                if not (neededSources.Contains exprSource) then
                    failwithf "%A received Jacobian from non-dependant %A" target exprSource
                receivedJacobiansFrom.[target].Add exprSource |> ignore
            | Choice2Of2 mcopSource ->
                neededSources
                |> Seq.filter (function 
                               | Nary (Channel (op, channel), es) when (op, es) = mcopSource -> true
                               | _ -> false)
                |> Seq.iter (fun src -> receivedJacobiansFrom.[target].Add src |> ignore)

            // check if target has received all Jacobians
            let receivedSources = receivedJacobiansFrom.[target] |> Set.ofSeq
            if receivedSources = neededSources then exprsWithFullJacobian.Enqueue target

        let transmitJacobians src jacobians =
            jacobians
            |> List.groupBy (fun (target, _) -> target)
            |> List.iter (fun (target, jacs) ->
                let jacSum = jacs |> List.map (fun (_, jac) -> jac) |> List.reduce (+)
                transmitJacobian src target jacSum)            

        let transmitMultiChannelOpJacobian mcOp channel jacobian =
            // add jacobian
            if not (multiChannelOpJacobians.ContainsKey mcOp) then
                multiChannelOpJacobians.[mcOp] <- Dictionary<ChannelT, ExprT> (HashIdentity.Structural)
            let mcoj = multiChannelOpJacobians.[mcOp]
            mcoj.[channel] <- jacobian

            // check if multi-channel op has received Jacobians on all its channels
            let received = Set.ofSeq mcoj.Keys
            let needed = exprInfo.UsedChannels mcOp
            if received = needed then multiChannelOpsWithFullJacobians.Enqueue mcOp

        // set Jacobian of root node
        incomingJacobian.[rootExpr] <- rootJacobian
        exprsWithFullJacobian.Enqueue rootExpr

        // process Jacobians in loop
        let mutable varJacs = Map.empty
        while exprsWithFullJacobian.Count > 0 || multiChannelOpsWithFullJacobians.Count > 0 do

            if exprsWithFullJacobian.Count > 0 then
                let expr = exprsWithFullJacobian.Dequeue ()

                // propagate Jacobians
                match expr with
                | Nary (Channel (op, channel), es) ->
                    transmitMultiChannelOpJacobian (op, es) channel incomingJacobian.[expr]
                | _ ->
                    reverseDiffStep expr incomingJacobian.[expr] |> transmitJacobians (Choice1Of2 expr)

                // extract variable Jacobians
                match expr with
                | Leaf (Var vs) -> varJacs <- varJacs |> Map.add vs incomingJacobian.[expr]
                | _ -> ()

            if multiChannelOpsWithFullJacobians.Count > 0 then
                let mcOp = multiChannelOpsWithFullJacobians.Dequeue ()
                let channelJacs = multiChannelOpJacobians.[mcOp] |> Map.ofDictionary               
                multiChannelDiffStep mcOp channelJacs |> transmitJacobians (Choice2Of2 mcOp)
        
        {
            FunElems  = rootJacobian.Shape.[0]
            Jacobians = varJacs
        }    

    /// computes the derivatives of the specified expression w.r.t. all variables occuring in it
    and compute (rootExpr: ExprT) : DerivT =
        if Debug.TraceCompile then printfn "Computing derivatives..."
        let sw = Stopwatch.StartNew()
        let rootJac = shapeOf rootExpr |> ShapeSpec.nElem |> identityOfSameType rootExpr
        let deriv = computeWithRootJacobian rootJac rootExpr
        if Debug.Timing then printfn "Computing derivatives took %A" sw.Elapsed
        deriv

    /// extracts the Jacobian of the given VarSpecT
    and ofVarSpec var (deriv: DerivT) =
        match deriv.Jacobians |> Map.tryFind var with
        | Some d -> d
        | None when Debug.FailIfVarNotInDerivative -> 
            failwithf "the variable %A is not present in the expression" var
        | None -> 
            let varExpr = Expr.makeVar var
            Expr.zerosOfSameType varExpr [deriv.FunElems; Expr.nElems varExpr]

    /// extracts the Jacobian of the given variable
    and ofVar var deriv =
        ofVarSpec (extractVar var) deriv                  


