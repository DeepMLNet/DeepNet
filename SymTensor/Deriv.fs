namespace SymTensor

open Basics

[<AutoOpen>]
module DerivTypes =
    open Expr

    /// Jacobians for each variable
    type DerivT = {
        /// the expression the derivative was calculated of
        Expr:       ExprT
        /// the Jacobians w.r.t. the variables occuring in the expression
        Jacobians:  Map<VarSpecT, ExprT>
    }


    type internal LoopDerivT = {
        Port:        LoopPortT
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
        if aGrads.Expr <> bGrads.Expr then
            failwith "derivatives must belong to same expression"
        let jacs =
            (aGrads.Jacobians, bGrads.Jacobians)
            ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                        | Some ovg -> m |> Map.add v (vg + ovg)
                                        | None -> m |> Map.add v vg) 
        {Expr=aGrads.Expr; Jacobians=jacs}

    /// empty derivatives for expression
    let private empty expr =
        {Expr=expr; Jacobians=Map.empty}

    /// reverse accumulation autodifferentiation of an expression
    let rec reverseDiff (baseExpr: ExprT) (expr: ExprT) (eg: ExprT) : DerivT =    
        let rds = reverseDiff baseExpr
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
        let egExpanded =
            eg |> reshape (funElems :: (shapeOf expr))

        /// flattens all but the first dimension into one dimension
        let collapse g =
            let wrtElems = (shapeOf g).[1..] |> ShapeSpec.nElem
            g |> reshape [funElems; wrtElems]

        /// total derivates given op's derivates
        let totalDerivates es des =
            (empty baseExpr, List.zip es des)
            ||> List.fold (fun totGrad (e, de) ->
                let eGrad = rds e de
                merge totGrad eGrad)

        /// logic op failure
        let failLogic op =
            failwithf "cannot calculate derivative of logic or comparison operation %A" op

        // useful numbers
        let zero = Expr.zeroOfSameType expr
        let one = Expr.oneOfSameType expr
        let two = Expr.twoOfSameType expr
        let scalar = Expr.scalarOfSameType expr
        let zeros = Expr.zerosOfSameType expr

        match expr with
        | Leaf(op) ->                  
            match op with
            | ScalarConst _ -> empty baseExpr
            | SizeValue _ -> empty baseExpr
            | Identity _ -> empty baseExpr
            | Var v -> {empty baseExpr with Jacobians=Map [v, eg]}

        | Unary(op, a) ->
            match op with
            | Negate -> -eg |> rds a
            | Abs -> egExpanded * padLeft (signt a) |> collapse |> rds a
            | SignT -> empty baseExpr
            | Log -> egExpanded * padLeft (a ** (-one)) |> collapse |> rds a
            | Log10 -> eg |> rds (log a / log (scalar 10))
            | Exp -> egExpanded * padLeft (exp a) |> collapse |> rds a
            | Sin -> egExpanded * padLeft (cos a) |> collapse |> rds a
            | Cos -> egExpanded * padLeft (-sin a) |> collapse |> rds a
            | Tan -> egExpanded * padLeft (one + (tan a)**two) |> collapse |> rds a
            | Asin -> egExpanded * padLeft (one / sqrtt (one - a**two)) |> collapse |> rds a
            | Acos -> egExpanded * padLeft (-one / sqrtt (one - a**two)) |> collapse |> rds a
            | Atan -> egExpanded * padLeft (one / (one + a**two)) |> collapse |> rds a
            | Sinh -> egExpanded * padLeft (cosh a) |> collapse |> rds a
            | Cosh -> egExpanded * padLeft (sinh a) |> collapse |> rds a
            | Tanh -> egExpanded * padLeft (one - (tanh a)**two) |> collapse |> rds a
            | Sqrt -> egExpanded * padLeft (one / (two * sqrtt a)) |> collapse |> rds a
            | Ceil -> empty baseExpr
            | Floor -> empty baseExpr
            | Round -> empty baseExpr
            | Truncate -> empty baseExpr
            
            | Not -> failLogic op

            | Diag (ax1, ax2) -> egExpanded |> diagMatAxis (ax1 + 1) (ax2 + 1) |> collapse |> rds a
            | DiagMat (ax1, ax2) -> egExpanded |> diagAxis (ax1 + 1) (ax2 + 1) |> collapse |> rds a
            | Invert -> -(padLeft expr.T) .* egExpanded .* (padLeft expr.T) |> collapse |> rds a
            | PermuteAxes perm -> 
                let backPerm = Permutation.invert perm
                let egePerm = 
                    0 :: List.map (fun p -> p + 1) backPerm
                egExpanded |> permuteAxes egePerm |> collapse |> rds a

            | Subtensor srs ->
                let agExpanded = zeros (funElems :: (shapeOf a))
                setSubtensor agExpanded.[SRSAll :: srs] egExpanded
                |> collapse 
                |> rds a
            | Reshape ss -> eg |> rds a
            | DoBroadcast ss -> 
                let mutable egUnbroadcasted = egExpanded
                for ax, (eSize, aSize) in List.indexed (List.zip ss (shapeOf a)) do
                    match eSize, aSize with
                    | SizeSpecT.Broadcast, SizeSpecT.Broadcast -> ()
                    | _, SizeSpecT.Broadcast ->
                        egUnbroadcasted <- egUnbroadcasted |> sumKeepingAxis (ax + 1)
                    | _ -> ()
                egUnbroadcasted |> collapse |> rds a
            | Held (derivsShp, heldOp) -> 
                Unary(Held (shapeOf a :: derivsShp, heldOp), eg) |> rds a                
            | Sum -> eg |> enableBroadcast 1 |> broadcast (funElems :: ShapeSpec.flatten (shapeOf a)) 
                        |> collapse |> rds a
            | SumAxis ax -> 
                let eeg = egExpanded 
                let bca = eeg |> reshape (shapeOf eeg |> ShapeSpec.insertBroadcastAxis (ax + 1))
                let ael = (shapeOf a).[ax]
                let bc = bca |> broadcast (shapeOf bca |> ShapeSpec.set (ax + 1) ael)
                bc |> collapse |> rds a
            | StoreToVar _ -> eg |> rds a

            | NullifyJacobian ->
                Expr.zerosLike eg |> rds a
            | AssumeJacobian jac ->
                let jacBc =
                    match eg.Shape.[0], jac.Shape.[0] with
                    | fl, jl when fl = jl -> jac
                    | fl, jl when jl = SizeSpec.broadcastable ->
                        jac |> Expr.broadcast [fl; jac.Shape.[1]]
                    | _ -> 
                        failwithf "cannot broadcast specified Jacobian of shape %A to required 
                                   Jacobian shape %A" jac.Shape eg.Shape
                jacBc |> rds a

            | Print _ -> eg |> rds a
            | Dump _ -> eg |> rds a
            | Annotated _ -> eg |> rds a
            | CheckFinite name ->
                eg |> checkFinite (sprintf "(partial) Jacobian wrt %s" name) |> rds a

        | Binary(op, a, b) ->
            let inline (.+) da db = totalDerivates [a; b] [da; db]

            match op with            
            | Add -> eg .+ eg
            | Substract -> eg .+ (-eg)
            | Multiply -> ((egExpanded * (padLeft b)) |> collapse) .+
                          ((egExpanded * (padLeft a)) |> collapse)
            | Divide -> eg |> rds (a * b ** (-one))
            | Modulo -> 
                failwith "Modulo gradient is broken"
                eg .+ (padLeft (-truncate (a / b)) |> collapse) 
            | Power -> (egExpanded * padLeft (b * a**(b - one)) |> collapse) .+ 
                       (egExpanded * padLeft (a**b * log a) |> collapse)
            
            | MaxElemwise -> eg |> rds (ifThenElse (a >>>> b) a b)
            | MinElemwise -> eg |> rds (ifThenElse (a <<<< b) a b)

            | Equal
            | Less
            | LessEqual
            | Greater
            | GreaterEqual
            | NotEqual
                -> failLogic op

            | And 
            | Or 
                -> failLogic op

            | IfThenElse cond ->
                let egZeros = zerosLike egExpanded
                let da = ifThenElse (padLeft cond) egExpanded egZeros |> collapse
                let db = ifThenElse (padLeft cond) egZeros egExpanded |> collapse
                da .+ db

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
                let egT = egExpanded.T |> collapse
                let daT = mxWrtX (b.T) (a.T) (expr.T) egT
                let da = daT |> reshape ([funElems] @ batchShp @ [aShp.[nd-1]; aShp.[nd-2]]) |> transpose |> collapse

                da .+ db
            | TensorProduct -> failwith "not implemented"
            | SetSubtensor sr ->
                let bgExpanded = egExpanded.[SRSAll::sr]
                let agExpanded = setSubtensor egExpanded.[SRSAll::sr] (zerosLike bgExpanded)
                (agExpanded |> collapse) .+ (bgExpanded |> collapse)

        | Nary(op, es) ->
            match op with
            | Elements (resShape, elemExpr) ->
                let desElemExprs = ElemExprDeriv.buildDerivElemExpr elemExpr resShape (es.Length)
                let des = 
                    List.zip es desElemExprs
                    |> List.map (fun (e, deElemExpr) -> 
                        let deShp = funElems :: (shapeOf e)
                        let deArgs = es @ [egExpanded]
                        Expr.elements deShp deElemExpr deArgs |> collapse)
                totalDerivates es des
            | Interpolate ip -> 
                match ip.Mode with
                | InterpolateLinearaly ->
                    let des = 
                        [for d=0 to es.Length-1 do
                            let ipd = ip |> Interpolator.getDerivative d 
                            yield egExpanded * padLeft (Expr.interpolate ipd es) |> collapse]
                    totalDerivates es des
                | InterpolateToLeft -> empty baseExpr

            | Loop (spec, output) ->
                

                failwith "TODO"

            | ExtensionOp eop -> eop.Deriv eg es |> totalDerivates es                
            | Discard -> failwith "cannot propagate derivative thorugh Discard op"


    /// computes the derivatives of the specified expression w.r.t. all variables occuring in it
    and compute (expr: ExprT) : DerivT =
        let eg = shapeOf expr |> ShapeSpec.nElem |> identityOfSameType expr
        reverseDiff expr expr eg

    and ofVarSpec var (deriv: DerivT) =
        match deriv.Jacobians |> Map.tryFind var with
        | Some d -> d
        | None when Debug.FailIfVarNotInDerivative -> 
            failwithf "the variable %A is not present in the expression" var
        | None -> 
            let varExpr = Expr.makeVar var
            Expr.zerosOfSameType varExpr [Expr.nElems deriv.Expr; Expr.nElems varExpr]

    /// extracts the Jacobian of the given variable
    and ofVar var deriv =
        ofVarSpec (extractVar var) deriv           
        


    and loopDeriv (funElems: SizeSpecT) (dOutputs: Map<LoopPortT, ExprT>) 
            (originalArgs: ExprT list) (spec: LoopSpecT) =

        let portType p =
            spec.Ports.[p].Expr.Type

        let portSliceShape p =
            spec.Ports.[p].Expr.Shape |> ShapeSpec.withoutAxis spec.Ports.[p].SliceDim

        let portSliceElems p =
            p |> portSliceShape |> ShapeSpec.nElem

        let args = ResizeArray<ExprT> originalArgs
        let addArg expr =
            match args |> Seq.tryFindIndex ((=) expr) with
            | Some idx -> idx
            | None ->
                let idx = args.Count
                args.Add expr
                idx

        // step 1: assign variables to (incoming) Jacobians of all ports        

        /// map from variable representing a derivative to the loop input specification
        let varInputSpecs = Dictionary<VarSpecT, LoopInputT> ()

        /// map from a loop output to the variable representing its derivative
        let dOutputVars = Dictionary<LoopPortT, VarSpecT> ()

        /// map from a loop PreviousPort to the variables representing its derivative sources
        let dPreviousVars = Dictionary<PreviousPortT, VarSpecT> ()

        let dConstArgSumVars = Dictionary<int, VarSpecT> ()

        /// map from a loop port to the value it must contain
        let portContents = Dictionary<LoopPortT, PortContentsT> ()

        /// map from argument index to the loop ports containing its derivative summands
        let argIdxDerivs = Dictionary<int, System.Collections.Generic.HashSet<LoopDerivT>> ()



        // go through loop outputs and create variables representing their derivatives
        for KeyValue (outPort, dExpr) in dOutputs do
            // create variable for incoming Jacobian
            let value = spec.Ports.[outPort]
            let dName = sprintf "d_%s" outPort
            let dVar =
                VarSpec.create dName value.Expr.Type (funElems :: value.Expr.Shape)
            dOutputVars.[outPort] <- dVar

            // create variable input specification:
            // source of incoming Jacobian is sequence of derivatives of the loop output
            let sas = {
                ArgIdx = addArg dOutputs.[outPort]
                SliceDim = value.SliceDim
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
                let dAccumVar = VarSpec.create dAccumName liType [funElems; liElems]

                // create loop port exposing the step Jacobian plus the accumulated Jacobian w.r.t. ConstArg argIdx
                let dPortName = sprintf "dSum_ConstArg%d" argIdx
                if not (portContents.ContainsKey dPortName) then
                    portContents.[dPortName] <- {DerivWrt=ResizeArray<_>(); ValueOf=Some dAccumVar; SliceDim=liDims+1}
                portContents.[dPortName].DerivWrt.Add usingVar

                // create variable input specification:
                // source is accumulated Jacobian w.r.t. ConstArg argIdx in previous derivative loop iteration
                let dpp = {
                    Port = dPortName
                    Delay = SizeSpec.one
                    Initial = InitialZero
                }
                varInputSpecs.Add (dAccumVar, PreviousPort dpp)

                // set Jacobian w.r.t. input argument argIdx specification
                let slice = [
                    yield RSAll                         // function element axis
                    for d=0 to liDims-1 do yield RSAll  // derivative axes
                    yield RSSymElem (spec.Length - 1)   // sequence slice axis
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

            | PreviousPort pp ->
                // create loop port exposing the derivative w.r.t. the PreviousPort
                let dPortName = sprintf "d_%s[%A]" pp.Port pp.Delay
                if not (portContents.ContainsKey dPortName) then
                    let sliceDim = spec.Ports.[pp.Port].SliceDim
                    portContents.Add (dPortName, {DerivWrt=ResizeArray<_>(); ValueOf=None; SliceDim=sliceDim})
                portContents.[dPortName].DerivWrt.Add usingVar

                // create a variable for Jacobian coming from a PreviousPort in a (future) loop iteration
                let dVar = VarSpec.create dPortName liType (funElems :: liShape)
                dPreviousVars.[pp] <- dVar

                // create corresponding variable input specification:
                // source is Jacobian calculated w.r.t. the PreviousPort in previous derivative loop iteration
                let dpp = {
                    Port = dPortName
                    Delay = pp.Delay
                    Initial = InitialZero
                }
                varInputSpecs.Add (dVar, PreviousPort dpp)                                 

                // check initial value
                match pp.Initial with
                | InitialArg argIdx ->
                    // If initial value(s) is specified by a sequence argument,
                    // we need to output the Jacboian w.r.t. to the initial sequence argument.
                    // It is available in the last "Delay" steps of the derivative loop port.
                    let sliceDim = spec.Ports.[pp.Port].SliceDim
                    let slice = [
                        yield RSAll                                 // function element axis
                        for d=0 to sliceDim-1 do yield RSAll        // derivative axes
                        yield RSSymStartSymEnd                      // sequence slice axis
                            (Some (spec.Length - pp.Delay),
                             Some (spec.Length - 1))                
                        for d=sliceDim to liDims-1 do yield RSAll   // derivative axes
                    ]
                    argIdxDerivs.[argIdx].Add {Port=dPortName; Slice=slice; ReverseAxis=Some (sliceDim+1)} |> ignore
                | InitialZero -> 
                    // For zero initial value, no Jacobian propagation needs to be done.
                    ()
                                               
            | IterationIndex 
            | IterationsRemaining -> 
                // iteration index is an intergral constant
                ()        

            
        /// derivatives of all ports w.r.t. all variables
        let portDerivs =
            spec.Ports
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
                            if previousPort.Port = port then yield Expr.makeVar dVar
                    } |> Seq.reduce (+)
                    
                // collapse Jacobian
                let incomingJacobian = incomingExpandedJacobian |> Expr.reshape [funElems; value.Expr.NElems]

                // calculate Jacobians w.r.t. all variables
                reverseDiff value.Expr value.Expr incomingJacobian)    
            |> Seq.reduce merge


        // go through portContents and create actual port contents
        let ports =
            portContents
            |> Seq.map (fun pc ->
                let port, {DerivWrt=derivWrts; ValueOf=valueOf; SliceDim=sliceDim} = pc.Key, pc.Value
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
                port, {Expr=expr; SliceDim=sliceDim})
            |> Map.ofSeq


        // create variable specification
        let varsFromDeriv = 
            varInputSpecs
            |> Seq.map (fun vis -> vis.Key, vis.Value)
            |> Map.ofSeq

        // need to map original vars
        // 1. ConstArg stays as is
        // 2. SequenceArgSlice gets remapped to reversed SequeceArgSlice
        // 3. PreviousPort gets remapped to output sequence of original op with appropriate delay and reversed
        // 4. IterationIndex gets remapped to IterationsRemaining
        // 5. IterationsRemaining gets remapped to IterationIndex

        let originalVars =
            spec.Vars
            |> Map.map (fun vs li ->
                match li with
                | ConstArg _ -> li
                | SequenceArgSlice {ArgIdx=argIdx; SliceDim=sliceDim} ->
                    let revExpr = Expr.reverseAxis sliceDim args.[argIdx]
                    SequenceArgSlice {ArgIdx=addArg revExpr; SliceDim=sliceDim}
                | PreviousPort pp ->
                    let portOutput = Expr.loop spec pp.Port originalArgs
                    let portExpr = spec.Ports.[pp.Port].Expr
                    let sliceDim = spec.Ports.[pp.Port].SliceDim

                    let initialValues =
                        match pp.Initial with
                        | InitialZero -> 
                            let initialShp = portOutput.Shape |> ShapeSpec.set sliceDim pp.Delay
                            Expr.zerosOfSameType portExpr initialShp
                        | InitialArg initialArgIdx ->
                            originalArgs.[initialArgIdx]

                    let portSeq = Expr.concat sliceDim [initialValues; portOutput]
                    let revPortSeq = portSeq |> Expr.reverseAxis sliceDim

                    let delaySlice : FullExprRngsSpecT = [
                        for d=0 to sliceDim-1 do yield RSAll 
                        yield RSSymStartSymEnd (Some pp.Delay, None)
                        for d=sliceDim to portExpr.NDims-1 do yield RSAll
                    ]
                    let delayedPortSeq = revPortSeq.[delaySlice]

                    SequenceArgSlice {ArgIdx=addArg delayedPortSeq; SliceDim=sliceDim}
                | IterationIndex -> IterationsRemaining
                | IterationsRemaining -> IterationIndex)

        let vars = Map.join originalVars varsFromDeriv

        let dSpec = {
            Length = spec.Length
            Vars   = vars
            Ports  = ports
        }

        // build derivatives w.r.t. our arguments
        let argIdxDerivExprs = 
            argIdxDerivs 
            |> Seq.map (fun aid -> 
                let argIdx, loopDerivs = aid.Key, aid.Value
                let dExpr =
                    loopDerivs
                    |> Seq.map (fun {Port=port; Slice=slice; ReverseAxis=reverseAxis} ->
                        let loopOutput = Expr.loop dSpec port (List.ofSeq args)
                        let sliced = loopOutput.[slice]
                        match reverseAxis with
                        | Some ax -> sliced |> Expr.reverseAxis ax
                        | None -> sliced)
                    |> Seq.reduce (+)
                argIdx, dExpr)
            |> Map.ofSeq

        let derivExprs = [
            for a=0 to originalArgs.Length-1 do
                yield argIdxDerivExprs.[a]
        ]

        derivExprs



