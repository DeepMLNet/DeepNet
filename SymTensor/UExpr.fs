namespace SymTensor

open System.Diagnostics
open System.Reflection
open System.Collections.Generic

open Basics
open Expr


[<AutoOpen>]
module UExprTypes = 

    let private uExprHashCache = Dictionary<obj, int> (HashIdentity.Reference)

    /// default channel
    let dfltChId : ChannelT = "#"

    // int holds the position of the subuexpr that has the dynamic value
    type UExprRngSpecT = SimpleRangeSpecT<int>
    type UExprRngsSpecT = SimpleRangesSpecT<int>

    /// An operation in an unified expression.
    type IUOp =
        inherit System.IComparable

    /// metadata for an unified expression
    type UMetadata = {
        /// the data type of the result channels
        ChannelType:   Map<ChannelT, TypeNameT>
        /// the numeric shape of the result channels
        ChannelShape:  Map<ChannelT, NShapeSpecT>
        /// the generating expression, if created from one
        Expr:          Expr.ExprT option
    }

    type ULoopValueT = {
        UExpr:      UExprT
        SliceDim:   int
    }

    and ULoopSpecT = {
        Length:     int64
        Vars:       Map<VarSpecT, LoopInputT>
        Channels:   Map<ChannelT, ULoopValueT>
    }

    and IndexArgs = int option list

    /// ops that have special handling
    and UExtraOpT =
        | Subtensor of UExprRngsSpecT 
        | SetSubtensor of UExprRngsSpecT
        | Elements of ShapeSpecT * UElemExpr.UElemFuncT
        | IfThenElse
        | Loop of ULoopSpecT
        | Channel of ChannelT
        | Gather of IndexArgs
        | Scatter of IndexArgs
        | ExtensionExtraOp of IUOp        

    /// unified op of any arity and type
    
    and UOpT =
        | ULeafOp of Expr.LeafOpT
        | UUnaryOp of Expr.UnaryOpT
        | UBinaryOp of Expr.BinaryOpT
        | UNaryOp of Expr.NaryOpT
        | UExtraOp of UExtraOpT

    /// unified expression (combines all arities and types and ops cannot have expressions as parameters)    
    and [<CustomComparison; CustomEquality; StructuredFormatDisplay("{Pretty}")>]
        UExprT = 
        | UExpr of UOpT * (UExprT list) * UMetadata

        member inline this.Op = match this with UExpr (op, _, _) -> op
        member inline this.Args = match this with UExpr (_, args, _) -> args
        member this.Metadata = match this with UExpr (_, _, metadata) -> metadata
        member this.ChannelType = this.Metadata.ChannelType
        member this.ChannelShape = this.Metadata.ChannelShape
        member this.Expr = this.Metadata.Expr       

        member inline private this.Proxy = 
            match this with UExpr (op, args, _) -> op, args

        // avoid comparing and hashing metadata
        override this.Equals other =
            match other with
            | :? UExprT as other -> (this :> System.IEquatable<_>).Equals other
            | _ -> false
        interface System.IEquatable<UExprT> with
            member this.Equals other = 
                if obj.ReferenceEquals (this, other) then true
                else this.Proxy = other.Proxy
        override this.GetHashCode() =
            match uExprHashCache.TryFind this with
            | Some h -> h
            | None ->
                let h = hash this.Proxy
                uExprHashCache.[this] <- h
                h
        interface System.IComparable<UExprT> with
            member this.CompareTo other =
                compare this.Proxy other.Proxy
                //compare (this.Op, this.Args) (other.Op, other.Args)
        interface System.IComparable with
            member this.CompareTo other =
                match other with
                | :? UExprT as other -> (this :> System.IComparable<_>).CompareTo other
                | _ -> failwithf "cannot compare UExprT to type %A" (other.GetType())

        /// pretty string
        member this.Pretty =
            match this with
            | UExpr (ULeafOp uop, subs, _) -> sprintf "{%A}" uop 
            | UExpr (UUnaryOp uop, subs, _) -> sprintf "{%A} (%A)" uop subs.[0]
            | UExpr (UBinaryOp uop, subs, _) -> sprintf "{%A} (%A, %A)" uop subs.[0] subs.[1]
            | UExpr (UNaryOp uop, subs, _) -> sprintf "{%A} (%A)" uop subs
            | UExpr (UExtraOp uop, subs, _) -> sprintf "{%A} (%A)" uop subs


    /// An IOp that can be converted to an unified expression for compilation.
    type ICompilableOp =
        inherit IOp

        /// Should create a unified expression from the given expression.
        /// This op is always the root of the passed expression.
        /// If there is a one-to-one relationship to a unified op, call the makeOneUop function
        /// with the corresponding Uop. It will generate the apropriate unified expression.
        abstract ToUExpr: expr:ExprT -> makeOneUop:(IUOp -> UExprT) -> UExprT


module UExprRngsSpec =

    // split into two ops:
    // one that does nothing, just changes the static layout
    // and another that does the copying if necessary
    // op1 : StaticSubtensor
    // op2 : DynamicSubtensor
    // but how does this work with SetSubtensor?

    /// converts a ExprRngsSpecT to a UExprRngSpecT
    let ofExprRngsSpec (sr: ExprRngsSpecT) =
        ([], sr)
        ||> List.mapFold (fun dynExprs rng ->
            let idx = List.length dynExprs 
            match rng with
            | SRSSymStartSymEnd  (s, fo)     -> SRSSymStartSymEnd (s, fo),       dynExprs
            | SRSDynStartSymSize (s, size)   -> SRSDynStartSymSize (idx, size),  dynExprs @ [s])

    /// converts a UExprRngSpecT to a ExprRngsSpecT
    let rec toExprRngsSpec (srs: UExprRngsSpecT) (drs: ExprT list)  =
        if drs |> List.exists (fun dr -> Expr.typename dr <> TypeName.ofType<int>) then
            failwith "need inttype for range spec"
        match srs, drs with
        | SRSSymStartSymEnd (s, fo) :: srs, _         -> SRSSymStartSymEnd (s, fo)   :: toExprRngsSpec srs drs
        | SRSDynStartSymSize (_, f) :: srs, dr :: rdrs-> SRSDynStartSymSize (dr, f)  :: toExprRngsSpec srs rdrs
        | []                              , []        -> []
        | _                               , _         -> failwith "invalid unified subtensor spec"

    /// checks that the static parts of the range specification are compatible with the given shape
    let checkCompatibility (shp: ShapeSpecT) (srs: UExprRngsSpecT) =
        let shp = ShapeSpec.eval shp
        let failRng () =
            failwithf "Subtensor range specification %A is invalid for tensor of shape %A." srs shp
        if shp.Length <> srs.Length then failRng ()
        (shp, srs) ||> List.iter2 (fun size rng ->           
            match rng with
            | SRSSymStartSymEnd (s, fo) ->
                let s, fo = SizeSpec.eval s, Option.map SizeSpec.eval fo
                if not (0L <= s && s < size) then failRng ()
                match fo with
                | Some fo when not (0L <= fo && fo < size && fo >= s-1L) -> failRng ()
                | _ -> ()
            | SRSDynStartSymSize _ -> ())        
        

module UExpr =

    type private UExprCaches = {
        UExprForExpr:       Dictionary<ExprT, UExprT>
        UExprs:             Dictionary<UExprT, UExprT>
        ULoopSpecs:         Dictionary<LoopSpecT, ULoopSpecT>
    }

    /// extracts all variables from the unified expression
    let rec extractVars (UExpr (op, args, metadata)) = 
        match op with
        | ULeafOp (Expr.Var vs) -> Set.singleton vs
        | _ -> args |> List.map extractVars |> Set.unionMany

    let internal indicesToIdxArgs indices =
        let idxArgNos, _ =
            (1, indices)
            ||> List.mapFold (fun argNo idx ->
                match idx with
                | Some _ -> Some argNo, argNo + 1
                | None -> None, argNo)
        let idxArgs = indices |> List.choose id
        idxArgNos, idxArgs

    let rec private toUExprRec (caches: UExprCaches) (expr: ExprT) =
        match caches.UExprForExpr.TryFind expr with
        | Some cached -> cached
        | None ->
            let toUExprRec = toUExprRec caches
            let metadata = {
                ChannelType     = Map [dfltChId, Expr.typename expr]
                ChannelShape    = Map [dfltChId, Expr.shapeOf expr |> ShapeSpec.eval]
                Expr            = Some expr
            }
            let extra uop se    = UExpr (UExtraOp uop, se |> List.map toUExprRec, metadata)

            let uExpr =
                match expr with
                // ops that need special handling
                | Expr.Unary (Expr.Subtensor sr, a)  ->
                    let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr    
                    usr |> UExprRngsSpec.checkCompatibility a.Shape
                    extra (Subtensor usr) (a :: dynExprs)
                | Expr.Unary (Expr.NullifyJacobian, a) -> toUExprRec a
                | Expr.Unary (Expr.AssumeJacobian _, a) -> toUExprRec a
                | Expr.Binary (Expr.SetSubtensor sr, a, b) ->
                    let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr   
                    usr |> UExprRngsSpec.checkCompatibility a.Shape
                    extra (SetSubtensor usr) (a :: b :: dynExprs)
                | Expr.Binary (Expr.IfThenElse cond, ifTrue, ifFalse) ->
                    extra IfThenElse [ifTrue; ifFalse; cond]
                | Expr.Nary (Expr.Elements (resShape, elemExpr), se) ->
                    let nDims = ShapeSpec.nDim resShape
                    let nArgs = List.length se
                    extra (Elements (resShape, UElemExpr.toUElemFunc elemExpr nDims nArgs)) se
                | Expr.Nary (Expr.Channel (Expr.Loop loopSpec, channel), se) ->
                    // build separate loop op
                    let uLoopSpec = loopSpecToULoopSpec caches loopSpec
                    let uLoopMetadata = {
                        ChannelType  = Expr.loopOutputTypeNames loopSpec
                        ChannelShape = Expr.loopOutputShapes loopSpec |> Map.map (fun ch shp -> ShapeSpec.eval shp)
                        Expr         = None
                    }
                    let uLoop = UExpr (UExtraOp (Loop uLoopSpec), se |> List.map toUExprRec, uLoopMetadata)

                    // try to find loop in cache for reference equality guarantee
                    let uLoop =
                        match caches.UExprs.TryFind uLoop with
                        | Some cached -> cached
                        | None -> caches.UExprs.[uLoop] <- uLoop; uLoop

                    // and separate op to extract referenced channel
                    UExpr (UExtraOp (Channel channel), [uLoop], metadata)
                | Expr.Unary (Expr.Gather indices, a) ->                
                    let idxArgNos, idxArgs = indicesToIdxArgs indices
                    extra (Gather idxArgNos) (a::idxArgs)
                | Expr.Unary (Expr.Scatter (indices, _), a) ->
                    let idxArgNos, idxArgs = indicesToIdxArgs indices
                    extra (Scatter idxArgNos) (a::idxArgs)
                | Expr.Unary (Expr.Held (_, heldOp) as holdOp, a) ->
                    failwithf "the held op %A must be expanded before conversion to UExpr \
                              (shape of argument is %A and Hold is %A)" heldOp a.Shape holdOp
                | Expr.Nary (Expr.ExtensionOp eop, se) -> 
                    match eop with
                    | :? ICompilableOp as eop ->
                        let makeOneUop uop = extra (ExtensionExtraOp uop) se
                        eop.ToUExpr expr makeOneUop
                    | _ -> UExpr (UNaryOp (Expr.ExtensionOp eop), se |> List.map toUExprRec, metadata) 
                    
                // all other ops are just copied over
                | Expr.Leaf op -> UExpr (ULeafOp op, [], metadata)
                | Expr.Unary (op, a) -> UExpr (UUnaryOp op, [toUExprRec a], metadata)
                | Expr.Binary (op, a, b) -> UExpr (UBinaryOp op, [toUExprRec a; toUExprRec b], metadata)
                | Expr.Nary (op, es) -> UExpr (UNaryOp op, es |> List.map toUExprRec, metadata)        

            // try to find UExpr that was already produced before and reuse it to guarantee
            // reference equality if structural equality holds
            let uExpr =
                match caches.UExprs.TryFind uExpr with
                | Some cached -> cached
                | None -> caches.UExprs.[uExpr] <- uExpr; uExpr
           
            caches.UExprForExpr.[expr] <- uExpr     
            uExpr       

    /// converts a list of expressions to a list of unified expressions
    and toUExprs (exprs: ExprT list) =
        let caches = {
            UExprForExpr    = Dictionary<ExprT, UExprT> () //(HashIdentity.LimitedStructural 10)
            UExprs          = Dictionary<UExprT, UExprT>()        
            ULoopSpecs      = Dictionary<LoopSpecT, ULoopSpecT> () //(HashIdentity.LimitedStructural 10)
        }     
        exprs 
        |> List.map Expr.check
        |> List.map (toUExprRec caches)
        |> removeUnusedChannels

    /// converts an expression to a unified expression
    and toUExpr (expr: ExprT) =
        toUExprs [expr] |> List.exactlyOne        

    /// converts a loop specification to an unified loop specification
    and private loopSpecToULoopSpec (caches: UExprCaches) (loopSpec: LoopSpecT) = 
        match caches.ULoopSpecs.TryFind loopSpec with
        | Some uLoopSpec -> uLoopSpec
        | None ->
            let channels = loopSpec.Channels |> Map.toList 
            let chUExprs = toUExprs (channels |> List.map (fun (_, lv) -> lv.Expr))
            let uChannels = 
                List.zip channels chUExprs
                |> List.map (fun ((ch, lv), uexpr) -> ch, {UExpr=uexpr; SliceDim=lv.SliceDim})
            let uLoopSpec = {
                Length   = SizeSpec.eval loopSpec.Length
                Vars     = loopSpec.Vars
                Channels = uChannels |> Map.ofList
            }          
            caches.ULoopSpecs.[loopSpec] <- uLoopSpec
            uLoopSpec

    /// removes unused loop channels from a loop expression
    and private removeUnusedLoopChannels (externallyUsedChannels: Set<ChannelT>) uexpr  =
        match uexpr with
        | UExpr (UExtraOp (Loop uLoopSpec), args, metadata) ->
            let rec usedChannelsAndVars prevUsedChs =
                let usedVars =
                    uLoopSpec.Channels
                    |> Map.filter (fun ch _ -> prevUsedChs |> Set.contains ch)
                    |> Map.toSeq
                    |> Seq.map (fun (ch, lv) -> extractVars lv.UExpr) 
                    |> Set.unionMany

                let usedChs =
                    (prevUsedChs, usedVars)
                    ||> Seq.fold (fun ucs vs ->
                        match uLoopSpec.Vars.[vs] with
                        | PreviousChannel {Channel=channel} -> ucs |> Set.add channel
                        | _ -> ucs)     

                if usedChs <> prevUsedChs then usedChannelsAndVars usedChs
                else usedChs, usedVars

            let usedChs, usedVars = usedChannelsAndVars externallyUsedChannels               
            let filterChs chMap = chMap |> Map.filter (fun ch _ -> usedChs.Contains ch) 
            let uLoopSpec = 
                {uLoopSpec with 
                    Vars     = uLoopSpec.Vars |> Map.filter (fun var _ -> usedVars.Contains var)
                    Channels = filterChs uLoopSpec.Channels}
            let metadata = 
                {metadata with
                    ChannelShape = filterChs metadata.ChannelShape
                    ChannelType  = filterChs metadata.ChannelType}
            UExpr (UExtraOp (Loop uLoopSpec), args, metadata)
        | _ -> failwith "not a loop expression"
        
    /// removes unused channels from multi-channels ops in a set of unified expressions
    and removeUnusedChannels (uexprs: UExprT list) =
        // build set of used channels
        let usedChannels = Dictionary<UExprT, HashSet<ChannelT>> (HashIdentity.Reference)
        let visited = HashSet<UExprT> (HashIdentity.Reference)
        let rec buildUsed (UExpr (op, args, _) as uexpr) =
            if not (visited.Contains uexpr) then
                match op with
                | UExtraOp (Channel ch) ->
                    let multichannelExpr = args.Head
                    if not (usedChannels.ContainsKey multichannelExpr) then
                        usedChannels.[multichannelExpr] <- HashSet<_> ()
                    usedChannels.[multichannelExpr].Add ch |> ignore
                | _ -> ()
                for arg in args do
                    buildUsed arg
                visited.Add uexpr |> ignore
        uexprs |> List.iter buildUsed 

        // filter unused channels
        let processed = Dictionary<UExprT, UExprT> (HashIdentity.Reference)
        let rec rebuild (UExpr (op, args, metadata) as origExpr) =
            match processed.TryFind origExpr with
            | Some replacement -> replacement
            | None ->
                let expr = UExpr (op, args |> List.map rebuild, metadata)
                let replacement = 
                    match op with
                    | UExtraOp (Loop _) -> 
                        removeUnusedLoopChannels (Set.ofSeq usedChannels.[origExpr]) expr
                    | _ -> expr
                processed.[origExpr] <- replacement
                processed.[replacement] <- replacement
                replacement
        uexprs |> List.map rebuild 

    /// Converts a unified expression to an expression if the unified expression
    /// was created using the toUExpr function. Otherwise returns None.
    let tryToExpr (UExpr (_, _, {Expr=exprOpt})) : ExprT option =
        match exprOpt with
        | Some exprObj -> Some (unbox exprObj)
        | None -> None

    /// Converts a unified expression to an expression.
    /// Only works if the unified expression was created using the toUExpr function.
    let toExpr uExpr =
        match tryToExpr uExpr with 
        | Some expr -> expr
        | None -> failwith "UExpr was not created from an Expr"

    /// the op of the given unified expression
    let op (UExpr(op, _, _)) = 
        op

    /// the type of the default channel of the given unified expression
    let dfltChType (UExpr(_, _, {ChannelType=ct})) = 
        ct.[dfltChId].Type

    /// the type of the default channel of the given unified expression
    let dfltChTypename (UExpr(_, _, {ChannelType=ct})) = 
        ct.[dfltChId]

    /// the shape of the default channel of the given unified expression
    let dfltChShape (UExpr(_, _, {ChannelShape=cs})) = 
        cs.[dfltChId]

    /// the shape of all channels of the given unified expression
    let channelShapes (UExpr(_, _, {ChannelShape=cs})) =
        cs

    /// counts unique subexpressions in a unified expression
    let countUniqueOps uexpr  =
        let visited = HashSet<UExprT> (HashIdentity.Reference)
        let rec doCount (UExpr (_, subExprs, _) as expr) =
            if visited.Contains expr then 0
            else
                visited.Add expr |> ignore

                subExprs
                |> List.map doCount
                |> List.sum
                |> fun n -> n + 1
        doCount uexpr

    /// counts how many times subExpr occurs in unified expression uexpr
    let subExprOccurrences uexpr =
        let cnt = Dictionary<UExprT, int>(HashIdentity.Reference)
        let rec build (UExpr (_, subExprs, _) as uexpr) =
            if cnt.ContainsKey(uexpr) then
                cnt.[uexpr] <- cnt.[uexpr] + 1
            else
                cnt.[uexpr] <- 1

            for subExpr in subExprs do
                build subExpr
        build uexpr

        fun subExpr ->
            if cnt.ContainsKey(subExpr) then cnt.[subExpr]
            else 0

