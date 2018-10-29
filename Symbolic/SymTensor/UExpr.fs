namespace SymTensor

open System.Diagnostics
open System.Reflection
open System.Collections.Generic

open Tensor.Utils
open Tensor
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
        UExpr:       UExprT
        SliceDim:    int
        OutputFrom:  int64
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
        

/// Functions for dealing with unified expressions.
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
            UExprForExpr    = Dictionary<ExprT, UExprT> () 
            UExprs          = Dictionary<UExprT, UExprT> ()        
            ULoopSpecs      = Dictionary<LoopSpecT, ULoopSpecT> () 
        }     
        exprs 
        |> List.map Expr.check
        |> List.map (toUExprRec caches)
        |> trimULoops

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
                |> List.map (fun ((ch, lv), uexpr) -> 
                    ch, {UExpr=uexpr; SliceDim=lv.SliceDim; OutputFrom=0L})
            let uLoopSpec = {
                Length   = SizeSpec.eval loopSpec.Length
                Vars     = loopSpec.Vars
                Channels = uChannels |> Map.ofList
            }          
            caches.ULoopSpecs.[loopSpec] <- uLoopSpec
            uLoopSpec

    /// removes unused channels from loops and trims their outputs when it is sliced
    and private trimULoops (uexprs: UExprT list) =
        // build set of used channels and slices
        let loopChFirst = 
            Dictionary<UExprT, Dictionary<ChannelT, int64>> (HashIdentity.Reference)

        let regLoopChSlice loopExpr channel first =
            if not (loopChFirst.ContainsKey loopExpr) then
                loopChFirst.[loopExpr] <- Dictionary<_, _> ()
            loopChFirst.[loopExpr].[channel] <-
                match loopChFirst.[loopExpr].TryFind channel with
                | Some pFirst -> min pFirst first
                | None -> first       

        let getChFirst loopExpr loopSpec channel =
            let maxDelay = 
                loopSpec.Vars
                |> Map.toSeq
                |> Seq.choose (function
                                | _, PreviousChannel pCh when pCh.Channel=channel -> 
                                    Some (SizeSpec.eval pCh.Delay)
                                | _ -> None)
                |> Seq.fold max 0L   
            match loopChFirst.[loopExpr].TryFind channel with
            | Some first -> min first (max 0L (loopSpec.Length - maxDelay))
            | None -> loopSpec.Length - maxDelay

        let visited = HashSet<UExprT> (HashIdentity.Reference)           
        let rec buildChFirst uexpr =
            if not (visited.Contains uexpr) then
                match uexpr with
                // slice of loop channel output
                | UExpr (UExtraOp (Subtensor (SimpleRangesSpec.Static as rng)), 
                         [UExpr (UExtraOp (Channel ch), 
                                 [UExpr (UExtraOp (Loop loopSpec), loopArgs, _) as loopExpr], _)], _) ->
                    let sliceRng = 
                        rng.[loopSpec.Channels.[ch].SliceDim] 
                        |> SimpleRangeSpec.eval (fun _ -> failwith "static")
                    let first = 
                        match sliceRng with
                        | Rng.Rng (Some first, _) -> first
                        | Rng.Rng (None, _) -> 0L
                        | Rng.Elem elem -> elem
                        | Rng.NewAxis | Rng.AllFill -> failwith "unexpected range"
                    regLoopChSlice loopExpr ch first               
                    for arg in loopArgs do
                        buildChFirst arg
                // full loop channel output
                | UExpr (UExtraOp (Channel ch), 
                         [UExpr (UExtraOp (Loop loopSpec), loopArgs, _) as loopExpr], _) ->
                    regLoopChSlice loopExpr ch 0L
                    for arg in loopArgs do
                        buildChFirst arg
                // other expression
                | UExpr (_, args, _) -> 
                    for arg in args do
                        buildChFirst arg
                visited.Add uexpr |> ignore
        uexprs |> List.iter buildChFirst 

        let processed = Dictionary<UExprT, UExprT> (HashIdentity.Reference)
        let rec rebuild uexpr =
            match processed.TryFind uexpr with
            | Some repl -> repl
            | None ->
                let repl = 
                    match uexpr with
                    // adjust ranges of loop output slices
                    | UExpr (UExtraOp (Subtensor (SimpleRangesSpec.Static as rng)), 
                             [UExpr (UExtraOp (Channel ch), 
                                     [UExpr (UExtraOp (Loop loopSpec), loopArgs, _) as loopExpr], 
                                     _) as channelExpr], metadata) ->
                        let sliceDim = loopSpec.Channels.[ch].SliceDim
                        let offset = getChFirst loopExpr loopSpec ch
                        let offsetSliceRng =
                            match rng.[sliceDim] with
                            | SRSSymStartSymEnd (first, Some last) -> 
                                SRSSymStartSymEnd (first - offset, Some (last - offset))
                            | SRSSymStartSymEnd (first, None) -> 
                                SRSSymStartSymEnd (first - offset, None)
                            | _ -> failwith "static range expected"
                        let offsetRng = rng |> List.set sliceDim offsetSliceRng
                        //printfn "Adjusting channel %A slice from %A to %A." ch rng offsetRng
                        UExpr (UExtraOp (Subtensor offsetRng), [rebuild channelExpr], metadata)
                    // update channel selection metadata due to shape change
                    | UExpr (UExtraOp (Channel ch), [UExpr (UExtraOp (Loop _), _, _) as loopExpr], metadata) ->
                        let UExpr (_, _, loopMetadata) as loopRepl = rebuild loopExpr
                        let metadata = 
                            {metadata with ChannelShape = Map [dfltChId, loopMetadata.ChannelShape.[ch]]}
                        UExpr (UExtraOp (Channel ch), [loopRepl], metadata)
                    // remove unused channels from loop specification and update output range
                    | UExpr (UExtraOp (Loop loopSpec), loopArgs, metadata) as loopExpr ->
                        let rec usedChannelsAndVars prevUsedChs =
                            let usedVars =
                                loopSpec.Channels
                                |> Map.filter (fun ch _ -> prevUsedChs |> Set.contains ch)
                                |> Map.toSeq
                                |> Seq.map (fun (ch, lv) -> extractVars lv.UExpr) 
                                |> Set.unionMany
                            let usedChs =
                                (prevUsedChs, usedVars)
                                ||> Seq.fold (fun ucs vs ->
                                    match loopSpec.Vars.[vs] with
                                    | PreviousChannel {Channel=channel} -> ucs |> Set.add channel
                                    | _ -> ucs)     
                            if usedChs <> prevUsedChs then usedChannelsAndVars usedChs
                            else usedChs, usedVars
                        let usedChs, usedVars = usedChannelsAndVars (Set.ofSeq loopChFirst.[loopExpr].Keys)               
                        let filterChs chMap = chMap |> Map.filter (fun ch _ -> usedChs.Contains ch) 
                        let trimChs chMap =                             
                            chMap |> Map.map (fun ch lv -> 
                                {lv with OutputFrom = lv.OutputFrom + getChFirst loopExpr loopSpec ch})
                        let trimChShapes chShps =
                            chShps |> Map.map (fun ch (shp: NShapeSpecT) ->
                                let sliceDim = loopSpec.Channels.[ch].SliceDim
                                let outputSize = shp.[sliceDim] - getChFirst loopExpr loopSpec ch
                                let trimedShp = shp |> List.set sliceDim outputSize
                                //printfn "Trimming loop channel %A from %A to %A." ch shp trimedShp
                                trimedShp)                                
                        let uLoopSpec = 
                            {loopSpec with 
                                Vars     = loopSpec.Vars |> Map.filter (fun var _ -> usedVars.Contains var)
                                Channels = loopSpec.Channels |> filterChs |> trimChs}
                        let metadata = 
                            {metadata with
                                ChannelShape = metadata.ChannelShape |> filterChs |> trimChShapes
                                ChannelType  = metadata.ChannelType |> filterChs}
                        UExpr (UExtraOp (Loop uLoopSpec), loopArgs |> List.map rebuild, metadata)
                    // other expression
                    | UExpr (op, args, metadata) -> 
                        UExpr (op, args |> List.map rebuild, metadata)
                processed.[uexpr] <- repl
                processed.[repl] <- repl
                repl
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


/// Information about a unified expression.
type UExprInfoT (expr: UExprT) =
      
    // build sets of dependants for each subexpression
    let dependants = 
        let processed = HashSet<UExprT> (HashIdentity.Reference)
        let dependants = Dictionary<UExprT, ResizeArray<UExprT>> (HashIdentity.Reference)              
        let addDependant node dependant =
            if not (dependants.ContainsKey node) then
                dependants.[node] <- ResizeArray<UExprT> ()
            dependants.[node].Add dependant
        let rec doBuild (UExpr(_, args, _) as expr) =
            if not (processed.Contains expr) then
                // update dependants recursively
                for arg in args do
                    addDependant arg expr
                for arg in args do
                    doBuild arg
                processed.Add expr |> ignore
        doBuild expr
        dependants

    /// Contained unified expression.
    member this.Expr = expr 

    /// Returns all expressions that depend on expr.
    /// A dependant will occur as many times as it references expr through its arguments.
    member this.Dependants expr =
        match dependants.TryFind expr with
        | Some deps -> deps.AsReadOnly()
        | None -> (ResizeArray<_> ()).AsReadOnly()
