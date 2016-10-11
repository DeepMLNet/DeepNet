namespace SymTensor

open System.Diagnostics
open System.Reflection
open System.Collections.Generic

open Basics
open Expr


[<AutoOpen>]
module UExprTypes = 

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
        Length:     int
        Vars:       Map<VarSpecT, LoopInputT>
        Channels:   Map<ChannelT, ULoopValueT>
    }

    /// ops that have special handling
    and UExtraOpT =
        | Subtensor of UExprRngsSpecT 
        | SetSubtensor of UExprRngsSpecT
        | Elements of ShapeSpecT * UElemExpr.UElemFuncT
        | IfThenElse
        | Loop of ULoopSpecT
        | Channel of ChannelT
        | ExtensionExtraOp of IUOp        

    /// unified op of any arity and type
    and UOpT =
        | ULeafOp of Expr.LeafOpT
        | UUnaryOp of Expr.UnaryOpT
        | UBinaryOp of Expr.BinaryOpT
        | UNaryOp of Expr.NaryOpT
        | UExtraOp of UExtraOpT

    /// unified expression (combines all arities and types and ops cannot have expressions as parameters)    
    and [<StructuralComparison; StructuralEquality; StructuredFormatDisplay("{Pretty}")>]
        UExprT = 
        | UExpr of UOpT * (UExprT list) * UMetadata

        /// pretty string
        member this.Pretty =
            match this with
            | UExpr (ULeafOp uop, subs, _) -> sprintf "{%A}" uop 
            | UExpr (UUnaryOp uop, subs, _) -> sprintf "{%A} (%A)" uop subs.[0]
            | UExpr (UBinaryOp uop, subs, _) -> sprintf "{%A} (%A, %A)" uop subs.[0] subs.[1]
            | UExpr (UNaryOp uop, subs, _) -> sprintf "{%A} (%A)" uop subs
            | UExpr (UExtraOp uop, subs, _) -> sprintf "{%A} (%A)" uop subs

        member this.Op = match this with UExpr (op, _, _) -> op
        member this.Args = match this with UExpr (_, args, _) -> args
        member this.Metadata = match this with UExpr (_, _, metadata) -> metadata
        member this.ChannelType = this.Metadata.ChannelType
        member this.ChannelShape = this.Metadata.ChannelShape
        member this.Expr = this.Metadata.Expr       

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


module UExpr =

    type UExprCaches = {
        UExprForExpr:       Dictionary<ExprT, UExprT>
        UExprs:             Dictionary<UExprT, UExprT>
    }

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
                    extra (Subtensor usr) (a :: dynExprs)
                | Expr.Unary (Expr.NullifyJacobian, a) -> toUExprRec a
                | Expr.Unary (Expr.AssumeJacobian _, a) -> toUExprRec a
                | Expr.Binary (Expr.SetSubtensor sr, a, b) ->
                    let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr   
                    extra (SetSubtensor usr) (a :: b :: dynExprs)
                | Expr.Binary (Expr.IfThenElse cond, ifTrue, ifFalse) ->
                    extra IfThenElse [ifTrue; ifFalse; cond]
                | Expr.Nary (Expr.Elements (resShape, elemExpr), se) ->
                    let nDims = ShapeSpec.nDim resShape
                    let nArgs = List.length se
                    extra (Elements (resShape, UElemExpr.toUElemFunc elemExpr nDims nArgs)) se
                | Expr.Nary (Expr.Channel (Expr.Loop loopSpec, channel), se) ->
                    // build separate loop op
                    let uLoopSpec = loopSpecToULoopSpec loopSpec
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
                | Expr.Unary (Expr.Held (_, heldOp), a) ->
                    failwithf "the held op %A must be expanded before conversion to UExpr" heldOp
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

    /// converts an expression to a unified expression
    and toUExprWithCache caches (expr: ExprT) =
        expr
        |> Expr.check
        |> toUExprRec caches 

    /// creates an unified expression cache cache
    and createCache () = 
        {
            UExprForExpr    = Dictionary<ExprT, UExprT>(HashIdentity.Structural)
            UExprs          = Dictionary<UExprT, UExprT>(HashIdentity.Structural)        
        }     

    /// converts an expression to a unified expression
    and toUExpr (expr: ExprT) =
        let caches = createCache ()
        toUExprWithCache caches expr

    /// converts a loop specification to an unified loop specification
    and loopSpecToULoopSpec (loopSpec: LoopSpecT) = 
        {
            Length   = SizeSpec.eval loopSpec.Length
            Vars     = loopSpec.Vars
            Channels = loopSpec.Channels 
                        |> Map.map (fun ch lv ->
                                {UExpr=toUExpr lv.Expr; SliceDim=lv.SliceDim})
        }           

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

