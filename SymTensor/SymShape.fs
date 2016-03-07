namespace SymTensor

open Basics

[<AutoOpen>]
module SizeSymbolTypes =

    /// a symbolic size
    [<StructuredFormatDisplay ("\"{Name}\"")>]
    type SizeSymbolT = {
        Name:       string;
        Flexible:   bool;
    }

    /// elementary size specification, can be either a symbol or a fixed quantity
    [<StructuredFormatDisplay ("{PrettyString}")>]
    type BaseSizeT =
        | Sym of SizeSymbolT
        | Fixed of int

        member this.PrettyString =
            match this with
            | Sym s -> sprintf "%A" s
            | Fixed f -> sprintf "%d" f


module SizeSymbol =
    [<Literal>]
    let AllowFlexible = false

    let name sym =
        sym.Name

    let isFlexible sym =
        AllowFlexible && sym.Flexible


//[<AutoOpen>]
module SizeProductTypes = 
      
    /// product of elementary size specifications
    [<StructuredFormatDisplay("{PrettyString}")>]
    type SizeProductT(symbols: Map<SizeSymbolT, int>) =
        let symbols =
            symbols 
            |> Map.filter (fun _ sPower ->
                if sPower < 0 then failwithf "SizeProduct cannot have negative exponents: %A" symbols
                else sPower > 0)
  
        new() = SizeProductT(Map.empty)
        new(b: SizeSymbolT) = SizeProductT(b, 1)
        new(b: SizeSymbolT, pow: int) = SizeProductT(Map.empty |> Map.add b pow)

        member this.Symbols = symbols

        static member (*) (a: SizeProductT, b: SizeProductT) =
            let pSymbols = 
                Map.fold 
                    (fun p bBase bPower -> 
                        match Map.tryFind bBase p with
                        | Some pPower -> Map.add bBase (pPower + bPower) p
                        | None -> Map.add bBase bPower p) 
                    a.Symbols b.Symbols
            SizeProductT(pSymbols)
         
        static member (*) (a: SizeProductT, b: SizeSymbolT) = a * SizeProductT(b)
        static member (*) (a: SizeSymbolT, b: SizeProductT) = SizeProductT(a) * b

        member this.PrettyString = 
            this.Symbols
            |> Map.toSeq
            |> Seq.map 
                (fun (tBase, tPower) ->
                    if tPower = 1 then sprintf "%A" tBase 
                    else sprintf "%A**%d" tBase tPower)
            |> Seq.toList
            |> String.concat " * "
                        
        override this.Equals(otherObj) =
            match otherObj with
            | :? SizeProductT as other -> this.Symbols = other.Symbols 
            | _ -> false

        override this.GetHashCode() =
            hash this.Symbols

        interface System.IComparable with
            member this.CompareTo otherObj =
                match otherObj with
                | :? SizeProductT as other -> compare this.Symbols other.Symbols
                | _ -> invalidArg "otherObj" "cannot compare values of different types"


module SizeProduct =
    open SizeProductTypes

    /// true if product is empty (equal to 1)
    let isEmpty (p: SizeProductT) =
        Map.isEmpty p.Symbols

    /// empty product (equal to 1)
    let empty =
        SizeProductT()

    /// matches if product consists of a single symbol with power 1
    let (|SingleSymbol|_|) (sp: SizeProductT) =
        let mc = Map.toList sp.Symbols
        match List.length mc with
        | 1 -> 
            let bs, power = mc.[0]
            if power = 1 then Some bs else None
        | _ -> None

    /// matches if product is empty (equal to 1)
    let (|Empty|_|) (sp: SizeProductT) =
        if isEmpty sp then Some () else None


//[<AutoOpen>]
module SizeMultinomTypes =
    open SizeProductTypes

    // symbolic multinomial
    [<StructuredFormatDisplay("{PrettyString}")>]
    type SizeMultinomT (products: Map<SizeProductT, int>) =
        let products = products |> Map.filter (fun _ fac -> fac <> 0)

        new (bs: BaseSizeT) = SizeMultinomT (1, bs, 1)
        new (bs: BaseSizeT, pow: int) = SizeMultinomT (1, bs, pow)
        new (fac: int, bs: BaseSizeT, pow: int) =
            let m =
                match bs with
                | Sym s -> Map.empty |> Map.add (SizeProductT (s, pow)) fac
                | Fixed f -> Map.empty |> Map.add (SizeProductT ()) (fac * (pown f pow))
            SizeMultinomT (m)

        member this.Products = products

        static member (~-) (a: SizeMultinomT) =
            a.Products 
            |> Map.map (fun _ fac -> -fac)
            |> SizeMultinomT 

        static member (+) (a: SizeMultinomT, b: SizeMultinomT) =
            (a.Products, b.Products)
            ||> Map.fold (fun res prod fac -> 
                match Map.tryFind prod res with
                | Some rFac -> res |> Map.add prod (fac + rFac)
                | None      -> res |> Map.add prod fac)
            |> SizeMultinomT
                
        static member (-) (a: SizeMultinomT, b: SizeMultinomT) = a + (-b)
        
        static member (*) (a: SizeMultinomT, b: SizeMultinomT) =
            seq { for KeyValue(ap, af) in a.Products do
                    for KeyValue(bp, bf) in b.Products do
                        yield ap*bp, af*bf }
            |> Map.ofSeq
            |> SizeMultinomT

        member this.PrettyString =
            products
            |> Map.toSeq
            |> Seq.map (fun (p, f) -> 
                if SizeProduct.isEmpty p then sprintf "%d" f
                elif f = 1 then sprintf "%A" p
                else sprintf "%d * %A" f p)
            |> String.concat " + "
        
        override this.Equals(otherObj) =
            match otherObj with
            | :? SizeMultinomT as other -> this.Products = other.Products 
            | _ -> false

        override this.GetHashCode() =
            hash this.Products

        interface System.IComparable with
            member this.CompareTo otherObj =
                match otherObj with
                | :? SizeMultinomT as other -> compare this.Products other.Products
                | _ -> invalidArg "otherObj" "cannot compare values of different types"


[<AutoOpen>]
module SizeSpecTypes =
    open SizeMultinomTypes

    /// symbolic size specification of a dimension (axis)
    [<StructuredFormatDisplay ("{PrettyString}")>]
    [<StructuralEquality; StructuralComparison>]
    type SizeSpecT =
        | Base of BaseSizeT               // fixed size or symbol
        | Broadcast                       // size 1 and broadcastable
        | Multinom of SizeMultinomT       // product of fixed sizes and symbols

        /// simplify size specification
        static member Simplify (ss: SizeSpecT) =
            match ss with
            | Multinom m -> 
                match m.Products |> Map.toList with
                | [SizeProduct.SingleSymbol s, f] when f = 1 -> Base (Sym s)
                | [SizeProduct.Empty, f] -> Base (Fixed f)
                | _ -> ss
            | _ -> ss

        static member (~-) (ssa: SizeSpecT) =
            match ssa with
            | Base (Fixed 0) -> ssa
            | Base b -> Multinom (-SizeMultinomT(b))
            | Broadcast -> Multinom (-SizeMultinomT(Fixed 1))
            | Multinom m -> Multinom (-m)
            |> SizeSpecT.Simplify

        static member (+) (ssa: SizeSpecT, ssb: SizeSpecT) =
            match ssa, ssb with
            | Base (Fixed 0), ss | ss, Base (Fixed 0) -> ss
            | Broadcast, ss | ss, Broadcast -> ss + (Base (Fixed 1))
            | Multinom ma, Multinom mb -> Multinom (ma + mb)
            | Multinom m, Base b | Base b, Multinom m -> Multinom (m + SizeMultinomT(b))
            | Base ba, Base bb -> Multinom (SizeMultinomT(ba) + SizeMultinomT(bb))
            |> SizeSpecT.Simplify

        static member (-) (ssa: SizeSpecT, ssb: SizeSpecT) =
            ssa + (-ssb)

        static member (*) (ssa: SizeSpecT, ssb: SizeSpecT) =
            match ssa, ssb with
            | Base (Fixed 0), _ | _, Base (Fixed 0) -> Base (Fixed 0)
            | Broadcast, ss | ss, Broadcast -> ss
            | Multinom ma, Multinom mb -> Multinom (ma * mb)
            | Multinom m, Base b | Base b, Multinom m -> Multinom (m * SizeMultinomT(b))
            | Base ba, Base bb -> Multinom (SizeMultinomT(ba) * SizeMultinomT(bb))
            |> SizeSpecT.Simplify

        static member Pow (ssa: SizeSpecT, pow: int) =
            match pow with
            | 0 -> Base (Fixed 1)
            | 1 -> ssa
            | _ ->
                match ssa with
                | Base (Fixed f) -> Base (Fixed (pown f pow))
                | Base (Sym s) -> Multinom (SizeMultinomT (Sym s, pow))
                | Broadcast -> Broadcast
                | Multinom m ->
                    m
                    |> Seq.replicate pow
                    |> Seq.reduce (*)
                    |> Multinom
            |> SizeSpecT.Simplify

        // operations with int
        static member (+) (ssa: SizeSpecT, ssb: int) = ssa + (Base (Fixed ssb))
        static member (+) (ssa: int, ssb: SizeSpecT) = (Base (Fixed ssa)) + ssb
        static member (-) (ssa: SizeSpecT, ssb: int) = ssa - (Base (Fixed ssb))
        static member (-) (ssa: int, ssb: SizeSpecT) = (Base (Fixed ssa)) - ssb
        static member (*) (ssa: SizeSpecT, ssb: int) = ssa * (Base (Fixed ssb))
        static member (*) (ssa: int, ssb: SizeSpecT) = (Base (Fixed ssa)) * ssb

        /// equal size with broadcastability
        static member (%=) (ssa: SizeSpecT, ssb: SizeSpecT) = 
            SizeSpecT.Simplify ssa = SizeSpecT.Simplify ssb 

        /// equal size ignoring broadcastability
        static member (.=) (ssa: SizeSpecT, ssb: SizeSpecT) = 
            match SizeSpecT.Simplify ssa, SizeSpecT.Simplify ssb with
            | Broadcast, Base (Fixed 1) | Base (Fixed 1), Broadcast -> true
            | a, b -> a = b

        /// unequal size ignoring broadcastability
        static member (.<>) (ssa: SizeSpecT, ssb: SizeSpecT) = not (ssa .= ssb)

        member this.PrettyString =
            match this with
            | Base b -> sprintf "%A" b
            | Broadcast -> "1*"
            | Multinom m -> sprintf "%A" m


module SizeSpec =
    open SizeSymbolTypes

    /// simplify size specification
    let simplify (ss: SizeSpecT) = SizeSpecT.Simplify ss

    /// True if both sizes have the same number of elements and 
    /// are both broadcastable or non-broadcastable.
    let equalWithBroadcastability (ssa: SizeSpecT) (ssb: SizeSpecT) = ssa %= ssb        

    /// True if both sizes have the same number of elements.
    /// Broadcastable and non-broadcastable are treated as equal.
    let equalWithoutBroadcastability (ssa: SizeSpecT) (ssb: SizeSpecT) = ssa .= ssb

    /// size zero
    let zero =
        Base (Fixed 0)

    /// not-broadcastable size one
    let one =
        Base (Fixed 1)

    /// fixed size
    let fix s =
        Base (Fixed s)

    /// symbolic size
    let symbol s =
        Base (Sym {Name=s; Flexible=false})

    /// flexible symbolic size
    let flexSymbol s =
        if not SizeSymbol.AllowFlexible then
            failwith "flexible symbol support is disabled"
        Base (Sym {Name=s; Flexible=true})

    /// broadcastable size one
    let broadcastable =
        Broadcast

    /// true if SizeSpec contains at least one flexible symbol
    let isFlexible ss =
        match ss with
        | Base (Sym sym) -> SizeSymbol.isFlexible sym
        | Base (Fixed _) -> false
        | Broadcast -> false
        | Multinom m ->            
            m.Products
            |> Map.exists (fun prod _ -> 
                prod.Symbols
                |> Map.exists (fun sym _ -> SizeSymbol.isFlexible sym))

    /// substitute the symbols into the SizeSpec and simplifies it
    let rec substSymbols symVals ss =
        match ss with
        | Base (Sym sym) ->
            match Map.tryFind sym symVals with
            | Some sv -> substSymbols symVals sv
            | None -> ss
        | Base (Fixed _) -> ss
        | Broadcast -> ss
        | Multinom m -> 
            // rebuild multinom with substituted values
            (zero, m.Products)
            ||> Map.fold 
                (fun substSum prod fac ->               
                    let substProd = 
                        (one, prod.Symbols)
                        ||> Map.fold 
                            (fun substProd sBaseSym sPow ->
                                let sBaseSubst = substSymbols symVals (Base (Sym sBaseSym))
                                substProd * (sBaseSubst ** sPow))
                    substSum + fac * substProd)

        |> simplify
            
    /// evaluate symbolic size specification to a number
    let tryEval ss =
        match simplify ss with
        | Base (Fixed f) -> Some f
        | Broadcast -> Some 1
        | _ -> None

    /// true, if evaluation to numeric shape is possible
    let canEval ss =
        match tryEval ss with
        | Some _ -> true
        | None -> false

    /// evaluate symbolic size specification to a number
    let eval ss =
        match tryEval ss with
        | Some s -> s
        | None -> failwithf "cannot evaluate %A to a numeric size since it contains symbols" ss


[<AutoOpen>]
module ShapeSpecTypes =

    /// shape specifcation of a tensor
    type ShapeSpecT = SizeSpecT list

    /// evaluated shape specification of a tensor
    type NShapeSpecT = int list


/// shape specification of a tensor
module ShapeSpec =
    open SizeSymbolTypes

    let withoutAxis ax (sa: ShapeSpecT) =
        sa |> List.without ax

    let insertBroadcastAxis ax (sa: ShapeSpecT) =
        sa |> List.insert ax Broadcast

    let set ax size (sa: ShapeSpecT) =
        sa |> List.set ax size

    let nDim (sa: ShapeSpecT) =
        List.length sa

    let nElem (sa: ShapeSpecT) =
        if List.isEmpty sa then SizeSpec.one
        else List.reduce (*) sa

    let flatten (sa: ShapeSpecT) =
        [nElem sa]

    let concat (sa: ShapeSpecT) (sb: ShapeSpecT) =
        sa @ sb

    let transpose (sa: ShapeSpecT) =
        if nDim sa <> 2 then failwithf "need matrix to transpose but have shape %A" sa
        List.rev sa

    let swap (ax1: int) (ax2: int) (sa: ShapeSpecT) =
        sa  |> List.set ax1 sa.[ax2]
            |> List.set ax2 sa.[ax1]

    let scalar = []

    let vector (ss: SizeSpecT) = [ss]

    let matrix (sr: SizeSpecT) (sc: SizeSpecT) = [sr; sc]

    let emptyVector = [Base (Fixed 0)]

    let padLeft (sa: ShapeSpecT) =
        (Broadcast)::sa

    let padRight (sa: ShapeSpecT) =
        sa @ [Broadcast]

    let rec padToSame sa sb =
        if nDim sa < nDim sb then padToSame (padRight sa) sb
        elif nDim sb < nDim sa then padToSame sa (padRight sb)
        else sa, sb

    let broadcast (sa: ShapeSpecT) dim size =
        match sa.[dim] with
            | Broadcast -> List.set dim size sa
            | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeBroadcast)" dim sa

    let broadcastToSame mustEqual saIn sbIn =
        let mutable sa, sb = saIn, sbIn
        if nDim sa <> nDim sb then 
            failwithf "cannot broadcast shapes %A and %A of different rank to same size" sa sb
        for d = 0 to (nDim sa) - 1 do
            match sa.[d], sb.[d] with
                | al, bl when al = Broadcast -> sa <- broadcast sa d bl
                | al, bl when bl = Broadcast -> sb <- broadcast sb d al
                | al, bl when (if mustEqual then al = bl else true) -> ()        
                | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb

    let enableBroadcast dim (sa: ShapeSpecT) =
        match sa.[dim] with
        | Base (Fixed 1) | Broadcast -> List.set dim Broadcast sa
        | _ -> failwithf "cannot enable broadcasting for dimension %d of shape %A" dim sa

    let disableBroadcast dim (sa: ShapeSpecT) =
        match sa.[dim] with
        | Base (Fixed 1) | Broadcast -> List.set dim (Base (Fixed 1)) sa
        | _ -> failwithf "cannot disable broadcasting for dimension %d of shape %A" dim sa

    let disableAllBroadcasts sa =
        List.map (fun ss -> if ss = Broadcast then Base (Fixed 1) else ss) sa

    let equalWithoutBroadcastability (sa: ShapeSpecT) (sb: ShapeSpecT) =
        List.forall2 (.=) sa sb

    /// evaluates shape to numeric shape, if possible
    let tryEval (sa: ShapeSpecT) : NShapeSpecT option =
        let c = List.map (SizeSpec.tryEval) sa
        if List.exists (Option.isNone) c then None
        else Some (List.map Option.get c)          

    /// true if evaluation to numeric shape is possible
    let canEval sa =
        match tryEval sa with
        | Some _ -> true
        | None -> false

    /// evaluates shape to numeric shape
    let eval (sa: ShapeSpecT) : NShapeSpecT =
        List.map (SizeSpec.eval) sa


[<AutoOpen>]
module RangeSpecTypes =

    /// symbolic/dynamic range specification for one dimension
    type RangeSpecT<'Dyn> = 
        // ranges with symbolic size (length)
        | RSSymElem            of SizeSpecT                           
        | RSDynElem            of 'Dyn                                
        | RSSymStartSymEnd     of (SizeSpecT option) * (SizeSpecT option)
        | RSDynStartSymSize    of 'Dyn * SizeSpecT                    
        | RSNewAxis                                                   
        | RSAllFill                                                   
        //| RngSymStartDynEnd     of SizeSpecT * ExprT<int>              // size: dynamic
        //| RngDynStartDynEnd     of ExprT<int> * ExprT<int>             // size: dynamic
        //| RngDynStartSymEnd     of ExprT<int> * SizeSpecT              // size: dynamic
        //| RngDynStartToEnd      of ExprT<int>                          // size: dynamic

    /// all elements
    let RSAll = RSSymStartSymEnd (None, None)

    // symbolic/dynamic subtensor specification
    type RangesSpecT<'Dyn> = RangeSpecT<'Dyn> list

    /// simple range specification
    type SimpleRangeSpecT<'Dyn> =
        | SRSSymStartSymEnd     of SizeSpecT * (SizeSpecT option)
        | SRSDynStartSymSize    of 'Dyn * SizeSpecT                    

    let SRSAll = SRSSymStartSymEnd (SizeSpec.zero, None)
        
    type SimpleRangesSpecT<'Dyn> = SimpleRangeSpecT<'Dyn> list

module SimpleRangeSpec =
    open ArrayNDNS

    /// evaluate a SimpleRangeSpecT to a RangeT
    let eval dynEvaluator rs =
        match rs with
        | SRSSymStartSymEnd (s, fo) -> 
            Rng (Some (SizeSpec.eval s), Option.map SizeSpec.eval fo)
        | SRSDynStartSymSize (s, elems) -> 
            let sv = dynEvaluator s
            Rng (Some sv, Some (sv + SizeSpec.eval elems))

    let canEvalSymbols rs =
        match rs with
        | SRSSymStartSymEnd (s, fo) ->
            SizeSpec.canEval s && Option.forall SizeSpec.canEval fo
        | SRSDynStartSymSize (_, elems) ->
            SizeSpec.canEval elems

    let isDynamic rs =
        match rs with
        | SRSDynStartSymSize _ -> true
        | _ -> false

module SimpleRangesSpec =

    /// evaluate a RangesSpecT to a RangeT list
    let eval dynEvaluator rs =
        rs
        |> List.map (SimpleRangeSpec.eval dynEvaluator)

    let isDynamic rs =
        rs
        |> List.exists SimpleRangeSpec.isDynamic

    let canEvalSymbols rs =
        rs
        |> List.forall SimpleRangeSpec.canEvalSymbols

