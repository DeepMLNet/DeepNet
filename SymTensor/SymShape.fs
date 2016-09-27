namespace SymTensor

open Basics

[<AutoOpen>]
module SizeSymbolTypes =

    /// a symbolic size
    [<StructuredFormatDisplay ("\"{Name}\"")>]
    type SizeSymbolT = {
        Name:       string;
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
    let name sym =
        sym.Name

    let ofName name =
        {Name=name}


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

        member this.ContainedSizeSymbols =
            products
            |> Map.toSeq
            |> Seq.map (fun (sp, _) -> sp.Symbols 
                                       |> Map.toSeq 
                                       |> Seq.map (fun (sym, _) -> sym))
            |> Seq.concat
            |> Set.ofSeq

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

        /// the set of all contained SizeSymbols
        member this.ContainedSizeSymbols =
            match this with
            | Base (Sym s)   -> Set [s]
            | Base (Fixed _) -> Set.empty
            | Broadcast      -> Set.empty
            | Multinom m     -> m.ContainedSizeSymbols
            
        /// true if the specified SizeSymbol occurs in this SizeSpec
        member this.ContainsSymbol sym =
            this.ContainedSizeSymbols.Contains sym

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
        Base (Sym {Name=s})

    /// broadcastable size one
    let broadcastable =
        Broadcast

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

    /// returns the set of all contained SizeSymbols
    let containedSizeSymbols (ss: SizeSpecT) =
        ss.ContainedSizeSymbols

    /// true if the specified SizeSymbol occurs in the SizeSpec
    let containsSymbol sym (ss: SizeSpecT) =
        ss.ContainsSymbol sym 

            


[<AutoOpen>]
module ShapeSpecTypes =

    /// shape specifcation of a tensor
    type ShapeSpecT = SizeSpecT list

    /// evaluated shape specification of a tensor
    type NShapeSpecT = int list


/// shape specification of a tensor
module ShapeSpec =
    open SizeSymbolTypes

    let insertAxis ax ss (sa: ShapeSpecT) : ShapeSpecT =
        sa |> List.insert ax ss

    let withoutAxis ax (sa: ShapeSpecT) : ShapeSpecT =
        sa |> List.without ax

    let insertBroadcastAxis ax (sa: ShapeSpecT) : ShapeSpecT =
        sa |> insertAxis ax Broadcast

    let set ax size (sa: ShapeSpecT) : ShapeSpecT =
        sa |> List.set ax size

    let nDim (sa: ShapeSpecT) =
        List.length sa

    let nElem (sa: ShapeSpecT) =
        if List.isEmpty sa then SizeSpec.one
        else List.reduce (*) sa

    let flatten (sa: ShapeSpecT) : ShapeSpecT =
        [nElem sa]

    let concat (sa: ShapeSpecT) (sb: ShapeSpecT) : ShapeSpecT =
        sa @ sb

    let transpose (sa: ShapeSpecT) : ShapeSpecT =
        if nDim sa <> 2 then failwithf "need matrix to transpose but have shape %A" sa
        List.rev sa

    let swap (ax1: int) (ax2: int) (sa: ShapeSpecT) : ShapeSpecT =
        sa  |> List.set ax1 sa.[ax2]
            |> List.set ax2 sa.[ax1]

    let scalar : ShapeSpecT = []

    let vector (ss: SizeSpecT) : ShapeSpecT = [ss]

    let matrix (sr: SizeSpecT) (sc: SizeSpecT) : ShapeSpecT = [sr; sc]

    let emptyVector : ShapeSpecT = [Base (Fixed 0)]

    let padLeft (sa: ShapeSpecT) : ShapeSpecT =
        (Broadcast)::sa

    let padRight (sa: ShapeSpecT) : ShapeSpecT =
        sa @ [Broadcast]

    /// pads shapes from the left until they have same rank
    let rec padToSame sa sb =
        if nDim sa < nDim sb then padToSame (padLeft sa) sb
        elif nDim sb < nDim sa then padToSame sa (padLeft sb)
        else sa, sb

    /// pads shapes from the left until they have same rank
    let rec padToSameMany sas =
        let nDimsNeeded = sas |> List.map nDim |> List.max
        sas 
        |> List.map (fun sa ->
            let mutable sa = sa
            while nDim sa < nDimsNeeded do
                sa <- padLeft sa
            sa)

    let broadcast (sa: ShapeSpecT) dim size : ShapeSpecT =
        match sa.[dim] with
        | Broadcast -> List.set dim size sa
        | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeBroadcast)" dim sa

    let broadcastToSameInDims dims mustEqual saIn sbIn =
        let mutable sa, sb = saIn, sbIn
        for d in dims do
            if not (d < nDim sa && d < nDim sb) then
                failwithf "cannot broadcast shapes %A and %A to same size in non-existant dimension %d" sa sb d
            match sa.[d], sb.[d] with
            | al, bl when al = Broadcast -> sa <- broadcast sa d bl
            | al, bl when bl = Broadcast -> sb <- broadcast sb d al
            | al, bl when (if mustEqual then al = bl else true) -> ()        
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb

    let broadcastToSameInDimsMany dims mustEqual sas =
        let mutable sas = sas
        for d in dims do
            if not (sas |> List.forall (fun sa -> d < nDim sa)) then
                failwithf "cannot broadcast shapes %A to same size in non-existant dimension %d" sas d
            let ls = sas |> List.map (fun sa -> sa.[d])
            if ls |> List.exists ((=) Broadcast) then
                let nonBc = ls |> List.filter (fun l -> l <> Broadcast)
                match Set nonBc |> Set.count with
                | 0 -> ()
                | 1 ->
                    let target = List.head nonBc
                    sas <- sas |> List.map (fun sa -> sa |> set d target)
                | _ ->
                    failwithf "cannot broadcast shapes %A to same size in dimension %d because \
                               they don't agree in the target size" sas d                
            elif mustEqual then
                if Set ls |> Set.count > 1 then
                    failwithf "non-broadcast dimension %d of shapes %A does not agree" d sas
        sas

    let broadcastToSame mustEqual sa sb =
        if nDim sa <> nDim sb then 
            failwithf "cannot broadcast shapes %A and %A of different rank to same size" sa sb
        broadcastToSameInDims [0 .. (nDim sa - 1)] mustEqual sa sb

    let broadcastToSameMany mustEqual sas =
        match sas with
        | [] -> []
        | sa::rSas ->
            if rSas |> List.exists (fun rsa -> nDim sa <> nDim rsa) then
                failwithf "cannot broadcast shapes %A of different rank to same size" sas                
            broadcastToSameInDimsMany [0 .. (nDim sa - 1)] mustEqual sas

    let enableBroadcast dim (sa: ShapeSpecT) : ShapeSpecT =
        match sa.[dim] with
        | Base (Fixed 1) | Broadcast -> List.set dim Broadcast sa
        | _ -> failwithf "cannot enable broadcasting for dimension %d of shape %A" dim sa

    let disableBroadcast dim (sa: ShapeSpecT) : ShapeSpecT =
        match sa.[dim] with
        | Base (Fixed 1) | Broadcast -> List.set dim (Base (Fixed 1)) sa
        | _ -> failwithf "cannot disable broadcasting for dimension %d of shape %A" dim sa

    let disableAllBroadcasts sa : ShapeSpecT =
        List.map (fun ss -> if ss = Broadcast then Base (Fixed 1) else ss) sa

    let equalWithBroadcastability (sa: ShapeSpecT) (sb: ShapeSpecT) =
        List.length sa = List.length sb &&
            List.forall2 SizeSpec.equalWithBroadcastability sa sb

    let equalWithoutBroadcastability (sa: ShapeSpecT) (sb: ShapeSpecT) =
         List.length sa = List.length sb &&
            List.forall2 SizeSpec.equalWithBroadcastability sa sb

    /// Permutes the axes as specified.
    let permuteAxes (permut: int list) (sa: ShapeSpecT) : ShapeSpecT =
        if nDim sa <> List.length permut then
            failwithf "permutation %A must have same rank as shape %A" permut sa
        sa |> List.permute (fun i -> permut.[i])

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

    /// substitute the symbols into the ShapeSpec and simplifies it
    let substSymbols symVals (sa: ShapeSpecT) : ShapeSpecT =
        List.map (SizeSpec.substSymbols symVals) sa


    type SolutionT = {
        LeftValues:     Map<SizeSymbolT, SizeSpecT>
        RightValues:    Map<SizeSymbolT, SizeSpecT>
    }        

    let solve (left: ShapeSpecT) (right: SizeSymbolT list) =
        if left.Length <> right.Length then failwith "dimension mismatch"
        if right |> Set.ofList |> Set.count <> right.Length then
            failwith "symbols on the right must be unique"
        
        let leftValues = Dictionary<SizeSymbolT, SizeSpecT>()
        let rightValues = Dictionary<SizeSymbolT, SizeSpecT>()

        for l, r in List.zip left right do
            match l with
            | Base (Fixed _)
            | Broadcast -> 
                rightValues.Add (r, l)
            | Base (Sym s) ->
                if leftValues.ContainsKey s then
                    let pv = leftValues.[s]
                    rightValues.Add (r, pv)
                else
                    leftValues.Add (s, Base (Sym r))
            | Multinom _ -> failwith "cannot solve with multinoms"
                
        {
            LeftValues = leftValues |> Map.ofDictionary
            RightValues = rightValues |> Map.ofDictionary
        }
                


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

