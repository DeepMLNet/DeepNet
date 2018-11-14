namespace SymTensor

open DeepNet.Utils


/// A symbolic size.
[<Struct; StructuredFormatDisplay ("\"{Name}\"")>]
type SizeSymbol = SizeSymbol of string with

    /// identifier
    static member name (SizeSymbol name) = name

    /// creates a symbolic size with the specified identifier
    static member ofName name = SizeSymbol name


/// Elementary size specification
/// Can be either a symbol or a fixed quantity.
[<RequireQualifiedAccess; StructuredFormatDisplay ("{Pretty}")>]
type BaseSize =
    /// symbolic size
    | Sym of SizeSymbol
    /// numeric size
    | Fixed of Frac

    /// pretty string
    member this.Pretty =
        match this with
        | BaseSize.Sym s -> sprintf "%A" s
        | BaseSize.Fixed f -> sprintf "%A" f


/// Internal types to represent symbolic sizes.
module SymShapeInternals =     

    /// Product of elementary size specifications.
    [<StructuredFormatDisplay("{Pretty}")>]
    type SizeProduct(symbols: Map<SizeSymbol, int64>) =
        let symbols = symbols |> Map.filter (fun _ sPower -> sPower <> 0L)
  
        new() = SizeProduct(Map.empty)
        new(b: SizeSymbol) = SizeProduct(b, 1L)
        new(b: SizeSymbol, pow: int64) = SizeProduct(Map.empty |> Map.add b pow)

        member this.Symbols = symbols

        static member (*) (a: SizeProduct, b: SizeProduct) =
            let pSymbols = 
                Map.fold 
                    (fun p bBase bPower -> 
                        match Map.tryFind bBase p with
                        | Some pPower -> Map.add bBase (pPower + bPower) p
                        | None -> Map.add bBase bPower p) 
                    a.Symbols b.Symbols
            SizeProduct(pSymbols)
         
        static member (*) (a: SizeProduct, b: SizeSymbol) = a * SizeProduct(b)
        static member (*) (a: SizeSymbol, b: SizeProduct) = SizeProduct(a) * b

        member this.Pretty = 
            this.Symbols
            |> Map.toSeq
            |> Seq.map 
                (fun (tBase, tPower) ->
                    if tPower = 1L then sprintf "%A" tBase 
                    else sprintf "%A**%d" tBase tPower)
            |> Seq.toList
            |> String.concat " * "
                        
        override this.Equals(otherObj) =
            match otherObj with
            | :? SizeProduct as other -> this.Symbols = other.Symbols 
            | _ -> false

        override this.GetHashCode() =
            hash this.Symbols

        interface System.IComparable with
            member this.CompareTo otherObj =
                match otherObj with
                | :? SizeProduct as other -> compare this.Symbols other.Symbols
                | _ -> invalidArg "otherObj" "cannot compare values of different types"

        /// true if product is empty (equal to 1)
        static member isEmpty (p: SizeProduct) =
            Map.isEmpty p.Symbols

        /// empty product (equal to 1)
        static member empty =
            SizeProduct()

    /// Active patterns for SizeProduct.
    module SizeProduct =

        /// matches if product consists of a single symbol with power 1
        let (|SingleSymbol|_|) (sp: SizeProduct) =
            let mc = Map.toList sp.Symbols
            match List.length mc with
            | 1 -> 
                let bs, power = mc.[0]
                if power = 1L then Some bs else None
            | _ -> None

        /// matches if product is empty (equal to 1)
        let (|Empty|_|) (sp: SizeProduct) =
            if SizeProduct.isEmpty sp then Some () else None


    /// Symbolic multinomial.
    [<StructuredFormatDisplay("{Pretty}")>]
    type SizeMultinom (products: Map<SizeProduct, Frac>) =
        let products = products |> Map.filter (fun _ fac -> fac .<> Frac.zero)

        new (bs: BaseSize) = SizeMultinom (Frac.one, bs, 1L)
        new (bs: BaseSize, pow: int64) = SizeMultinom (Frac.one, bs, pow)
        new (fac: Frac, bs: BaseSize, pow: int64) =
            let m =
                match bs with
                | BaseSize.Sym s -> Map [SizeProduct (s, pow), fac]
                | BaseSize.Fixed f -> Map [SizeProduct (), fac * (pown f (int32 pow))]
            SizeMultinom (m)

        member this.Products = products

        static member (~-) (a: SizeMultinom) =
            a.Products 
            |> Map.map (fun _ fac -> -fac)
            |> SizeMultinom 

        static member (+) (a: SizeMultinom, b: SizeMultinom) =
            (a.Products, b.Products)
            ||> Map.fold (fun res prod fac -> 
                match Map.tryFind prod res with
                | Some rFac -> res |> Map.add prod (fac + rFac)
                | None      -> res |> Map.add prod fac)
            |> SizeMultinom
                
        static member (-) (a: SizeMultinom, b: SizeMultinom) = a + (-b)
        
        static member (*) (a: SizeMultinom, b: SizeMultinom) =
            seq { for KeyValue(ap, af) in a.Products do
                    for KeyValue(bp, bf) in b.Products do
                        yield ap*bp, af*bf }
            |> Map.ofSeq
            |> SizeMultinom

        member this.ContainedSizeSymbols =
            products
            |> Map.toSeq
            |> Seq.map (fun (sp, _) -> sp.Symbols 
                                       |> Map.toSeq 
                                       |> Seq.map (fun (sym, _) -> sym))
            |> Seq.concat
            |> Set.ofSeq

        member this.Pretty =
            if Map.isEmpty products then "0"
            else
                products
                |> Map.toSeq
                |> Seq.map (fun (p, f) -> 
                    if SizeProduct.isEmpty p then sprintf "%A" f
                    elif f = Frac.one then sprintf "%A" p
                    else sprintf "%A * %A" f p)
                |> String.concat " + "
        
        override this.Equals(otherObj) =
            match otherObj with
            | :? SizeMultinom as other -> this.Products = other.Products 
            | _ -> false

        override this.GetHashCode() =
            hash this.Products

        interface System.IComparable with
            member this.CompareTo otherObj =
                match otherObj with
                | :? SizeMultinom as other -> compare this.Products other.Products
                | _ -> invalidArg "otherObj" "cannot compare values of different types"


open SymShapeInternals

/// symbolic size specification of a dimension (axis)
[<RequireQualifiedAccess; StructuralEquality; StructuralComparison; StructuredFormatDisplay ("{Pretty}")>]
type SizeSpec =
    | Base of BaseSize               // fixed size or symbol
    | Broadcast                      // size 1 and broadcastable
    | Multinom of SizeMultinom       // product of fixed sizes and symbols

    /// simplify size specification
    static member Simplify (ss: SizeSpec) =
        match ss with
        | SizeSpec.Multinom m -> 
            match m.Products |> Map.toList with
            | [SizeProduct.SingleSymbol s, Frac.One] -> SizeSpec.Base (BaseSize.Sym s)
            | [SizeProduct.Empty, f] -> SizeSpec.Base (BaseSize.Fixed f)
            | [] -> SizeSpec.Base (BaseSize.Fixed Frac.zero)
            | _ -> ss
        | _ -> ss

    static member get_Zero () = SizeSpec.Base (BaseSize.Fixed Frac.zero)

    static member (~-) (ssa: SizeSpec) =
        match ssa with
        | SizeSpec.Base (BaseSize.Fixed Frac.Zero) -> ssa
        | SizeSpec.Base b -> SizeSpec.Multinom (-SizeMultinom(b))
        | SizeSpec.Broadcast -> SizeSpec.Multinom (-SizeMultinom(BaseSize.Fixed Frac.one))
        | SizeSpec.Multinom m -> SizeSpec.Multinom (-m)
        |> SizeSpec.Simplify

    static member (+) (ssa: SizeSpec, ssb: SizeSpec) =
        match ssa, ssb with
        | SizeSpec.Base (BaseSize.Fixed Frac.Zero), ss | ss, SizeSpec.Base (BaseSize.Fixed Frac.Zero) -> ss
        | SizeSpec.Broadcast, ss | ss, SizeSpec.Broadcast -> ss + (SizeSpec.Base (BaseSize.Fixed Frac.one))
        | SizeSpec.Multinom ma, SizeSpec.Multinom mb -> SizeSpec.Multinom (ma + mb)
        | SizeSpec.Multinom m, SizeSpec.Base b | SizeSpec.Base b, SizeSpec.Multinom m -> SizeSpec.Multinom (m + SizeMultinom(b))
        | SizeSpec.Base ba, SizeSpec.Base bb -> SizeSpec.Multinom (SizeMultinom(ba) + SizeMultinom(bb))
        |> SizeSpec.Simplify

    static member (-) (ssa: SizeSpec, ssb: SizeSpec) =
        ssa + (-ssb)

    static member (*) (ssa: SizeSpec, ssb: SizeSpec) =
        match ssa, ssb with
        | SizeSpec.Base (BaseSize.Fixed Frac.Zero), _ | _, SizeSpec.Base (BaseSize.Fixed Frac.Zero) -> SizeSpec.Base (BaseSize.Fixed Frac.zero)
        | SizeSpec.Broadcast, ss | ss, SizeSpec.Broadcast -> ss
        | SizeSpec.Multinom ma, SizeSpec.Multinom mb -> SizeSpec.Multinom (ma * mb)
        | SizeSpec.Multinom m, SizeSpec.Base b | SizeSpec.Base b, SizeSpec.Multinom m -> SizeSpec.Multinom (m * SizeMultinom(b))
        | SizeSpec.Base ba, SizeSpec.Base bb -> SizeSpec.Multinom (SizeMultinom(ba) * SizeMultinom(bb))
        |> SizeSpec.Simplify

    static member Pow (ssa: SizeSpec, pow: int64) =
        match pow with
        | 0L -> SizeSpec.Base (BaseSize.Fixed Frac.one)
        | 1L -> ssa
        | _ ->
            match ssa with
            | SizeSpec.Base (BaseSize.Fixed f) -> SizeSpec.Base (BaseSize.Fixed (pown f (int32 pow)))
            | SizeSpec.Base (BaseSize.Sym s) -> SizeSpec.Multinom (SizeMultinom (BaseSize.Sym s, pow))
            | SizeSpec.Broadcast -> SizeSpec.Broadcast
            | SizeSpec.Multinom m ->
                m
                |> Seq.replicate (int32 pow)
                |> Seq.reduce (*)
                |> SizeSpec.Multinom
        |> SizeSpec.Simplify

    // operations with FracT
    static member (+) (ssa: SizeSpec, ssb: Frac) = ssa + (SizeSpec.Base (BaseSize.Fixed ssb))
    static member (+) (ssa: Frac, ssb: SizeSpec) = (SizeSpec.Base (BaseSize.Fixed ssa)) + ssb
    static member (-) (ssa: SizeSpec, ssb: Frac) = ssa - (SizeSpec.Base (BaseSize.Fixed ssb))
    static member (-) (ssa: Frac, ssb: SizeSpec) = (SizeSpec.Base (BaseSize.Fixed ssa)) - ssb
    static member (*) (ssa: SizeSpec, ssb: Frac) = ssa * (SizeSpec.Base (BaseSize.Fixed ssb))
    static member (*) (ssa: Frac, ssb: SizeSpec) = (SizeSpec.Base (BaseSize.Fixed ssa)) * ssb

    // operations with int
    static member (+) (ssa: SizeSpec, ssb: int64) = ssa + Frac ssb
    static member (+) (ssa: int64, ssb: SizeSpec) = Frac ssa + ssb
    static member (-) (ssa: SizeSpec, ssb: int64) = ssa - Frac ssb
    static member (-) (ssa: int64, ssb: SizeSpec) = Frac ssa - ssb
    static member (*) (ssa: SizeSpec, ssb: int64) = ssa * Frac ssb
    static member (*) (ssa: int64, ssb: SizeSpec) = Frac ssa * ssb

    /// equal size with broadcastability
    static member (%=) (ssa: SizeSpec, ssb: SizeSpec) = 
        SizeSpec.Simplify ssa = SizeSpec.Simplify ssb 

    /// equal size ignoring broadcastability
    static member (.=) (ssa: SizeSpec, ssb: SizeSpec) = 
        match SizeSpec.Simplify ssa, SizeSpec.Simplify ssb with
        | SizeSpec.Broadcast, SizeSpec.Base (BaseSize.Fixed Frac.One) | SizeSpec.Base (BaseSize.Fixed Frac.One), SizeSpec.Broadcast -> true
        | a, b -> a = b

    /// unequal size ignoring broadcastability
    static member (.<>) (ssa: SizeSpec, ssb: SizeSpec) = not (ssa .= ssb)

    /// the set of all contained SizeSymbols
    member this.ContainedSizeSymbols =
        match this with
        | SizeSpec.Base (BaseSize.Sym s)   -> Set [s]
        | SizeSpec.Base (BaseSize.Fixed _) -> Set.empty
        | SizeSpec.Broadcast      -> Set.empty
        | SizeSpec.Multinom m     -> m.ContainedSizeSymbols
            
    /// true if the specified SizeSymbol occurs in this SizeSpec
    member this.ContainsSymbol sym =
        this.ContainedSizeSymbols.Contains sym

    member this.Pretty =
        match this with
        | SizeSpec.Base b -> sprintf "%A" b
        | SizeSpec.Broadcast -> "1*"
        | SizeSpec.Multinom m -> sprintf "%A" m

    /// simplify size specification
    static member simplify (ss: SizeSpec) = SizeSpec.Simplify ss

    /// True if both sizes have the same number of elements and 
    /// are both broadcastable or non-broadcastable.
    static member equalWithBroadcastability (ssa: SizeSpec) (ssb: SizeSpec) = ssa %= ssb        

    /// True if both sizes have the same number of elements.
    /// Broadcastable and non-broadcastable are treated as equal.
    static member equalWithoutBroadcastability (ssa: SizeSpec) (ssb: SizeSpec) = ssa .= ssb

    /// size zero
    static member zero = SizeSpec.Base (BaseSize.Fixed Frac.zero)

    /// not-broadcastable size one
    static member one = SizeSpec.Base (BaseSize.Fixed Frac.one)

    /// fixed integer size
    static member fix s = SizeSpec.Base (BaseSize.Fixed (Frac s))

    /// fixed fractional size
    static member fixFrac nom dnm = SizeSpec.Base (BaseSize.Fixed (Frac (nom, dnm)))

    /// symbolic size
    static member symbol s = SizeSpec.Base (BaseSize.Sym (SizeSymbol s))

    /// broadcastable size one
    static member broadcastable = SizeSpec.Broadcast

    /// extracts the size symbol
    static member extractSymbol s =
        match s with
        | SizeSpec.Base (BaseSize.Sym sym) -> sym
        | _ -> failwith "specified SizeSpec is not a symbol"

    /// substitute the symbols into the SizeSpec and simplifies it
    static member substSymbols symVals ss =
        match ss with
        | SizeSpec.Base (BaseSize.Sym sym) ->
            match Map.tryFind sym symVals with
            | Some sv -> SizeSpec.substSymbols symVals sv
            | None -> ss
        | SizeSpec.Base (BaseSize.Fixed _) -> ss
        | SizeSpec.Broadcast -> ss
        | SizeSpec.Multinom m -> 
            // rebuild multinom with substituted values
            (zero, m.Products)
            ||> Map.fold 
                (fun substSum prod fac ->               
                    let substProd = 
                        (one, prod.Symbols)
                        ||> Map.fold 
                            (fun substProd sBaseSym sPow ->
                                let sBaseSubst = SizeSpec.substSymbols symVals (SizeSpec.Base (BaseSize.Sym sBaseSym))
                                substProd * (sBaseSubst ** sPow))
                    substSum + fac * substProd)
        |> SizeSpec.simplify
            
    /// evaluate symbolic size specification to a number
    static member tryEval ss =
        match SizeSpec.simplify ss with
        | SizeSpec.Base (BaseSize.Fixed (Frac.Integral i)) -> Some i
        | SizeSpec.Broadcast -> Some 1L
        | _ -> None

    /// true, if evaluation to numeric shape is possible
    static member canEval ss =
        match SizeSpec.tryEval ss with
        | Some _ -> true
        | None -> false

    /// evaluate symbolic size specification to a number
    static member eval ss =
        match SizeSpec.tryEval ss with
        | Some s -> s
        | None -> failwithf "cannot evaluate %A to a numeric size since it contains symbols" ss

    /// returns the set of all contained SizeSymbols
    static member containedSizeSymbols (ss: SizeSpec) = ss.ContainedSizeSymbols

    /// true if the specified SizeSymbol occurs in the SizeSpec
    static member containsSymbol sym (ss: SizeSpec) = ss.ContainsSymbol sym 
         

/// shape specifcation of a tensor
type ShapeSpec = SizeSpec list

/// evaluated shape specification of a tensor
type NShapeSpec = int64 list

/// shape specification of a tensor
module ShapeSpec =

    let insertAxis ax ss (sa: ShapeSpec) : ShapeSpec =
        sa |> List.insert ax ss

    let withoutAxis ax (sa: ShapeSpec) : ShapeSpec =
        sa |> List.without ax

    let insertBroadcastAxis ax (sa: ShapeSpec) : ShapeSpec =
        sa |> insertAxis ax SizeSpec.Broadcast

    let set ax size (sa: ShapeSpec) : ShapeSpec =
        sa |> List.set ax size

    let nDim (sa: ShapeSpec) =
        List.length sa

    let nElem (sa: ShapeSpec) =
        if List.isEmpty sa then SizeSpec.one
        else List.reduce (*) sa

    let flatten (sa: ShapeSpec) : ShapeSpec =
        [nElem sa]

    let concat (sa: ShapeSpec) (sb: ShapeSpec) : ShapeSpec =
        sa @ sb

    let transpose (sa: ShapeSpec) : ShapeSpec =
        if nDim sa <> 2 then failwithf "need matrix to transpose but have shape %A" sa
        List.rev sa

    let swap (ax1: int) (ax2: int) (sa: ShapeSpec) : ShapeSpec =
        sa  |> List.set ax1 sa.[ax2]
            |> List.set ax2 sa.[ax1]

    let scalar : ShapeSpec = []

    let vector (ss: SizeSpec) : ShapeSpec = [ss]

    let matrix (sr: SizeSpec) (sc: SizeSpec) : ShapeSpec = [sr; sc]

    let emptyVector : ShapeSpec = [SizeSpec.zero]

    /// pads shape by inserting broadcast dimension on the left
    let padLeft (sa: ShapeSpec) : ShapeSpec =
        (SizeSpec.Broadcast)::sa

    /// pads shape by inserting broadcast dimension on the right
    let padRight (sa: ShapeSpec) : ShapeSpec =
        sa @ [SizeSpec.Broadcast]

    /// pads shape from the left to specified number of dimensions
    let padTo dims saIn =
        let mutable sa = saIn
        while nDim sa < dims do sa <- padLeft sa
        if nDim sa <> dims then
            failwithf "cannot pad higher-rank shape %A to %d dimensions" saIn dims
        sa

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

    let broadcast (sa: ShapeSpec) dim size : ShapeSpec =
        match sa.[dim] with
        | SizeSpec.Broadcast -> List.set dim size sa
        | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeBroadcast)" dim sa

    let broadcastToShape (trgtShp: ShapeSpec) (saIn: ShapeSpec) : ShapeSpec =
        let mutable sa = saIn
        if nDim sa <> nDim trgtShp then
            failwithf "cannot broadcast shape %A to shape %A" saIn trgtShp
        for d=0 to nDim trgtShp - 1 do
            match sa.[d], trgtShp.[d] with
            | al, bl when al = SizeSpec.Broadcast -> sa <- broadcast sa d bl
            | al, bl when al = bl -> ()
            | _ -> failwithf "cannot broadcast shape %A to %A in dimension %d" sa trgtShp d
        sa

    let broadcastToSameInDims dims mustEqual saIn sbIn =
        let mutable sa, sb = saIn, sbIn
        for d in dims do
            if not (d < nDim sa && d < nDim sb) then
                failwithf "cannot broadcast shapes %A and %A to same size in non-existant dimension %d" sa sb d
            match sa.[d], sb.[d] with
            | al, bl when al = SizeSpec.Broadcast -> sa <- broadcast sa d bl
            | al, bl when bl = SizeSpec.Broadcast -> sb <- broadcast sb d al
            | al, bl when (if mustEqual then al = bl else true) -> ()        
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb

    let broadcastToSameInDimsMany dims mustEqual sas =
        let mutable sas = sas
        for d in dims do
            if not (sas |> List.forall (fun sa -> d < nDim sa)) then
                failwithf "cannot broadcast shapes %A to same size in non-existant dimension %d" sas d
            let ls = sas |> List.map (fun sa -> sa.[d])
            if ls |> List.exists ((=) SizeSpec.Broadcast) then
                let nonBc = ls |> List.filter (fun l -> l <> SizeSpec.Broadcast)
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

    let enableBroadcast dim (sa: ShapeSpec) : ShapeSpec =
        match sa.[dim] with
        | SizeSpec.Base (BaseSize.Fixed Frac.One) | SizeSpec.Broadcast -> List.set dim SizeSpec.Broadcast sa
        | _ -> failwithf "cannot enable broadcasting for dimension %d of shape %A" dim sa

    let disableBroadcast dim (sa: ShapeSpec) : ShapeSpec =
        match sa.[dim] with
        | SizeSpec.Base (BaseSize.Fixed Frac.One) | SizeSpec.Broadcast -> List.set dim (SizeSpec.Base (BaseSize.Fixed Frac.one)) sa
        | _ -> failwithf "cannot disable broadcasting for dimension %d of shape %A" dim sa

    let disableAllBroadcasts sa : ShapeSpec =
        List.map (fun ss -> if ss = SizeSpec.Broadcast then SizeSpec.Base (BaseSize.Fixed Frac.one) else ss) sa
        
    /// True if both shape have the same number of elements and 
    /// are both broadcastable or non-broadcastable in each dimension.
    let equalWithBroadcastability (sa: ShapeSpec) (sb: ShapeSpec) =
        List.length sa = List.length sb &&
            List.forall2 SizeSpec.equalWithBroadcastability sa sb

    /// True if both shapes have the same number of elements in each dimension.
    /// Broadcastable and non-broadcastable are treated as equal.            
    let equalWithoutBroadcastability (sa: ShapeSpec) (sb: ShapeSpec) =
         List.length sa = List.length sb &&
            List.forall2 SizeSpec.equalWithoutBroadcastability sa sb

    /// Permutes the axes as specified.
    let permuteAxes (permut: int list) (sa: ShapeSpec) : ShapeSpec =
        if nDim sa <> List.length permut then
            failwithf "permutation %A must have same rank as shape %A" permut sa
        sa |> List.permute (fun i -> permut.[i])

    /// evaluates shape to numeric shape, if possible
    let tryEval (sa: ShapeSpec) : NShapeSpec option =
        let c = List.map (SizeSpec.tryEval) sa
        if List.exists (Option.isNone) c then None
        else Some (List.map Option.get c)          

    /// true if evaluation to numeric shape is possible
    let canEval sa =
        match tryEval sa with
        | Some _ -> true
        | None -> false

    /// evaluates shape to numeric shape
    let eval (sa: ShapeSpec) : NShapeSpec =
        List.map (SizeSpec.eval) sa

    /// substitute the symbols into the ShapeSpec and simplifies it
    let substSymbols symVals (sa: ShapeSpec) : ShapeSpec =
        List.map (SizeSpec.substSymbols symVals) sa

    type SolutionT = {
        LeftValues:     Map<SizeSymbol, SizeSpec>
        RightValues:    Map<SizeSymbol, SizeSpec>
    }        

    let solve (left: ShapeSpec) (right: SizeSymbol list) =
        if left.Length <> right.Length then failwith "dimension mismatch"
        if right |> Set.ofList |> Set.count <> right.Length then
            failwith "symbols on the right must be unique"
        
        let leftValues = Dictionary<SizeSymbol, SizeSpec>()
        let rightValues = Dictionary<SizeSymbol, SizeSpec>()

        for l, r in List.zip left right do
            match l with
            | SizeSpec.Base (BaseSize.Fixed _)
            | SizeSpec.Broadcast -> 
                rightValues.Add (r, l)
            | SizeSpec.Base (BaseSize.Sym s) ->
                if leftValues.ContainsKey s then
                    let pv = leftValues.[s]
                    rightValues.Add (r, pv)
                else
                    leftValues.Add (s, SizeSpec.Base (BaseSize.Sym r))
            | SizeSpec.Multinom _ -> failwith "cannot solve with multinoms"
                
        {
            LeftValues = leftValues |> Map.ofDictionary
            RightValues = rightValues |> Map.ofDictionary
        }
                

/// basic range specification for one dimension
type BaseRangeSpec = SizeSpec * SizeSpec

/// basic range specification for multiple dimensions
type BaseRangesSpec = BaseRangeSpec list

/// Functions for working with BaseRangesSpec.
module BaseRangesSpec =

    /// Try to evalualte a BaseRangesSpecT to a numeric range.
    let tryEval (rng: BaseRangesSpec) =
        let rec doEval rng =
            match rng with
            | (first, last) :: rrng ->
                match SizeSpec.tryEval first, SizeSpec.tryEval last, doEval rrng with
                | Some first, Some last, Some rrng -> Some ((first, last) :: rrng)
                | _ -> None
            | [] -> Some []
        doEval rng

    /// True if a BaseRangesSpecT can be evaluated to a numeric range.
    let canEval (rng: BaseRangesSpec) =
        match tryEval rng with
        | Some _ -> true
        | None -> false

    /// Evaluates a BaseRangesSpecT to a numeric range.
    let eval (rng: BaseRangesSpec) =
        match tryEval rng with
        | Some rng -> rng
        | None -> failwithf "cannot evaluate BaseRangesSpecT %A to numeric range" rng

    /// checks that the BaseRangesSpec is valid
    let check (rng: BaseRangesSpec) =
        match tryEval rng with
        | Some rng ->
            for first, last in rng do
                if last < first then 
                    failwithf "invalid BaseRangesSpec: %A" rng
        | None -> ()

    /// True if two BaseRangesSpec overlap.
    /// Both BaseRangesSpec must be evaluateble to numeric ranges.
    let overlapping (a: BaseRangesSpec) (b: BaseRangesSpec) =
        check a; check b
        (eval a, eval b)
        ||> List.forall2 (fun (aFirst, aLast) (bFirst, bLast) ->
            aFirst <= bFirst && bFirst <= aLast ||
            aFirst <= bLast  && bLast  <= aLast ||
            bFirst <= aFirst && aLast  <= bLast)

    /// True if any two ranges are overlapping.
    /// This has complexity O(N^2) currently.
    /// All BaseRangesSpec must be evaluateble to numeric ranges.
    let areOverlapping (rngs: BaseRangesSpec list) =       
        let rec testOvlp nonOvlp cands =
            match cands with
            | cand::rCands ->
                if nonOvlp |> List.exists (overlapping cand) then true
                else testOvlp (cand::nonOvlp) rCands
            | [] -> false
        testOvlp [] rngs

    /// True if the BaseRangesSpecTs cover a tensor of the specified shape completely without overlap.
    /// All BaseRangesSpecT and the ShapeSpecT must be evaluatable to numeric ranges and a
    /// numeric shape respectively.
    let areCoveringWithoutOverlap (shp: ShapeSpec) (rngs: BaseRangesSpec list) =       
        if areOverlapping rngs then false
        else
            let shpElems = 
                shp 
                |> ShapeSpec.eval 
                |> List.fold (*) 1L
            let rngElems = 
                rngs
                |> List.map eval
                |> List.map (List.fold (fun p (first, last) -> p * (last - first + 1L)) 1L)
                |> List.sum
            shpElems = rngElems


/// symbolic/dynamic range specification for one dimension
[<RequireQualifiedAccess>]
type RangeSpec<'Dyn> = 
    // ranges with symbolic size (length)
    | SymElem            of SizeSpec                           
    | DynElem            of 'Dyn                                
    | SymStartSymEnd     of (SizeSpec option) * (SizeSpec option)
    | DynStartSymSize    of 'Dyn * SizeSpec                    
    | NewAxis                                                   
    | AllFill                                                   
    //| RngSymStartDynEnd     of SizeSpecT * ExprT<int>              // size: dynamic
    //| RngDynStartDynEnd     of ExprT<int> * ExprT<int>             // size: dynamic
    //| RngDynStartSymEnd     of ExprT<int> * SizeSpecT              // size: dynamic
    //| RngDynStartToEnd      of ExprT<int>                          // size: dynamic

    static member All = RangeSpec<'Dyn>.SymStartSymEnd (None, None)

// symbolic/dynamic subtensor specification
type RangesSpec<'Dyn> = RangeSpec<'Dyn> list

/// Simple range specification for one dimension.
[<RequireQualifiedAccess; StructuredFormatDisplay("{Pretty}")>]
type SimpleRangeSpec<'Dyn> =
    | SymStartSymEnd     of SizeSpec * (SizeSpec option)
    | DynStartSymSize    of 'Dyn * SizeSpec                    

    member this.Pretty =
        match this with
        | SimpleRangeSpec.SymStartSymEnd (first, Some last) -> sprintf "%A..%A" first last
        | SimpleRangeSpec.SymStartSymEnd (first, None) -> sprintf "%A.." first
        | SimpleRangeSpec.DynStartSymSize (first, size) -> sprintf "D%A..D%A+%A-1" first first size
    
    static member All = SimpleRangeSpec<'Dyn>.SymStartSymEnd (SizeSpec.zero, None)
     
    /// evaluate a SimpleRangeSpec to a Tensor.Rng
    static member eval dynEvaluator (rs: SimpleRangeSpec<'Dyn>) =
        match rs with
        | SimpleRangeSpec.SymStartSymEnd (s, fo) -> 
            Tensor.Rng.Rng (Some (SizeSpec.eval s), Option.map SizeSpec.eval fo)
        | SimpleRangeSpec.DynStartSymSize (s, elems) -> 
            let sv = dynEvaluator s
            Tensor.Rng.Rng (Some sv, Some (sv + SizeSpec.eval elems))

    static member canEvalSymbols (rs: SimpleRangeSpec<'Dyn>) =
        match rs with
        | SimpleRangeSpec.SymStartSymEnd (s, fo) ->
            SizeSpec.canEval s && Option.forall SizeSpec.canEval fo
        | SimpleRangeSpec.DynStartSymSize (_, elems) ->
            SizeSpec.canEval elems

    static member isDynamic (rs: SimpleRangeSpec<'Dyn>) =
        match rs with
        | SimpleRangeSpec.DynStartSymSize _ -> true
        | _ -> false

    static member toBaseRangeSpec (size: SizeSpec) (rs: SimpleRangeSpec<'Dyn>) =
        match rs with
        | SimpleRangeSpec.SymStartSymEnd (first, Some last) -> first, last
        | SimpleRangeSpec.SymStartSymEnd (first, None) -> first, size - 1L
        | _ -> failwithf "cannot convert %A to BaseRangeSpec" rs

/// Active patterns for SimpleRangeSpec.
module SimpleRangeSpec =

   let (|Dynamic|Static|) rs =
        if SimpleRangeSpec.isDynamic rs then Dynamic else Static


/// Simple range specification for multiple dimensions.
type SimpleRangesSpec<'Dyn> = SimpleRangeSpec<'Dyn> list

/// Functions for working with SimpleRangesSpec.
module SimpleRangesSpec =

    /// evaluate a RangesSpecT to a RangeT list
    let eval dynEvaluator rs =
        rs |> List.map (SimpleRangeSpec.eval dynEvaluator)

    let isDynamic rs =
        rs |> List.exists SimpleRangeSpec.isDynamic

    let canEvalSymbols rs =
        rs |> List.forall SimpleRangeSpec.canEvalSymbols

    let (|Dynamic|Static|) rs =
        if isDynamic rs then Dynamic else Static

    let toBaseRangesSpec (shape: ShapeSpec) rs =
        (shape, rs) ||> List.map2 SimpleRangeSpec.toBaseRangeSpec


