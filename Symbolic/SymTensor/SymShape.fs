namespace SymTensor

open Tensor.Utils


module Utils =

    /// greatest common divisor of a and b 
    let rec gcd a b =
        // Euclidean algorithm
        if a < 0L then gcd -a b
        elif b < 0L then gcd a -b
        elif a = 0L then b
        elif b = 0L then a
        elif a < b then gcd b a
        else
            //let q = a / b
            let r = a % b
            if r = 0L then b
            else gcd b r

    /// least common multiple of a and b
    let lcm a b =
        abs (a * b) / gcd a b


[<AutoOpen>]
module SizeSymbolTypes =
    open Utils

    /// a symbolic size
    [<StructuredFormatDisplay ("\"{Name}\"")>]
    type SizeSymbolT = {
        Name:       string;
    }

    [<StructuredFormatDisplay("{Pretty}")>]
    [<Struct>]
    /// a rational number
    type FracT = 
        val Nom: int64
        val Dnm: int64

        new (nom, dnm) = 
            let nom, dnm =
                match dnm with
                | 0L -> failwith "denominator cannot be zero"
                | _ when dnm < 0L -> -nom, -dnm
                | _ -> nom, dnm
            let cd = gcd nom dnm
            {Nom=nom/cd; Dnm=dnm/cd}
        new (value) = FracT (value, 1L)               

        static member (~-) (a: FracT) = FracT (-a.Nom, a.Dnm)
        static member (+) (a: FracT, b: FracT) = FracT (a.Nom * b.Dnm + b.Nom * a.Dnm, a.Dnm * b.Dnm)
        static member (-) (a: FracT, b: FracT) = a + (-b)
        static member (*) (a: FracT, b: FracT) = FracT (a.Nom * b.Nom, a.Dnm * b.Dnm)
        static member (/) (a: FracT, b: FracT) = FracT (a.Nom * b.Dnm, a.Dnm * b.Nom)
        static member (.=) (a: FracT, b: FracT) = a = b
        static member (.<>) (a: FracT, b: FracT) = a <> b
        static member get_Zero () = FracT (0L)
        static member get_One () = FracT (1L)

        static member (+) (a: FracT, b: int64) = a + FracT b
        static member (-) (a: FracT, b: int64) = a - FracT b
        static member (*) (a: FracT, b: int64) = a * FracT b
        static member (/) (a: FracT, b: int64) = a / FracT b
        static member (.=) (a: FracT, b: int64) = a .= FracT b
        static member (.<>) (a: FracT, b: int64) = a .<> FracT b

        static member (+) (a: int64, b: FracT) = FracT a + b
        static member (-) (a: int64, b: FracT) = FracT a - b
        static member (*) (a: int64, b: FracT) = FracT a * b
        static member (/) (a: int64, b: FracT) = FracT a / b
        static member (.=) (a: int64, b: FracT) = FracT a .= b
        static member (.<>) (a: int64, b: FracT) = FracT a .<> b
         
        member this.IntValue = 
            if this.Dnm = 1L then this.Nom
            else failwithf "%A is not an integer" this

        member this.Pretty =
            if this.Dnm = 1L then sprintf "%d" this.Nom
            else sprintf "(%d/%d)" this.Nom this.Dnm

    /// elementary size specification, can be either a symbol or a fixed quantity
    [<StructuredFormatDisplay ("{Pretty}")>]
    type BaseSizeT =
        | Sym of SizeSymbolT
        | Fixed of FracT

        member this.Pretty =
            match this with
            | Sym s -> sprintf "%A" s
            | Fixed f -> sprintf "%A" f


module SizeSymbol =
    let name sym =
        sym.Name

    let ofName name =
        {Name=name}


module Frac =
    let nom (frac: FracT) =
        frac.Nom

    let dnm (frac: FracT) =
        frac.Dnm

    let ofInt i =
        FracT (i)

    let toInt (frac: FracT) =
        frac.IntValue

    let zero =
        FracT (0L)

    let one =
        FracT (1L)

    let (|Zero|_|) frac =
        if frac = zero then Some ()
        else None

    let (|One|_|) frac =
        if frac = one then Some ()
        else None

    let (|Integral|_|) (frac: FracT) =
        if frac.Dnm = 1L then Some frac.Nom
        else None

    let roundTowardZero (f: FracT) =
        FracT (f.Nom / f.Dnm)

    let roundAwayFromZero (f: FracT) =
        if f.Nom % f.Dnm = 0L then
            FracT (f.Nom / f.Dnm)
        elif f.Nom > 0L then
            FracT (f.Nom / f.Dnm + 1L)
        else
            FracT (f.Nom / f.Dnm - 1L)

//[<AutoOpen>]
module SizeProductTypes = 
      
    /// product of elementary size specifications
    [<StructuredFormatDisplay("{Pretty}")>]
    type SizeProductT(symbols: Map<SizeSymbolT, int64>) =
        let symbols = symbols |> Map.filter (fun _ sPower -> sPower <> 0L)
  
        new() = SizeProductT(Map.empty)
        new(b: SizeSymbolT) = SizeProductT(b, 1L)
        new(b: SizeSymbolT, pow: int64) = SizeProductT(Map.empty |> Map.add b pow)

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
            if power = 1L then Some bs else None
        | _ -> None

    /// matches if product is empty (equal to 1)
    let (|Empty|_|) (sp: SizeProductT) =
        if isEmpty sp then Some () else None


//[<AutoOpen>]
module SizeMultinomTypes =
    open SizeProductTypes

    // symbolic multinomial
    [<StructuredFormatDisplay("{Pretty}")>]
    type SizeMultinomT (products: Map<SizeProductT, FracT>) =
        let products = products |> Map.filter (fun _ fac -> fac .<> Frac.zero)

        new (bs: BaseSizeT) = SizeMultinomT (Frac.one, bs, 1L)
        new (bs: BaseSizeT, pow: int64) = SizeMultinomT (Frac.one, bs, pow)
        new (fac: FracT, bs: BaseSizeT, pow: int64) =
            let m =
                match bs with
                | Sym s -> Map [SizeProductT (s, pow), fac]
                | Fixed f -> Map [SizeProductT (), fac * (pown f (int32 pow))]
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
    [<StructuredFormatDisplay ("{Pretty}")>]
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
                | [SizeProduct.SingleSymbol s, Frac.One] -> Base (Sym s)
                | [SizeProduct.Empty, f] -> Base (Fixed f)
                | [] -> Base (Fixed Frac.zero)
                | _ -> ss
            | _ -> ss

        static member get_Zero () = Base (Fixed Frac.zero)

        static member (~-) (ssa: SizeSpecT) =
            match ssa with
            | Base (Fixed Frac.Zero) -> ssa
            | Base b -> Multinom (-SizeMultinomT(b))
            | Broadcast -> Multinom (-SizeMultinomT(Fixed Frac.one))
            | Multinom m -> Multinom (-m)
            |> SizeSpecT.Simplify

        static member (+) (ssa: SizeSpecT, ssb: SizeSpecT) =
            match ssa, ssb with
            | Base (Fixed Frac.Zero), ss | ss, Base (Fixed Frac.Zero) -> ss
            | Broadcast, ss | ss, Broadcast -> ss + (Base (Fixed Frac.one))
            | Multinom ma, Multinom mb -> Multinom (ma + mb)
            | Multinom m, Base b | Base b, Multinom m -> Multinom (m + SizeMultinomT(b))
            | Base ba, Base bb -> Multinom (SizeMultinomT(ba) + SizeMultinomT(bb))
            |> SizeSpecT.Simplify

        static member (-) (ssa: SizeSpecT, ssb: SizeSpecT) =
            ssa + (-ssb)

        static member (*) (ssa: SizeSpecT, ssb: SizeSpecT) =
            match ssa, ssb with
            | Base (Fixed Frac.Zero), _ | _, Base (Fixed Frac.Zero) -> Base (Fixed Frac.zero)
            | Broadcast, ss | ss, Broadcast -> ss
            | Multinom ma, Multinom mb -> Multinom (ma * mb)
            | Multinom m, Base b | Base b, Multinom m -> Multinom (m * SizeMultinomT(b))
            | Base ba, Base bb -> Multinom (SizeMultinomT(ba) * SizeMultinomT(bb))
            |> SizeSpecT.Simplify

        static member Pow (ssa: SizeSpecT, pow: int64) =
            match pow with
            | 0L -> Base (Fixed Frac.one)
            | 1L -> ssa
            | _ ->
                match ssa with
                | Base (Fixed f) -> Base (Fixed (pown f (int32 pow)))
                | Base (Sym s) -> Multinom (SizeMultinomT (Sym s, pow))
                | Broadcast -> Broadcast
                | Multinom m ->
                    m
                    |> Seq.replicate (int32 pow)
                    |> Seq.reduce (*)
                    |> Multinom
            |> SizeSpecT.Simplify

        // operations with FracT
        static member (+) (ssa: SizeSpecT, ssb: FracT) = ssa + (Base (Fixed ssb))
        static member (+) (ssa: FracT, ssb: SizeSpecT) = (Base (Fixed ssa)) + ssb
        static member (-) (ssa: SizeSpecT, ssb: FracT) = ssa - (Base (Fixed ssb))
        static member (-) (ssa: FracT, ssb: SizeSpecT) = (Base (Fixed ssa)) - ssb
        static member (*) (ssa: SizeSpecT, ssb: FracT) = ssa * (Base (Fixed ssb))
        static member (*) (ssa: FracT, ssb: SizeSpecT) = (Base (Fixed ssa)) * ssb

        // operations with int
        static member (+) (ssa: SizeSpecT, ssb: int64) = ssa + FracT ssb
        static member (+) (ssa: int64, ssb: SizeSpecT) = FracT ssa + ssb
        static member (-) (ssa: SizeSpecT, ssb: int64) = ssa - FracT ssb
        static member (-) (ssa: int64, ssb: SizeSpecT) = FracT ssa - ssb
        static member (*) (ssa: SizeSpecT, ssb: int64) = ssa * FracT ssb
        static member (*) (ssa: int64, ssb: SizeSpecT) = FracT ssa * ssb

        /// equal size with broadcastability
        static member (%=) (ssa: SizeSpecT, ssb: SizeSpecT) = 
            SizeSpecT.Simplify ssa = SizeSpecT.Simplify ssb 

        /// equal size ignoring broadcastability
        static member (.=) (ssa: SizeSpecT, ssb: SizeSpecT) = 
            match SizeSpecT.Simplify ssa, SizeSpecT.Simplify ssb with
            | Broadcast, Base (Fixed Frac.One) | Base (Fixed Frac.One), Broadcast -> true
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

        member this.Pretty =
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
        Base (Fixed Frac.zero)

    /// not-broadcastable size one
    let one =
        Base (Fixed Frac.one)

    /// fixed integer size
    let fix s =
        Base (Fixed (FracT s))

    /// fixed fractional size
    let fixFrac nom dnm =
        Base (Fixed (FracT (nom, dnm)))

    /// symbolic size
    let symbol s =
        Base (Sym {Name=s})

    /// broadcastable size one
    let broadcastable =
        Broadcast

    /// extracts the size symbol
    let extractSymbol s =
        match s with
        | Base (Sym sym) -> sym
        | _ -> failwith "specified SizeSpec is not a symbol"

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
        | Base (Fixed (Frac.Integral i)) -> Some i
        | Broadcast -> Some 1L
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
    type NShapeSpecT = int64 list


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

    let emptyVector : ShapeSpecT = [SizeSpec.zero]

    /// pads shape by inserting broadcast dimension on the left
    let padLeft (sa: ShapeSpecT) : ShapeSpecT =
        (Broadcast)::sa

    /// pads shape by inserting broadcast dimension on the right
    let padRight (sa: ShapeSpecT) : ShapeSpecT =
        sa @ [Broadcast]

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

    let broadcast (sa: ShapeSpecT) dim size : ShapeSpecT =
        match sa.[dim] with
        | Broadcast -> List.set dim size sa
        | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeBroadcast)" dim sa

    let broadcastToShape (trgtShp: ShapeSpecT) (saIn: ShapeSpecT) : ShapeSpecT =
        let mutable sa = saIn
        if nDim sa <> nDim trgtShp then
            failwithf "cannot broadcast shape %A to shape %A" saIn trgtShp
        for d=0 to nDim trgtShp - 1 do
            match sa.[d], trgtShp.[d] with
            | al, bl when al = Broadcast -> sa <- broadcast sa d bl
            | al, bl when al = bl -> ()
            | _ -> failwithf "cannot broadcast shape %A to %A in dimension %d" sa trgtShp d
        sa

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
        | Base (Fixed Frac.One) | Broadcast -> List.set dim Broadcast sa
        | _ -> failwithf "cannot enable broadcasting for dimension %d of shape %A" dim sa

    let disableBroadcast dim (sa: ShapeSpecT) : ShapeSpecT =
        match sa.[dim] with
        | Base (Fixed Frac.One) | Broadcast -> List.set dim (Base (Fixed Frac.one)) sa
        | _ -> failwithf "cannot disable broadcasting for dimension %d of shape %A" dim sa

    let disableAllBroadcasts sa : ShapeSpecT =
        List.map (fun ss -> if ss = Broadcast then Base (Fixed Frac.one) else ss) sa
        
    /// True if both shape have the same number of elements and 
    /// are both broadcastable or non-broadcastable in each dimension.
    let equalWithBroadcastability (sa: ShapeSpecT) (sb: ShapeSpecT) =
        List.length sa = List.length sb &&
            List.forall2 SizeSpec.equalWithBroadcastability sa sb

    /// True if both shapes have the same number of elements in each dimension.
    /// Broadcastable and non-broadcastable are treated as equal.            
    let equalWithoutBroadcastability (sa: ShapeSpecT) (sb: ShapeSpecT) =
         List.length sa = List.length sb &&
            List.forall2 SizeSpec.equalWithoutBroadcastability sa sb

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

    /// basic range specification for one dimension
    type BaseRangeSpecT = SizeSpecT * SizeSpecT
    /// basic range specification for multiple dimensions
    type BaseRangesSpecT = BaseRangeSpecT list

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

    /// simple range specification for one dimension
    [<StructuredFormatDisplay("{Pretty}")>]
    type SimpleRangeSpecT<'Dyn> =
        | SRSSymStartSymEnd     of SizeSpecT * (SizeSpecT option)
        | SRSDynStartSymSize    of 'Dyn * SizeSpecT                    
        member this.Pretty =
            match this with
            | SRSSymStartSymEnd (first, Some last) -> sprintf "%A..%A" first last
            | SRSSymStartSymEnd (first, None) -> sprintf "%A.." first
            | SRSDynStartSymSize (first, size) -> sprintf "D%A..D%A+%A-1" first first size

    /// all elements
    let SRSAll = SRSSymStartSymEnd (SizeSpec.zero, None)
        
    /// simple range specification for multiple dimensions
    type SimpleRangesSpecT<'Dyn> = SimpleRangeSpecT<'Dyn> list


module BaseRangesSpec =

    /// Try to evalualte a BaseRangesSpecT to a numeric range.
    let tryEval (rng: BaseRangesSpecT) =
        let rec doEval rng =
            match rng with
            | (first, last) :: rrng ->
                match SizeSpec.tryEval first, SizeSpec.tryEval last, doEval rrng with
                | Some first, Some last, Some rrng -> Some ((first, last) :: rrng)
                | _ -> None
            | [] -> Some []
        doEval rng

    /// True if a BaseRangesSpecT can be evaluated to a numeric range.
    let canEval (rng: BaseRangesSpecT) =
        match tryEval rng with
        | Some _ -> true
        | None -> false

    /// Evaluates a BaseRangesSpecT to a numeric range.
    let eval (rng: BaseRangesSpecT) =
        match tryEval rng with
        | Some rng -> rng
        | None -> failwithf "cannot evaluate BaseRangesSpecT %A to numeric range" rng

    /// checks that the BaseRangesSpec is valid
    let check (rng: BaseRangesSpecT) =
        match tryEval rng with
        | Some rng ->
            for first, last in rng do
                if last < first then 
                    failwithf "invalid BaseRangesSpec: %A" rng
        | None -> ()

    /// True if two BaseRangesSpec overlap.
    /// Both BaseRangesSpec must be evaluateble to numeric ranges.
    let overlapping (a: BaseRangesSpecT) (b: BaseRangesSpecT) =
        check a; check b
        (eval a, eval b)
        ||> List.forall2 (fun (aFirst, aLast) (bFirst, bLast) ->
            aFirst <= bFirst && bFirst <= aLast ||
            aFirst <= bLast  && bLast  <= aLast ||
            bFirst <= aFirst && aLast  <= bLast)

    /// True if any two ranges are overlapping.
    /// This has complexity O(N^2) currently.
    /// All BaseRangesSpec must be evaluateble to numeric ranges.
    let areOverlapping (rngs: BaseRangesSpecT list) =       
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
    let areCoveringWithoutOverlap (shp: ShapeSpecT) (rngs: BaseRangesSpecT list) =       
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


module SimpleRangeSpec =
    open Tensor

    /// evaluate a SimpleRangeSpecT to a RangeT
    let eval dynEvaluator rs =
        match rs with
        | SRSSymStartSymEnd (s, fo) -> 
            Rng.Rng (Some (SizeSpec.eval s), Option.map SizeSpec.eval fo)
        | SRSDynStartSymSize (s, elems) -> 
            let sv = dynEvaluator s
            Rng.Rng (Some sv, Some (sv + SizeSpec.eval elems))

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

    let (|Dynamic|Static|) rs =
        if isDynamic rs then Dynamic else Static

    let toBaseRangeSpec (size: SizeSpecT) rs =
        match rs with
        | SRSSymStartSymEnd (first, Some last) -> first, last
        | SRSSymStartSymEnd (first, None) -> first, size - 1L
        | _ -> failwithf "cannot convert %A to BaseRangeSpec" rs

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

    let toBaseRangesSpec (shape: ShapeSpecT) rs =
        (shape, rs) ||> List.map2 SimpleRangeSpec.toBaseRangeSpec



