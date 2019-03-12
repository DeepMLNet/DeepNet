namespace Tensor.Expr

open DeepNet.Utils


/// A symbolic size.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type SizeSymbol = SizeSymbol of string with

    /// pretty string
    member this.Pretty = 
        let (SizeSymbol name) = this
        sprintf "\"%s\"" name

    /// identifier
    static member name (SizeSymbol name) = name

    /// creates a symbolic size with the specified identifier
    static member ofName name = SizeSymbol name


/// Elementary size specification
/// Can be either a symbol or a fixed quantity.
[<RequireQualifiedAccess; StructuredFormatDisplay("{Pretty}")>]
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
    static member Zero = SizeSpec.zero

    /// not-broadcastable size one
    static member one = SizeSpec.Base (BaseSize.Fixed Frac.one)
    static member One = SizeSpec.one

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
            (SizeSpec.zero, m.Products)
            ||> Map.fold 
                (fun substSum prod fac ->               
                    let substProd = 
                        (SizeSpec.one, prod.Symbols)
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
         
