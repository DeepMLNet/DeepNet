namespace Tensor.Expr

open DeepNet.Utils


/// A size symbol.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type SizeSym = SizeSym of string with

    /// Name.
    member this.Name =
        let (SizeSym name) = this
        name

    /// Name.
    static member name (SizeSym name) = name

    /// pretty string
    member this.Pretty = 
        sprintf "\"%s\"" this.Name



/// Elementary size specification.
/// Can be either a symbol or a fixed quantity.
[<RequireQualifiedAccess; StructuredFormatDisplay("{Pretty}")>]
type SizeAtom =
    /// symbolic size
    | Sym of SizeSym
    /// numeric size
    | Fixed of Frac

    /// pretty string
    member this.Pretty =
        match this with
        | SizeAtom.Sym s -> sprintf "%A" s
        | SizeAtom.Fixed f -> sprintf "%A" f



/// Product of elementary size specifications with integer exponents.
[<StructuredFormatDisplay("{Pretty}")>]
type SizeProduct(symbols: Map<SizeSym, int64>) =
    let symbols = symbols |> Map.filter (fun _ sPower -> sPower <> 0L)
  
    new() = SizeProduct(Map.empty)
    new(b: SizeSym) = SizeProduct(b, 1L)
    new(b: SizeSym, pow: int64) = SizeProduct(Map.empty |> Map.add b pow)

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
         
    static member (*) (a: SizeProduct, b: SizeSym) = a * SizeProduct(b)
    static member (*) (a: SizeSym, b: SizeProduct) = SizeProduct(a) * b

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


/// Multinomial of symbolic sizes with rational coefficients.
[<StructuredFormatDisplay("{Pretty}")>]
type SizeMultinom (products: Map<SizeProduct, Frac>) =
    let products = products |> Map.filter (fun _ fac -> fac .<> Frac.zero)

    new (bs: SizeAtom) = SizeMultinom (Frac.one, bs, 1L)
    new (bs: SizeAtom, pow: int64) = SizeMultinom (Frac.one, bs, pow)
    new (fac: Frac, bs: SizeAtom, pow: int64) =
        let m =
            match bs with
            | SizeAtom.Sym s -> Map [SizeProduct (s, pow), fac]
            | SizeAtom.Fixed f -> Map [SizeProduct (), fac * (pown f (int32 pow))]
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



/// Symbolic size. 
[<RequireQualifiedAccess; StructuralEquality; StructuralComparison; StructuredFormatDisplay ("{Pretty}")>]
type Size =
    /// Fixed size of symbol.
    | Atom of SizeAtom               
    /// Broadcastable size one.
    | Broadcast                      
    /// Multinom of symbolic sizes.
    | Multinom of SizeMultinom       


    static member (~-) (ssa: Size) =
        match ssa with
        | Size.Atom (SizeAtom.Fixed Frac.Zero) -> ssa
        | Size.Atom b -> Size.Multinom (-SizeMultinom(b))
        | Size.Broadcast -> Size.Multinom (-SizeMultinom(SizeAtom.Fixed Frac.one))
        | Size.Multinom m -> Size.Multinom (-m)
        |> Size.simplify

    static member (+) (ssa: Size, ssb: Size) =
        match ssa, ssb with
        | Size.Atom (SizeAtom.Fixed Frac.Zero), ss | ss, Size.Atom (SizeAtom.Fixed Frac.Zero) -> ss
        | Size.Broadcast, ss | ss, Size.Broadcast -> ss + (Size.Atom (SizeAtom.Fixed Frac.one))
        | Size.Multinom ma, Size.Multinom mb -> Size.Multinom (ma + mb)
        | Size.Multinom m, Size.Atom b | Size.Atom b, Size.Multinom m -> Size.Multinom (m + SizeMultinom(b))
        | Size.Atom ba, Size.Atom bb -> Size.Multinom (SizeMultinom(ba) + SizeMultinom(bb))
        |> Size.simplify

    static member (-) (ssa: Size, ssb: Size) =
        ssa + (-ssb)

    static member (*) (ssa: Size, ssb: Size) =
        match ssa, ssb with
        | Size.Atom (SizeAtom.Fixed Frac.Zero), _ | _, Size.Atom (SizeAtom.Fixed Frac.Zero) -> Size.Atom (SizeAtom.Fixed Frac.zero)
        | Size.Broadcast, ss | ss, Size.Broadcast -> ss
        | Size.Multinom ma, Size.Multinom mb -> Size.Multinom (ma * mb)
        | Size.Multinom m, Size.Atom b | Size.Atom b, Size.Multinom m -> Size.Multinom (m * SizeMultinom(b))
        | Size.Atom ba, Size.Atom bb -> Size.Multinom (SizeMultinom(ba) * SizeMultinom(bb))
        |> Size.simplify

    static member Pow (ssa: Size, pow: int64) =
        match pow with
        | 0L -> Size.Atom (SizeAtom.Fixed Frac.one)
        | 1L -> ssa
        | _ ->
            match ssa with
            | Size.Atom (SizeAtom.Fixed f) -> Size.Atom (SizeAtom.Fixed (pown f (int32 pow)))
            | Size.Atom (SizeAtom.Sym s) -> Size.Multinom (SizeMultinom (SizeAtom.Sym s, pow))
            | Size.Broadcast -> Size.Broadcast
            | Size.Multinom m ->
                m
                |> Seq.replicate (int32 pow)
                |> Seq.reduce (*)
                |> Size.Multinom
        |> Size.simplify

    // operations with FracT
    static member (+) (ssa: Size, ssb: Frac) = ssa + (Size.Atom (SizeAtom.Fixed ssb))
    static member (+) (ssa: Frac, ssb: Size) = (Size.Atom (SizeAtom.Fixed ssa)) + ssb
    static member (-) (ssa: Size, ssb: Frac) = ssa - (Size.Atom (SizeAtom.Fixed ssb))
    static member (-) (ssa: Frac, ssb: Size) = (Size.Atom (SizeAtom.Fixed ssa)) - ssb
    static member (*) (ssa: Size, ssb: Frac) = ssa * (Size.Atom (SizeAtom.Fixed ssb))
    static member (*) (ssa: Frac, ssb: Size) = (Size.Atom (SizeAtom.Fixed ssa)) * ssb

    // operations with int
    static member (+) (ssa: Size, ssb: int64) = ssa + Frac ssb
    static member (+) (ssa: int64, ssb: Size) = Frac ssa + ssb
    static member (-) (ssa: Size, ssb: int64) = ssa - Frac ssb
    static member (-) (ssa: int64, ssb: Size) = Frac ssa - ssb
    static member (*) (ssa: Size, ssb: int64) = ssa * Frac ssb
    static member (*) (ssa: int64, ssb: Size) = Frac ssa * ssb

    /// True if both sizes have the same number of elements and 
    /// are both broadcastable or non-broadcastable.
    static member equalRespectingBc (ssa: Size) (ssb: Size) = 
        Size.simplify ssa = Size.simplify ssb 

    /// True if both sizes have the same number of elements.
    /// Broadcastable and non-broadcastable are treated as equal.
    static member equalIgnoringBc (ssa: Size) (ssb: Size) = 
        match Size.simplify ssa, Size.simplify ssb with
        | Size.Broadcast, Size.Atom (SizeAtom.Fixed Frac.One) 
        | Size.Atom (SizeAtom.Fixed Frac.One), Size.Broadcast -> true
        | a, b -> a = b

    /// the set of all contained SizeSymbols
    member this.ContainedSyms =
        match this with
        | Size.Atom (SizeAtom.Sym s)   -> Set [s]
        | Size.Atom (SizeAtom.Fixed _) -> Set.empty
        | Size.Broadcast               -> Set.empty
        | Size.Multinom m              -> m.ContainedSizeSymbols
            
    /// Pretty string.
    member this.Pretty =
        match this with
        | Size.Atom b -> sprintf "%A" b
        | Size.Broadcast -> "1*"
        | Size.Multinom m -> sprintf "%A" m

    /// simplify size specification
    static member simplify (ss: Size) = 
        match ss with
        | Size.Multinom m -> 
            match m.Products |> Map.toList with
            | [SizeProduct.SingleSymbol s, Frac.One] -> Size.Atom (SizeAtom.Sym s)
            | [SizeProduct.Empty, f] -> Size.Atom (SizeAtom.Fixed f)
            | [] -> Size.Atom (SizeAtom.Fixed Frac.zero)
            | _ -> ss
        | _ -> ss

    /// size zero
    static member zero = Size.Atom (SizeAtom.Fixed Frac.zero)
    /// size zero
    static member Zero = Size.zero

    /// not-broadcastable size one
    static member one = Size.Atom (SizeAtom.Fixed Frac.one)
    /// not-broadcastable size one
    static member One = Size.one

    /// fixed integer size
    static member fix s = Size.Atom (SizeAtom.Fixed (Frac s))

    /// fixed fractional size
    static member fixFrac nom dnm = Size.Atom (SizeAtom.Fixed (Frac (nom, dnm)))

    /// symbolic size
    static member sym s = Size.Atom (SizeAtom.Sym s)

    /// broadcastable size one
    static member broadcastable = Size.Broadcast

    /// Substitutes symbol values into size.
    static member subst symVals ss =
        match ss with
        | Size.Atom (SizeAtom.Sym sym) ->
            match Map.tryFind sym symVals with
            | Some sv -> Size.subst symVals sv
            | None -> ss
        | Size.Atom (SizeAtom.Fixed _) -> ss
        | Size.Broadcast -> ss
        | Size.Multinom m -> 
            // rebuild multinom with substituted values
            (Size.zero, m.Products)
            ||> Map.fold 
                (fun substSum prod fac ->               
                    let substProd = 
                        (Size.one, prod.Symbols)
                        ||> Map.fold 
                            (fun substProd sBaseSym sPow ->
                                let sBaseSubst = Size.subst symVals (Size.Atom (SizeAtom.Sym sBaseSym))
                                substProd * (sBaseSubst ** sPow))
                    substSum + fac * substProd)
        |> Size.simplify
            
    /// evaluate symbolic size specification to a number
    static member tryEval ss =
        match Size.simplify ss with
        | Size.Atom (SizeAtom.Fixed (Frac.Integral i)) -> Some i
        | Size.Broadcast -> Some 1L
        | _ -> None

    /// true, if evaluation to numeric shape is possible
    static member canEval ss =
        match Size.tryEval ss with
        | Some _ -> true
        | None -> false

    /// evaluate symbolic size specification to a number
    static member eval ss =
        match Size.tryEval ss with
        | Some s -> s
        | None -> failwithf "cannot evaluate %A to a numeric size since it contains symbols" ss

    /// returns the set of all contained SizeSymbols
    static member containedSyms (ss: Size) = ss.ContainedSyms
