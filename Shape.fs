module Shape

open Util

type SymbolSpec = string

/// elementary size specification, can be either a symbol or a fixed quantity
type BaseSize =
    | Symbol of SymbolSpec
    | Fixed of int
       
/// product of elementary size specifications
type SizeProductT(factor: int, inSymbols: Map<SymbolSpec, int>) =
    let symbols =
        Map.fold (fun c sBase sPower ->
                    match sBase, sPower with
                    | _, p when p = 0 -> c
                    | _, p when p > 0 -> Map.add sBase sPower c
                    | _ -> failwithf "SizeProduct cannot have negative exponents: %A" inSymbols) 
            Map.empty inSymbols
  
    new() = SizeProductT(1, Map.empty)
    new(f: int) = SizeProductT(f, Map.empty)
    new(b: SymbolSpec) = SizeProductT(1, Map.empty |> Map.add b 1)

    member this.Symbols = symbols
    member this.Factor = factor

    static member (*) (a: SizeProductT, b: SizeProductT) =
        let pSymbols = 
            Map.fold 
                (fun p bBase bPower -> 
                    match Map.tryFind bBase p with
                    | Some pPower -> Map.add bBase (pPower + bPower) p
                    | None -> Map.add bBase bPower p) 
                a.Symbols b.Symbols
        SizeProductT(a.Factor * b.Factor, pSymbols)
         
    static member (*) (a: SizeProductT, b: SymbolSpec) = a * SizeProductT(b)
    static member (*) (a: SizeProductT, b: int) = a * SizeProductT(b)
    static member (*) (a: SymbolSpec, b: SizeProductT) = SizeProductT(a) * b
    static member (*) (a: int, b: SizeProductT) = SizeProductT(a) * b

    override this.ToString() = 
        let ft = if this.Factor = 1 then "" else sprintf "%d " this.Factor
        let txt = Map.fold (fun txt tBase tPower -> 
                                let t = if tPower = 1 then tBase else sprintf "%s**%d" tBase tPower
                                txt + t + " ") 
                            ft this.Symbols                
        "'" + txt.Trim() + "'"
                     
    override this.Equals(otherObj) =
        match otherObj with
        | :? SizeProductT as other ->
            this.Symbols = other.Symbols && this.Factor = other.Factor
        | _ -> false

    override this.GetHashCode() =
        hash (this.Factor, this.Symbols)

module SizeProduct =
    let isEmpty (p: SizeProductT) =
        Map.isEmpty p.Symbols

    let ofBaseSize (bs: BaseSize) =
        match bs with
        | Symbol s -> SizeProductT(s)
        | Fixed f -> SizeProductT(f)

    let empty =
        SizeProductT()

    let (|SingleSymbol|_|) (sp: SizeProductT) =
        let mc = Map.toList sp.Symbols
        match sp.Factor, List.length mc with
        | 1, 1 -> 
            let bs, power = mc.[0]
            if power = 1 then Some bs else None
        | _ -> None

    let (|SingleFactor|_|) (sp: SizeProductT) =
        if Map.isEmpty sp.Symbols then
            Some sp.Factor
        else
            None

/// size specification of a dimension (axis)
type SizeSpecT =
    | Base of BaseSize              // fixed size or symbol
    | Broadcast                     // size 1 and broadcastable
    | Product of SizeProductT       // product of fixed sizes and symbols

     static member (*) (ssa: SizeSpecT, ssb: SizeSpecT) =
        match ssa, ssb with
        | Base (Fixed 0), _ | _, Base (Fixed 0) -> Base (Fixed 0)
        | Broadcast, ss | ss, Broadcast -> ss
        | Product spa, Product spb -> Product (spa * spb)
        | Product sp, Base b | Base b, Product sp -> Product (sp * (SizeProduct.ofBaseSize b))
        | Base ba, Base bb -> Product ((SizeProduct.ofBaseSize ba) * (SizeProduct.ofBaseSize bb))


module SizeSpec =

    let simplify (ss: SizeSpecT) = 
        match ss with
        | Product (SizeProduct.SingleFactor f) -> Base (Fixed f)
        | Product (SizeProduct.SingleSymbol s) -> Base (Symbol s)
        | _ -> ss

    /// True if both sizes have the same number of elements and 
    /// are both broadcastable or non-broadcastable.
    let equalWithBroadcastability ssa ssb =
        simplify ssa = simplify ssb 

    /// True if both sizes have the same number of elements.
    /// Broadcastable and non-broadcastable are treated as equal.
    let equalWithoutBroadcastability ssa ssb =
        match simplify ssa, simplify ssb with
        | Broadcast, Base (Fixed 1) | Base (Fixed 1), Broadcast -> true
        | a, b -> a = b

    /// not-broadcastable size one
    let one =
        Base (Fixed 1)

    /// symbolic size
    let symbol s =
        Base (Symbol s)

    /// broadcastable size one
    let broadcastable =
        Broadcast

let symbol = SizeSpec.symbol

type SizeSpecT with
    static member (%=) (ssa: SizeSpecT, ssb: SizeSpecT) = SizeSpec.equalWithBroadcastability ssa ssb
    static member (.=) (ssa: SizeSpecT, ssb: SizeSpecT) = SizeSpec.equalWithoutBroadcastability ssa ssb

/// shape specifcation of a tensor
type ShapeSpecT = SizeSpecT list


module ShapeSpec =
    let withoutAxis ax sa =
        List.without ax sa

    let nDim sa =
        List.length sa

    let nElem sa =
        List.fold 
            (fun p ss ->
                match ss with
                | Base (Symbol b) -> p * b
                | Base (Fixed f) -> p * f
                | Broadcast -> p 
                | Product pb -> p * pb)
            SizeProduct.empty sa 
            |> Product   
            |> SizeSpec.simplify       

    let concat sa sb =
        sa @ sb

    let transpose sa =
        List.rev sa

    let swap (ax1: int) (ax2: int) (sa: ShapeSpecT) =
        sa  |> List.set ax1 sa.[ax2]
            |> List.set ax2 sa.[ax1]

    let scalar = []

    let vector (ss: SizeSpecT) = [ss]

    let matrix (sr: SizeSpecT) (sc: SizeSpecT) = [sr; sc]

    let padLeft sa =
        (Broadcast)::sa

    let padRight sa =
        sa @ [Broadcast]

    let rec padToSame sa sb =
        if nDim sa < nDim sb then
            padToSame (padLeft sa) sb
        elif nDim sb < nDim sa then
            padToSame sa (padLeft sb)
        else
            sa, sb

    let broadcast (sa: ShapeSpecT) dim size =
        match sa.[dim] with
            | Broadcast -> List.set dim size sa
            | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeBroadcast)" dim sa

    let broadcastToSame saIn sbIn =
        let mutable sa, sb = saIn, sbIn
        if nDim sa <> nDim sb then 
            failwithf "cannot broadcast shapes %A and %A of different rank to same size" sa sb
        for d = 0 to (nDim sa) - 1 do
            match sa.[d], sb.[d] with
                | al, bl when al = bl -> ()
                | al, bl when al = Broadcast -> sa <- broadcast sa d bl
                | al, bl when bl = Broadcast -> sb <- broadcast sb d al
                | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb

    let enableBroadcast (sa: ShapeSpecT) dim =
        match sa.[dim] with
        | Base (Fixed 1) | Broadcast -> List.set dim Broadcast sa
        | _ -> failwithf "cannot enable broadcasting for dimension %d of shape %A" dim sa

    let disableBroadcast (sa: ShapeSpecT) dim =
        match sa.[dim] with
        | Base (Fixed 1) | Broadcast -> List.set dim (Base (Fixed 1)) sa
        | _ -> failwithf "cannot disable broadcasting for dimension %d of shape %A" dim sa

    let disableAllBroadcasts sa =
        List.map (fun ss -> if ss = Broadcast then Base (Fixed 1) else ss) sa

