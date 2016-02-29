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
    let name sym =
        sym.Name

    let flexible sym =
        sym.Flexible


[<AutoOpen>]
module SizeProductTypes = 
      
    /// product of elementary size specifications
    type SizeProductT(factor: int, inSymbols: Map<SizeSymbolT, int>) =
        let symbols =
            Map.fold (fun c sBase sPower ->
                        match sBase, sPower with
                        | _, p when p = 0 -> c
                        | _, p when p > 0 -> Map.add sBase sPower c
                        | _ -> failwithf "SizeProduct cannot have negative exponents: %A" inSymbols) 
                Map.empty inSymbols
  
        new() = SizeProductT(1, Map.empty)
        new(f: int) = SizeProductT(f, Map.empty)
        new(b: SizeSymbolT) = SizeProductT(1, Map.empty |> Map.add b 1)

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
         
        static member (*) (a: SizeProductT, b: SizeSymbolT) = a * SizeProductT(b)
        static member (*) (a: SizeProductT, b: int) = a * SizeProductT(b)
        static member (*) (a: SizeSymbolT, b: SizeProductT) = SizeProductT(a) * b
        static member (*) (a: int, b: SizeProductT) = SizeProductT(a) * b
        static member (*) (a: SizeProductT, b: BaseSizeT) =
            match b with
            | Sym s -> a * s
            | Fixed f -> a * f
        static member (*) (a: BaseSizeT, b: SizeProductT) = b * a

        override this.ToString() = 
            let ft = if this.Factor = 1 then [] else [sprintf "%d " this.Factor]
            let st = 
                this.Symbols
                |> Map.toSeq
                |> Seq.map 
                    (fun (tBase, tPower) ->
                        if tPower = 1 then sprintf "%A" tBase 
                        else sprintf "%A**%d" tBase tPower)
                |> Seq.toList
            String.concat " * " (ft @ st)
                        
        override this.Equals(otherObj) =
            match otherObj with
            | :? SizeProductT as other ->
                this.Symbols = other.Symbols && this.Factor = other.Factor
            | _ -> false

        override this.GetHashCode() =
            hash (this.Factor, this.Symbols)

        interface System.IComparable with
            member this.CompareTo otherObj =
                match otherObj with
                | :? SizeProductT as other ->
                    let ms = Map.add {Name="__factor__"; Flexible=false} this.Factor this.Symbols
                    let os = Map.add {Name="__factor__"; Flexible=false} other.Factor other.Symbols
                    compare ms os
                | _ -> invalidArg "otherObj" "cannot compare values of different types"


module SizeProduct =
    open SizeSymbolTypes

    let isEmpty (p: SizeProductT) =
        Map.isEmpty p.Symbols

    let ofBaseSize (bs: BaseSizeT) =
        match bs with
        | Sym s -> SizeProductT(s)
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

    let mult (sBase: BaseSizeT) (sPower: int) (p: SizeProductT) =
        Seq.fold (*) p (Seq.replicate sPower sBase) 

    /// tries to evaluate to a numeric size
    let tryEval (p: SizeProductT) =
        match p with
        | SingleFactor f -> Some f
        | _ -> None

    /// evaluate to a numeric size
    let eval (p: SizeProductT) =
        match tryEval p with
        | Some f -> f
        | _ -> failwithf "cannot evaluate %A to a numeric size, since it contains symbols" p

//    /// true if it can be evaluated to a numeric size
//    let canEval (p: SizeProductT) =
//        match p with
//        | SingleFactor f -> true
//        | _ -> false



[<AutoOpen>]
module SizeSpecTypes =
    /// size specification of a dimension (axis)
    [<StructuredFormatDisplay ("{PrettyString}")>]
    [<StructuralEquality; StructuralComparison>]
    type SizeSpecT =
        | Base of BaseSizeT              // fixed size or symbol
        | Broadcast                     // size 1 and broadcastable
        | Product of SizeProductT       // product of fixed sizes and symbols
        
        static member (*) (ssa: SizeSpecT, ssb: SizeSpecT) =
            match ssa, ssb with
            | Base (Fixed 0), _ | _, Base (Fixed 0) -> Base (Fixed 0)
            | Broadcast, ss | ss, Broadcast -> ss
            | Product spa, Product spb -> Product (spa * spb)
            | Product sp, Base b | Base b, Product sp -> Product (sp * (SizeProduct.ofBaseSize b))
            | Base ba, Base bb -> Product ((SizeProduct.ofBaseSize ba) * (SizeProduct.ofBaseSize bb))

        /// simplify size specification
        static member Simplify (ss: SizeSpecT) =
            match ss with
            | Product (SizeProduct.SingleFactor f) -> Base (Fixed f)
            | Product (SizeProduct.SingleSymbol s) -> Base (Sym s)
            | _ -> ss

        /// equal size with broadcastability
        static member (%=) (ssa: SizeSpecT, ssb: SizeSpecT) = SizeSpecT.Simplify ssa = SizeSpecT.Simplify ssb 

        /// equal size ignoring broadcastability
        static member (.=) (ssa: SizeSpecT, ssb: SizeSpecT) = 
            match SizeSpecT.Simplify ssa, SizeSpecT.Simplify ssb with
            | Broadcast, Base (Fixed 1) | Base (Fixed 1), Broadcast -> true
            | a, b -> a = b

        member this.PrettyString =
            match this with
            | Base b -> sprintf "%A" b
            | Broadcast -> "1*"
            | Product p -> sprintf "%A" p


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
        Base (Sym {Name=s; Flexible=true})

    /// broadcastable size one
    let broadcastable =
        Broadcast

    /// evaluate symbolic size specification to a number
    let tryEval ss =
        match ss with
        | Base (Sym sym) ->  None
        | Base (Fixed f) -> Some f
        | Broadcast -> Some 1
        | Product p -> SizeProduct.tryEval p        

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

    let withoutAxis ax sa =
        sa |> List.without ax

    let insertBroadcastAxis ax sa =
        sa |> List.insert ax Broadcast

    let set ax size sa =
        sa |> List.set ax size

    let nDim sa =
        List.length sa

    let nElem sa =
        List.fold 
            (fun p ss ->
                match ss with
                | Base (Sym b) -> p * b
                | Base (Fixed f) -> p * f
                | Broadcast -> p 
                | Product pb -> p * pb)
            SizeProduct.empty sa 
            |> Product   
            |> SizeSpec.simplify       

    let flatten sa =
        [nElem sa]

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

    let emptyVector = [Base (Fixed 0)]

    let padLeft sa =
        (Broadcast)::sa

    let padRight sa =
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


//[<AutoOpen>]
//module SymRngTypes =



        