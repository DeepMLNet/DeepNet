module Shape

open Util


/// size specification of a dimension (axis)
type SizeSpecT =
    | SizeSymbol of string
    | SizeConst of int
    | SizeBroadcast
    | SizeProduct of SizeSpecT list

module SizeSpec =
    let rec simplify ss =
        match ss with
        | SizeProduct sp -> simplifyProduct 1 sp
        | _ -> ss

    and simplifyProduct constProd sp =
        match sp with
        | [] -> if constProd = 1 then SizeBroadcast else SizeConst constProd
        | s::sps ->
            match simplify s with
            | SizeConst c -> simplifyProduct (c * constProd) sps
            | SizeBroadcast -> simplifyProduct constProd sps
            | SizeProduct spr -> simplifyProduct constProd (spr @ sps)
            | s -> doMultiply s (simplifyProduct constProd sps)

    and doMultiply ssa ssb =
        match ssa, ssb with
        | SizeConst 0, _ | _, SizeConst 0 -> SizeConst 0
        | SizeBroadcast, _ -> ssb
        | _, SizeBroadcast -> ssa
        | SizeProduct spa, SizeProduct spb -> SizeProduct (spa @ spb)
        | SizeProduct spa, _ -> SizeProduct (ssb::spa)
        | _, SizeProduct spb -> SizeProduct (ssa::spb)
        | _, _ -> SizeProduct [ssa; ssb]

    let multiply ssa ssb =
        doMultiply ssa ssb |> simplify

    let equal ssa ssb =
        simplify ssa = simplify ssb // TODO


/// shape specifcation of a tensor
type ShapeSpecT = SizeSpecT list

module ShapeSpec =
    let withoutAxis ax sa =
        List.without ax sa

    let nDim sa =
        List.length sa

    let nElem sa =
        match nDim sa with
        | 0 -> SizeConst 1
        | 1 -> 
            match sa.[0] with
            | SizeBroadcast -> SizeConst 1
            | s -> s
        | _ -> SizeProduct(sa)
      
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
        (SizeBroadcast)::sa

    let padRight sa =
        sa @ [SizeBroadcast]

    let rec padToSame sa sb =
        if nDim sa < nDim sb then
            padToSame (padLeft sa) sb
        elif nDim sb < nDim sa then
            padToSame sa (padLeft sb)
        else
            sa, sb

    let broadcast (sa: ShapeSpecT) dim size =
        match sa.[dim] with
            | SizeBroadcast -> List.set dim size sa
            | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeOne)" dim sa

    let broadcastToSame saIn sbIn =
        let mutable sa, sb = padToSame saIn sbIn
        for d = 0 to (nDim sa) - 1 do
            match sa.[d], sb.[d] with
                | al, bl when al = bl -> ()
                | al, bl when al = SizeBroadcast -> sa <- broadcast sa d bl
                | al, bl when bl = SizeBroadcast -> sb <- broadcast sb d al
                | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb
