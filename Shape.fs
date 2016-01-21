module Shape

open Util


/// size specification of a dimension (axis)
type SizeSpec =
    | SizeSymbol of string
    | SizeConst of int
    | SizeOne
    | SizeProduct of SizeSpec list

/// shape specifcation of a tensor
type ShapeSpecT = SizeSpec list

module ShapeSpec =
    let withoutAxis ax sa =
        List.without ax sa

    let nDim sa =
        List.length sa

    let nElem sa =
        match nDim sa with
        | 0 -> SizeOne
        | 1 -> sa.[0]
        | _ -> SizeProduct(sa)

    let concat sa sb =
        sa @ sb

    let transpose sa =
        List.rev sa

    let swap (ax1: int) (ax2: int) (sa: ShapeSpecT) =
        sa  |> List.set ax1 sa.[ax2]
            |> List.set ax2 sa.[ax1]

    let scalar = []

    let vector (ss: SizeSpec) = [ss]

    let matrix (sr: SizeSpec) (sc: SizeSpec) = [sr; sc]

    let padLeft sa =
        (SizeOne)::sa

    let padRight sa =
        sa @ [SizeOne]

    let broadcast (sa: ShapeSpecT) dim size =
        match sa.[dim] with
            | SizeOne -> List.set dim size sa
            | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeOne)" dim sa

    let broadcastToSame saIn sbIn =
        let mutable sa = saIn
        let mutable sb = sbIn 
        while nDim sa < nDim sb do
            sa <- padLeft sa
        while nDim sb < nDim sa do
            sb <- padLeft sb
        for d = 0 to (nDim sa) - 1 do
            match sa.[d], sb.[d] with
                | al, bl when al = bl -> ()
                | al, bl when al = SizeOne -> sa <- broadcast sa d bl
                | al, bl when bl = SizeOne -> sb <- broadcast sb d al
                | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb
