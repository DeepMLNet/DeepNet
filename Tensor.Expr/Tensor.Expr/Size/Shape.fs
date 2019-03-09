namespace Tensor.Expr

open DeepNet.Utils


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

    let fix (nShape: int64 list) : ShapeSpec =
        nShape |> List.map SizeSpec.fix

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
                
