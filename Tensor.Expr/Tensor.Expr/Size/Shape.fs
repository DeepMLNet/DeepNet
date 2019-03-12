namespace Tensor.Expr

open DeepNet.Utils


/// Symbolic shape of an expression.
type Shape = Size list



/// Functions for working with symbolic expression shapes.
module Shape =

    let insertAxis ax ss (sa: Shape) : Shape =
        sa |> List.insert ax ss

    let withoutAxis ax (sa: Shape) : Shape =
        sa |> List.without ax

    let insertBroadcastAxis ax (sa: Shape) : Shape =
        sa |> insertAxis ax Size.Broadcast

    let set ax size (sa: Shape) : Shape =
        sa |> List.set ax size

    let nDim (sa: Shape) =
        List.length sa

    let nElem (sa: Shape) =
        if List.isEmpty sa then Size.one
        else List.reduce (*) sa

    let flatten (sa: Shape) : Shape =
        [nElem sa]

    let concat (sa: Shape) (sb: Shape) : Shape =
        sa @ sb

    let transpose (sa: Shape) : Shape =
        if nDim sa <> 2 then failwithf "need matrix to transpose but have shape %A" sa
        List.rev sa

    let swap (ax1: int) (ax2: int) (sa: Shape) : Shape =
        sa  |> List.set ax1 sa.[ax2]
            |> List.set ax2 sa.[ax1]

    let scalar : Shape = []

    let vector (ss: Size) : Shape = [ss]

    let matrix (sr: Size) (sc: Size) : Shape = [sr; sc]

    let emptyVector : Shape = [Size.zero]

    let fix (nShape: int64 list) : Shape =
        nShape |> List.map Size.fix

    /// pads shape by inserting broadcast dimension on the left
    let padLeft (sa: Shape) : Shape =
        (Size.Broadcast)::sa

    /// pads shape by inserting broadcast dimension on the right
    let padRight (sa: Shape) : Shape =
        sa @ [Size.Broadcast]

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

    let broadcast (sa: Shape) dim size : Shape =
        match sa.[dim] with
        | Size.Broadcast -> List.set dim size sa
        | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeBroadcast)" dim sa

    let broadcastToShape (trgtShp: Shape) (saIn: Shape) : Shape =
        let mutable sa = saIn
        if nDim sa <> nDim trgtShp then
            failwithf "cannot broadcast shape %A to shape %A" saIn trgtShp
        for d=0 to nDim trgtShp - 1 do
            match sa.[d], trgtShp.[d] with
            | al, bl when al = Size.Broadcast -> sa <- broadcast sa d bl
            | al, bl when al = bl -> ()
            | _ -> failwithf "cannot broadcast shape %A to %A in dimension %d" sa trgtShp d
        sa

    let broadcastToSameInDims dims mustEqual saIn sbIn =
        let mutable sa, sb = saIn, sbIn
        for d in dims do
            if not (d < nDim sa && d < nDim sb) then
                failwithf "cannot broadcast shapes %A and %A to same size in non-existant dimension %d" sa sb d
            match sa.[d], sb.[d] with
            | al, bl when al = Size.Broadcast -> sa <- broadcast sa d bl
            | al, bl when bl = Size.Broadcast -> sb <- broadcast sb d al
            | al, bl when (if mustEqual then al = bl else true) -> ()        
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb

    let broadcastToSameInDimsMany dims mustEqual sas =
        let mutable sas = sas
        for d in dims do
            if not (sas |> List.forall (fun sa -> d < nDim sa)) then
                failwithf "cannot broadcast shapes %A to same size in non-existant dimension %d" sas d
            let ls = sas |> List.map (fun sa -> sa.[d])
            if ls |> List.exists ((=) Size.Broadcast) then
                let nonBc = ls |> List.filter (fun l -> l <> Size.Broadcast)
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

    let enableBc dim (sa: Shape) : Shape =
        match sa.[dim] with
        | Size.Atom (SizeAtom.Fixed Frac.One) | Size.Broadcast -> List.set dim Size.Broadcast sa
        | _ -> failwithf "cannot enable broadcasting for dimension %d of shape %A" dim sa

    let disableBc dim (sa: Shape) : Shape =
        match sa.[dim] with
        | Size.Atom (SizeAtom.Fixed Frac.One) | Size.Broadcast -> List.set dim (Size.Atom (SizeAtom.Fixed Frac.one)) sa
        | _ -> failwithf "cannot disable broadcasting for dimension %d of shape %A" dim sa

    let disableAllBc sa : Shape =
        List.map (fun ss -> if ss = Size.Broadcast then Size.Atom (SizeAtom.Fixed Frac.one) else ss) sa
        
    /// True if both shape have the same number of elements and 
    /// are both broadcastable or non-broadcastable in each dimension.
    let equalRespectingBc (sa: Shape) (sb: Shape) =
        List.length sa = List.length sb &&
            List.forall2 Size.equalRespectingBc sa sb

    /// True if both shapes have the same number of elements in each dimension.
    /// Broadcastable and non-broadcastable are treated as equal.            
    let equalIgnoringBc (sa: Shape) (sb: Shape) =
         List.length sa = List.length sb &&
            List.forall2 Size.equalIgnoringBc sa sb

    /// Permutes the axes as specified.
    let permuteAxes (permut: int list) (sa: Shape) : Shape =
        if nDim sa <> List.length permut then
            failwithf "permutation %A must have same rank as shape %A" permut sa
        sa |> List.permute (fun i -> permut.[i])

    /// evaluates shape to numeric shape, if possible
    let tryEval (sa: Shape) : int64 list option =
        let c = List.map (Size.tryEval) sa
        if List.exists (Option.isNone) c then None
        else Some (List.map Option.get c)          

    /// true if evaluation to numeric shape is possible
    let canEval sa =
        match tryEval sa with
        | Some _ -> true
        | None -> false

    /// evaluates shape to numeric shape
    let eval (sa: Shape) : int64 list =
        List.map (Size.eval) sa

    /// substitute the symbols into the Shape and simplifies it
    let substSymbols symVals (sa: Shape) : Shape =
        List.map (Size.substSyms symVals) sa

    type SolutionT = {
        LeftValues:     Map<SizeSym, Size>
        RightValues:    Map<SizeSym, Size>
    }        

    let solve (left: Shape) (right: SizeSym list) =
        if left.Length <> right.Length then failwith "dimension mismatch"
        if right |> Set.ofList |> Set.count <> right.Length then
            failwith "symbols on the right must be unique"
        
        let leftValues = Dictionary<SizeSym, Size>()
        let rightValues = Dictionary<SizeSym, Size>()

        for l, r in List.zip left right do
            match l with
            | Size.Atom (SizeAtom.Fixed _)
            | Size.Broadcast -> 
                rightValues.Add (r, l)
            | Size.Atom (SizeAtom.Sym s) ->
                if leftValues.ContainsKey s then
                    let pv = leftValues.[s]
                    rightValues.Add (r, pv)
                else
                    leftValues.Add (s, Size.Atom (SizeAtom.Sym r))
            | Size.Multinom _ -> failwith "cannot solve with multinoms"
                
        {
            LeftValues = leftValues |> Map.ofDictionary
            RightValues = rightValues |> Map.ofDictionary
        }
                
