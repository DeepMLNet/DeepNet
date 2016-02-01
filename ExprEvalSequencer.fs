module ExprEvalSequencer

open Op


type StorageT = string

type ExeOpT =
    | LeafExe of StorageT * LeafOpT
    | UnaryExe of StorageT * UnaryOpT * StorageT
    | BinaryExe of StorageT * BinaryOpT * StorageT * StorageT


[<StructuredFormatDisplay("{AsString}")>]
type ExeSequenceItemT =
    | ExeSequenceItem of ExeOpT * ExprT option

    member this.AsString =
        match this with
        | ExeSequenceItem (eop, _) -> sprintf "%A" eop

type ExeSequenceT = ExeSequenceItemT list


module ExeOp =
    /// returns the target storage of the ExeOp
    let target eop =
        match eop with
        | LeafExe(t, _) | UnaryExe(t, _, _) | BinaryExe(t, _, _, _) -> t


module ExeSequence =
    let empty : ExeSequenceT = []

    /// finds the ExeSequenceItem that computes expr
    let tryFindExpr expr eseq =
        eseq |> List.tryFind 
            (fun esi -> match esi with
                        | ExeSequenceItem(_, Some e) when expr = e -> true
                        | _ -> false)


let buildSequence (expr: ExprT) =
    let mutable storageCount = 0
    let newStorage () : StorageT =
        storageCount <- storageCount + 1
        sprintf "s%d" storageCount

    let rec buildSubsequence (eseq: ExeSequenceT) (expr: ExprT) =   
        // see if expr has already been evaluated and stored somewhere
        match ExeSequence.tryFindExpr expr eseq with
        | Some (ExeSequenceItem(exeOp, _)) ->
            // Expr is already evaluated and stored. No commands need to be emitted.
            let eStorage = ExeOp.target exeOp
            eStorage, eseq
        | None ->
            // Expr needs to be evaluated and stored.
            let eStorage = newStorage ()

            let appendExe exeOp eseq =
                eseq @ [ExeSequenceItem(exeOp, Some expr)]

            let eseq = 
                match expr with
                | Leaf(op) -> 
                    eseq |> appendExe (LeafExe(eStorage, op))
                | Unary(op, a) -> 
                    let aStorage, eseq = buildSubsequence eseq a
                    eseq |> appendExe (UnaryExe(eStorage, op, aStorage))
                | Binary(op, a, b) ->
                    let aStorage, eseq = buildSubsequence eseq a
                    let bStorage, eseq = buildSubsequence eseq b
                    eseq |> appendExe (BinaryExe(eStorage, op, aStorage, bStorage))
            eStorage, eseq

    let resStorage, eseq = buildSubsequence ExeSequence.empty expr
    resStorage, eseq


