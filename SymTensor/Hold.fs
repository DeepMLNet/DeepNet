namespace SymTensor

open Expr


/// functions for working with held op
module Hold =
 
    /// tries to release all ops that are "held" (via UnaryOp.Held) in the expression
    let rec tryRelease (expr: ExprT) : ExprT =
        match expr with

        | Unary (Held ([], heldOp) as op, a) ->
            let a = tryRelease a
            match heldOp with
            // rewrite ReplicateTo into replicate and slicing once the necessary
            // sizes can be evaluated
            | ReplicateTo (dim, size) when SizeSpec.canEval size && SizeSpec.canEval a.Shape.[dim] ->                 
                let nTrgt, nSrc = SizeSpec.eval size, SizeSpec.eval a.Shape.[dim]
                let nReps, haveRemaining = 
                    if nTrgt % nSrc = 0 then nTrgt / nSrc, false
                    else nTrgt / nSrc + 1, true
                let aRep = a |> replicate dim (SizeSpec.fix nReps)
                if haveRemaining then
                    let slice : ExprRngsSpecT = 
                        [0 .. aRep.NDims-1]
                        |> List.map (fun d -> 
                            if d = dim then SRSSymStartSymEnd (SizeSpec.zero, Some (size - 1))
                            else SRSAll)
                    aRep.[slice]
                else aRep

            | _ -> Unary (op, a)

        // derivatives
        | Unary (Held (wrtShp :: wrtShps, heldOp) as op, eg) ->
            let eg = tryRelease eg
            // try to release contained expression            
            let wrt = Expr.varOfType "HeldWRT" eg.Type wrtShp
            match tryRelease (Unary (Held (wrtShps, heldOp), wrt)) with
            | Unary (Held _, _) ->
                // release was not possible, keep held derivative
                Unary (op, eg)
            | relExpr ->
                // release occured, calculate derivative of released expression
                let dWrt = Deriv.computeWithRootJacobian eg relExpr |> Deriv.ofVar wrt
                if dWrt |> Expr.contains wrt then
                    failwithf "held op %A must not contain their input value in their derivative" heldOp
                dWrt

        | Leaf _ -> expr
        | Unary (op, a) -> Unary (op, tryRelease a)
        | Binary (op, a, b) -> Binary (op, tryRelease a, tryRelease b)
        | Nary (op, es) -> Nary (op, es |> List.map tryRelease)


