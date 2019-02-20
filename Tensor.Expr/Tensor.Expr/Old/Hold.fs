namespace Tensor.Expr

open DeepNet.Utils


/// functions for working with held op
module Hold =

    /// Cache of released expressions.
    let private released = Dictionary<Expr, Expr> (HashIdentity.Reference)
 
    /// tries to release all ops that are "held" (via UnaryOp.Held) in the expression
    let rec tryRelease (expr: Expr) : Expr =
        match released.LockedTryFind expr with
        | Some rel -> rel
        | None ->
            let rel =
                match expr with
                | Unary (Held ([], heldOp) as op, a) ->
                    let a = tryRelease a
                    match heldOp with
                    // rewrite ReplicateTo into replicate and slicing once the necessary
                    // sizes can be evaluated
                    | ReplicateTo (dim, size) when SizeSpec.canEval size && SizeSpec.canEval a.Shape.[dim] ->                 
                        let nTrgt, nSrc = SizeSpec.eval size, SizeSpec.eval a.Shape.[dim]
                        let nReps, haveRemaining = 
                            if nTrgt % nSrc = 0L then nTrgt / nSrc, false
                            else nTrgt / nSrc + 1L, true
                        let aRep = a |> Expr.replicate dim (SizeSpec.fix nReps)
                        if haveRemaining then
                            let slice : ExprRngsSpec = 
                                [0 .. aRep.NDims-1]
                                |> List.map (fun d -> 
                                    if d = dim then SimpleRangeSpec.SymStartSymEnd (SizeSpec.zero, Some (size - 1L))
                                    else SimpleRangeSpec.All)
                            aRep.[slice]
                        else aRep

                    | _ -> 
                        Unary (op, a)

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
                        dWrt |> tryRelease

                // pass-throguh
                | Leaf _ -> expr

                | Unary (Gather indices, a) ->
                    let indices = indices |> List.map (Option.map tryRelease)
                    Unary (Gather indices, tryRelease a)
                | Unary (Scatter (indices, shp), a) ->
                    let indices = indices |> List.map (Option.map tryRelease)
                    Unary (Scatter (indices, shp), tryRelease a)
                | Unary (AssumeJacobian jac, a) -> 
                    Unary (AssumeJacobian (tryRelease jac), tryRelease a)
                | Unary (op, a) -> Unary (op, tryRelease a)

                | Binary (IfThenElse c, a, b) -> Binary (IfThenElse (tryRelease c), tryRelease a, tryRelease b)
                | Binary (op, a, b) -> Binary (op, tryRelease a, tryRelease b)

                | Nary (Channel (Loop spec, channel), es) ->
                    let spec = 
                        {spec with Channels = spec.Channels 
                                              |> Map.map (fun _ lv -> {lv with Expr=tryRelease lv.Expr})}
                    Nary (NaryOp.Channel (Loop spec, channel), es |> List.map tryRelease)
                | Nary (op, es) -> Nary (op, es |> List.map tryRelease)
            
            released.[expr] <- rel
            rel


