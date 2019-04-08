namespace Tensor.Expr.Opt

open DeepNet.Utils
open Tensor.Expr
//open Tensor.Expr.Ops
open Tensor.Expr.Opt.Tools



/// Optimizes shape operations.
[<Optimizer>]
type ShapeOptimizer() =

    interface IOptimizer with
        member __.Order = 10

    interface IUExprOptimizer with
        member __.Optimize opt expr =
            match expr with
            // remove unnecessary axes permutations
            | UExpr.PermuteAxes (perm, x) when Permutation.isIdentity perm -> x

            // remove unnecessary reshapes
            | UExpr.Reshape (ss, x) when Shape.equalRespectingBc ss x.Shape -> x

            // remove unnecessary broadcasts
            | UExpr.DoBroadcast (ss, x) when Shape.equalRespectingBc ss x.Shape -> x

            // combine subsequent axes permutations
            | UExpr.PermuteAxes (perm1, UExpr.PermuteAxes (perm2, x)) ->
                UExpr.permuteAxes (Permutation.chain perm1 perm2) x

            // remove unneccessary permutation of size-one axes before reshape
            | UExpr.Reshape (ss, UExpr.PermuteAxes (Permutation.Swap (ax1, ax2), x)) when
                    (Size.equalIgnoringBc x.Shape.[ax1] Size.one || 
                     Size.equalIgnoringBc x.Shape.[ax2] Size.one) &&
                    x.Shape.[ax1+1 .. ax2-1] |> List.forall (Size.equalIgnoringBc Size.one) ->
                UExpr.reshape ss x

            // combine subsequent reshapes
            | UExpr.Reshape (ss, UExpr.Reshape (_, x)) ->
                UExpr.reshape ss x

            // combine subsequent broadcasts
            | UExpr.DoBroadcast (bc, UExpr.DoBroadcast (_, x)) ->
                UExpr.broadcast bc x

            // remove unnecessary broadcasts after reshape
            | UExpr.DoBroadcast (bcShp, UExpr.Reshape (reShp, x)) when 
                    Shape.equalIgnoringBc bcShp reShp ->
                UExpr.reshape bcShp x

            // pull permute through broadcast
            | UExpr.DoBroadcast (bc, UExpr.PermuteAxes (perm, x)) ->
                let bcPerm = bc |> Permutation.apply (Permutation.invert perm)
                UExpr.permuteAxes perm (UExpr.broadcast bcPerm x)

            // pull permute through unary elementwise ops
            | UExpr.UnaryElemwise (op, UExpr.PermuteAxes (perm, x)) ->
                UExpr.permuteAxes perm (replaceUnaryArg op x)

            // pull broadcast through unary elementwise ops
            | UExpr.UnaryElemwise (op, UExpr.Reshape (ss, x)) ->
                UExpr.reshape ss (replaceUnaryArg op x)

            // pull reshape through unary elementwise ops
            | UExpr.UnaryElemwise (op, UExpr.DoBroadcast (bc, x)) ->
                UExpr.broadcast bc (replaceUnaryArg op x)

            // pull broadcast over batched dimensions through Diag
            | UExpr.Diag (ax1, ax2, (UExpr.DoBroadcast (_bc, a) as ba))
                        when List.indexed (axesBroadcasted ba)
                             |> List.exists (fun (d, bc) -> d <> ax1 && d <> ax2 && bc.IsBC) ->
                let aOptBc =
                    List.indexed (axesBroadcasted ba)   
                    |> List.map (function | d, bc when d = ax1 || d = ax2 -> bc.Size
                                            | _, Bc _ -> Size.broadcastable
                                            | _, NotBc s -> s)
                let baOpt = UExpr.broadcast aOptBc a
                UExpr.broadcast expr.Shape (UExpr.diagAxis ax1 ax2 baOpt)

            // pull broadcast over batched dimensions through DiagMat 
            | UExpr.DiagMat (ax1, ax2, (UExpr.DoBroadcast (_bc, a) as ba))
                        when List.indexed (axesBroadcasted ba)
                             |> List.exists (fun (d, bc) -> d <> ax1 && bc.IsBC) ->
                let aOptBc =
                    List.indexed (axesBroadcasted ba)   
                    |> List.map (function | d, bc when d = ax1 -> bc.Size
                                            | _, Bc _ -> Size.broadcastable
                                            | _, NotBc s -> s)
                let baOpt = UExpr.broadcast aOptBc a
                UExpr.broadcast expr.Shape (UExpr.diagMatAxis ax1 ax2 baOpt)

            // pull matching permute through binary elementwise ops
            | UExpr.BinaryElemwise (op, UExpr.PermuteAxes (permA, a), UExpr.PermuteAxes (permB, b)) 
                    when permA = permB && a.Shape = b.Shape ->
                UExpr.permuteAxes permA (replaceBinaryArgs op a b)

            // pull matching reshape through binary elementwise ops
            | UExpr.BinaryElemwise (op, UExpr.Reshape (shpA, a), UExpr.Reshape (shpB, b)) 
                    when shpA = shpB && a.Shape = b.Shape ->
                UExpr.reshape shpA (replaceBinaryArgs op a b)

            // pull matching broadcast through binary elementwise ops
            | UExpr.BinaryElemwise (op, UExpr.DoBroadcast (bcA, a), UExpr.DoBroadcast (bcB, b))
                    when bcA = bcB && a.Shape = b.Shape ->
                UExpr.broadcast bcA (replaceBinaryArgs op a b)

            // pull matching broadcasts over batched dimensions through dot op
            | UExpr.Dot ((UExpr.DoBroadcast (_bcA, a) as ba), (UExpr.DoBroadcast (_bcB, b) as bb))
                    when List.zip (axesBroadcasted ba) (axesBroadcasted bb)
                            |> List.indexed
                            |> List.exists (fun (d, (aBc, bBc)) -> d < ba.NDims - 2 && aBc.IsBC 
                                                                                    && bBc.IsBC) ->
                let aOptBc, bOptBc =
                    List.zip (axesBroadcasted ba) (axesBroadcasted bb)
                    |> List.indexed
                    |> List.map (function | d, (aBc, bBc) when d >= ba.NDims-2 -> aBc.Size, bBc.Size
                                          | _, (Bc _, Bc _) -> Size.broadcastable, Size.broadcastable
                                          | _, (aBc, bBc) -> aBc.Size, bBc.Size)
                    |> List.unzip
                let baOpt = UExpr.broadcast aOptBc a
                let bbOpt = UExpr.broadcast bOptBc b
                UExpr.broadcast expr.Shape (baOpt .* bbOpt)

            | _ -> expr
            

            