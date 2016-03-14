namespace SymTensor

open Basics
open Expr


module Optimizer =
    
    let rec optimize expr =
        match expr with
        | Leaf _ -> expr

        // combine subsequent reshapes
        | Unary (Reshape ss, Unary (Reshape _, a)) ->
            optimize (Unary(Reshape ss, a))

        // remove unnecessary reshapes
        | Unary (Reshape ss, a) when ShapeSpec.equalWithBroadcastability ss (shapeOf a) ->
            optimize a            

        | Unary(op, a) -> Unary(op, optimize a)            
        | Binary(op, a, b) -> Binary(op, optimize a, optimize b)
        | Nary(op, es) -> Nary(op, List.map optimize es)



