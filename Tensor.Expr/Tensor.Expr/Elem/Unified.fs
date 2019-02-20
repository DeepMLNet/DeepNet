namespace SymTensor.Elem

open SymTensor
open DeepNet.Utils


/// unified element expression
module Unified = 

    /// unified element expression op
    type UOp =
        | ULeafOp of LeafOp
        | UUnaryOp of UnaryOp
        | UBinaryOp of BinaryOp
    
    /// unified element expression
    type UExpr = {
        Op:         UOp
        Args:       UExpr list
        Type:       TypeName
    }

    /// element function
    type UFunc = {
        /// element expression
        Expr:       UExpr
        /// number of dimensions of the result
        NDims:      int
        /// number of input arguments
        NArgs:      int
    }

    /// converts an element expression to a unified element expression
    let toUExpr (elemExpr: Expr) =
        let cache = Dictionary<Expr, UExpr> ()
        let rec build elemExpr =
            match cache.TryFind elemExpr with
            | Some uElemExpr -> uElemExpr
            | None ->
                let uElemExpr =
                    let tn = Expr.typeName elemExpr
                    match elemExpr with
                    | Leaf op -> { Op=ULeafOp op; Args=[]; Type=tn }
                    | Unary (op, a) -> { Op=UUnaryOp op; Args=[build a]; Type=tn }
                    | Binary (op, a, b) -> { Op=UBinaryOp op; Args=[build a; build b]; Type=tn }
                cache.[elemExpr] <- uElemExpr
                uElemExpr
        build elemExpr

    /// converts an element expression to a unified element function
    let toUFunc elemExpr nDims nArgs =
        {
            Expr    = toUExpr elemExpr
            NDims   = nDims
            NArgs   = nArgs
        }
