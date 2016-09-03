namespace SymTensor

open Basics
open ArrayNDNS
open ShapeSpec
open VarSpec
open ElemExpr


module UElemExpr = 

    type ArgT = ElemExpr.ArgT
    type ArgElementSpecT = ElemExpr.ArgElementSpecT

    type ULeafOpT =
        | Const of System.IComparable
        | SizeValue of SizeSpecT
        | ArgElement of ArgElementSpecT

    and UUnaryOpT = 
        | Negate                        
        | Abs
        | SignT
        | Log
        | Log10                           
        | Exp                           
        | Sin
        | Cos
        | Tan
        | Asin
        | Acos
        | Atan
        | Sinh
        | Cosh
        | Tanh
        | Sqrt
        | Ceil
        | Floor
        | Round
        | Truncate
        | Sum of SizeSymbolT * SizeSpecT * SizeSpecT
        | KroneckerRng of SizeSpecT * SizeSpecT * SizeSpecT

    and UBinaryOpT = 
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power        
        | IfThenElse of SizeSpecT * SizeSpecT

        
    and UOpT =
        | ULeafOp of ULeafOpT
        | UUnaryOp of UUnaryOpT
        | UBinaryOp of UBinaryOpT
    
    and UElemExprT =
        | UElemExpr of UOpT * (UElemExprT list) * TypeNameT


    /// element function
    type UElemFuncT = {
        /// element expression
        Expr:       UElemExprT
        /// number of dimensions of the result
        NDims:      int
        /// number of input arguments
        NArgs:      int
    }

    /// converts an element expression to a unified element expression
    let rec toUElemExpr (elemExpr: ElemExprT<'T>) =
        let tn = TypeName.ofType<'T>
        let leaf uop        = UElemExpr (ULeafOp uop, [], tn)
        let unary uop a     = UElemExpr (UUnaryOp uop, [toUElemExpr a], tn)
        let binary uop a b  = UElemExpr (UBinaryOp uop, [toUElemExpr a; toUElemExpr b], tn)

        match elemExpr with
        | Leaf (ElemExpr.Const v)           -> leaf (Const (box v :?> System.IComparable))
        | Leaf (ElemExpr.SizeValue sv)      -> leaf (SizeValue sv)
        | Leaf (ElemExpr.ArgElement ae)     -> leaf (ArgElement ae)

        | Unary (ElemExpr.Negate, a)        -> unary Negate a
        | Unary (ElemExpr.Abs, a)           -> unary Abs a
        | Unary (ElemExpr.SignT, a)         -> unary SignT a
        | Unary (ElemExpr.Log, a)           -> unary Log a
        | Unary (ElemExpr.Log10, a)         -> unary Log10 a
        | Unary (ElemExpr.Exp, a)           -> unary Exp a
        | Unary (ElemExpr.Sin, a)           -> unary Sin a
        | Unary (ElemExpr.Cos, a)           -> unary Cos a
        | Unary (ElemExpr.Tan, a)           -> unary Tan a
        | Unary (ElemExpr.Asin, a)          -> unary Asin a
        | Unary (ElemExpr.Acos, a)          -> unary Acos a
        | Unary (ElemExpr.Atan, a)          -> unary Atan a
        | Unary (ElemExpr.Sinh, a)          -> unary Sinh a
        | Unary (ElemExpr.Cosh, a)          -> unary Cosh a
        | Unary (ElemExpr.Tanh, a)          -> unary Tanh a
        | Unary (ElemExpr.Sqrt, a)          -> unary Sqrt a
        | Unary (ElemExpr.Ceil, a)          -> unary Ceil a
        | Unary (ElemExpr.Floor, a)         -> unary Floor a
        | Unary (ElemExpr.Round, a)         -> unary Round a
        | Unary (ElemExpr.Truncate, a)      -> unary Truncate a
        | Unary (ElemExpr.Sum (sym, first, last), a) -> unary (Sum (sym, first, last)) a
        | Unary (ElemExpr.KroneckerRng (s, first, last), a) -> unary (KroneckerRng (s, first, last)) a

        | Binary (ElemExpr.Add, a, b)       -> binary Add a b
        | Binary (ElemExpr.Substract, a, b) -> binary Substract a b
        | Binary (ElemExpr.Multiply, a, b)  -> binary Multiply a b                     
        | Binary (ElemExpr.Divide, a, b)    -> binary Divide a b             
        | Binary (ElemExpr.Modulo, a, b)    -> binary Modulo a b          
        | Binary (ElemExpr.Power, a, b)     -> binary Power a b  
        | Binary (ElemExpr.IfThenElse (left, right), a, b) -> binary (IfThenElse (left, right)) a b
            

    /// converts an element expression to a unified element function
    let toUElemFunc elemExpr nDims nArgs =
        {
            Expr    = toUElemExpr elemExpr
            NDims   = nDims
            NArgs   = nArgs
        }