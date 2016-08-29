namespace SymTensor


//[<AutoOpen>]
module UExprTypes = 

    /// unified variable specification
    [<StructuredFormatDisplay("\"{Name}\" {Shape}")>]
    type UVarSpecT = {
        Name:      string
        Shape:     ShapeSpecT
        TypeName:  TypeNameT
    }

    // int holds the position of the subuexpr that has the dynamic value
    type UExprRngSpecT = SimpleRangeSpecT<int>
    type UExprRngsSpecT = SimpleRangesSpecT<int>

    type ULeafOpT =
        | Identity of SizeSpecT
        | Zeros of ShapeSpecT                   
        | ScalarConst of System.IComparable
        | SizeValue of SizeSpecT
        | Var of UVarSpecT

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
        | Diag of int * int
        | DiagMat of int * int
        | Invert
        | Sum                           
        | SumAxis of int                
        | Reshape of ShapeSpecT         
        | DoBroadcast of ShapeSpecT       
        | SwapDim of int * int       
        | StoreToVar of UVarSpecT
        | Annotated of string       

    and UBinaryOpT =
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide       
        | Modulo                 
        | Power                         
        | Dot                           
        | TensorProduct                 

    and UNaryOpT =
        | Discard        
        | Subtensor of UExprRngsSpecT 
        | SetSubtensor of UExprRngsSpecT
        | ExtensionOp of IUOp             

    /// unified op of any arity and type
    and UOpT =
        | ULeafOp of ULeafOpT
        | UUnaryOp of UUnaryOpT
        | UBinaryOp of UBinaryOpT
        | UNaryOp of UNaryOpT

    /// metadata for an unified expression
    and UMetadata = {
        /// the data type of the result of the generating expression
        TargetType:     TypeNameT
        /// the symbolic shape of the result of the generating expression
        TargetShape:    ShapeSpecT
        /// the numeric shape of the result of the generating expression
        TargetNShape:   NShapeSpecT
        /// the generating expression, if created from one
        Expr:           System.IComparable option
    }

    /// unified expression (combines all arities and types and ops cannot have expressions as parameters)    
    and [<StructuralComparison; StructuralEquality; StructuredFormatDisplay("{PrettyString}")>]
        UExprT = 
        | UExpr of UOpT * (UExprT list) * UMetadata

        member this.PrettyString =
            match this with
            | UExpr (ULeafOp uop, subs, _) -> sprintf "%A" uop 
            | UExpr (UUnaryOp uop, subs, _) -> sprintf "%A (%A)" uop subs.[0]
            | UExpr (UBinaryOp uop, subs, _) -> sprintf "%A (%A, %A)" uop subs.[0] subs.[1]
            | UExpr (UNaryOp uop, subs, _) -> sprintf "%A (%A)" uop subs

    /// An operation in an unified expression.
    and IUOp =
        inherit System.IComparable


