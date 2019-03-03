namespace Tensor.Expr.Elem

open System

open Tensor.Expr
open DeepNet.Utils


/// argument 
type Arg = Arg of pos:int

/// element of an argument
type ArgElementSpec = Arg * ShapeSpec
    
type LeafOp =
    | Const of Const
    | SizeValue of value:SizeSpec * typ:TypeName
    | ArgElement of argElem:ArgElementSpec * typ:TypeName

and UnaryOp = 
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
    | Sum of SizeSymbol * SizeSpec * SizeSpec
    | KroneckerRng of SizeSpec * SizeSpec * SizeSpec

and BinaryOp = 
    | Add                           
    | Substract                     
    | Multiply                      
    | Divide                        
    | Modulo
    | Power        
    | IfThenElse of SizeSpec * SizeSpec
        
/// an element expression
and [<StructuredFormatDisplay("{Pretty}")>] Expr =        
    | Leaf of LeafOp
    | Unary of UnaryOp * Expr
    | Binary of BinaryOp * Expr * Expr
with

    /// typename of data
    member this.TypeName = 
        match this with
        | Leaf (Const cs) -> cs.TypeName
        | Leaf (SizeValue (_, tn)) -> tn
        | Leaf (ArgElement (_, tn)) -> tn

        | Unary (_, a) -> a.TypeName
        | Binary (_, a, b) ->
            let ta, tb = a.TypeName, b.TypeName
            if ta <> tb then
                failwithf "Elem.Expr must be of one type only but got types %A and %A"
                            ta.Type tb.Type
            ta

    /// type of data
    member this.Type = TypeName.getType this.TypeName

    /// data type name of the element expression
    static member typeName (expr: Expr) = expr.TypeName

    /// checks the elements expression
    static member check (expr: Expr) =
        expr.Type |> ignore
        expr

    /// a constant value given by a Const
    static member constSpec cs =
        Leaf (Const cs) |> Expr.check

    /// a constant value
    static member scalar f =
        Expr.constSpec (Tensor.Expr.Const f)              

    // elementwise unary
    static member (~+) (a: Expr) = a |> Expr.check
    static member (~-) (a: Expr) = Unary(Negate, a) |> Expr.check
    static member Abs (a: Expr) = Unary(Abs, a) |> Expr.check
    static member SignT (a: Expr) = Unary(SignT, a) |> Expr.check
    static member Log (a: Expr) = Unary(Log, a) |> Expr.check
    static member Log10 (a: Expr) = Unary(Log10, a) |> Expr.check
    static member Exp (a: Expr) = Unary(Exp, a) |> Expr.check
    static member Sin (a: Expr) = Unary(Sin, a) |> Expr.check
    static member Cos (a: Expr) = Unary(Cos, a) |> Expr.check
    static member Tan (a: Expr) = Unary(Tan, a) |> Expr.check
    static member Asin (a: Expr) = Unary(Asin, a) |> Expr.check
    static member Acos (a: Expr) = Unary(Acos, a) |> Expr.check
    static member Atan (a: Expr) = Unary(Atan, a) |> Expr.check
    static member Sinh (a: Expr) = Unary(Sinh, a) |> Expr.check
    static member Cosh (a: Expr) = Unary(Cosh, a) |> Expr.check
    static member Tanh (a: Expr) = Unary(Tanh, a) |> Expr.check
    static member Sqrt (a: Expr) = Unary(Sqrt, a) |> Expr.check
    static member Ceiling (a: Expr) = Unary(Ceil, a) |> Expr.check
    static member Floor (a: Expr) = Unary(Floor, a) |> Expr.check
    static member Round (a: Expr) = Unary(Round, a) |> Expr.check
    static member Truncate (a: Expr) = Unary(Truncate, a) |> Expr.check

    // elementwise binary
    static member (+) (a: Expr, b: Expr) = Binary(Add, a, b) |> Expr.check
    static member (-) (a: Expr, b: Expr) = Binary(Substract, a, b) |> Expr.check
    static member (*) (a: Expr, b: Expr) = Binary(Multiply, a, b) |> Expr.check
    static member (/) (a: Expr, b: Expr) = Binary(Divide, a, b) |> Expr.check
    static member (%) (a: Expr, b: Expr) = Binary(Modulo, a, b) |> Expr.check
    static member Pow (a: Expr, b: Expr) = Binary(Power, a, b) |> Expr.check
    static member ( *** ) (a: Expr, b: Expr) = a ** b 

    // elementwise binary with basetype
    static member (+) (a: Expr, b: 'T) = a + (Expr.scalar b) |> Expr.check
    static member (-) (a: Expr, b: 'T) = a - (Expr.scalar b) |> Expr.check
    static member (*) (a: Expr, b: 'T) = a * (Expr.scalar b) |> Expr.check
    static member (/) (a: Expr, b: 'T) = a / (Expr.scalar b) |> Expr.check
    static member (%) (a: Expr, b: 'T) = a % (Expr.scalar b) |> Expr.check
    static member Pow (a: Expr, b: 'T) = a ** (Expr.scalar b) |> Expr.check
    static member ( *** ) (a: Expr, b: 'T) = a ** (Expr.scalar b)

    static member (+) (a: 'T, b: Expr) = (Expr.scalar a) + b |> Expr.check
    static member (-) (a: 'T, b: Expr) = (Expr.scalar a) - b |> Expr.check
    static member (*) (a: 'T, b: Expr) = (Expr.scalar a) * b |> Expr.check
    static member (/) (a: 'T, b: Expr) = (Expr.scalar a) / b |> Expr.check
    static member (%) (a: 'T, b: Expr) = (Expr.scalar a) % b |> Expr.check
    static member Pow (a: 'T, b: Expr) = (Expr.scalar a) ** b |> Expr.check  
    static member ( *** ) (a: 'T, b: Expr) = (Expr.scalar a) ** b          
          
    /// sign keeping type
    static member signt (a: Expr) =
        Expr.SignT a |> Expr.check

    /// square root
    static member sqrtt (a: Expr) =
        Expr.Sqrt a |> Expr.check
                  
    /// index symbol for given dimension of the result
    static member idxSymbol dim =
        sprintf "R%d" dim
        |> SizeSymbol.ofName

    /// index size of given dimension of the result
    static member idx dim =
        SizeSpec.Base (BaseSize.Sym (Expr.idxSymbol dim)) 

    /// summation symbol of given name
    static member sumSymbol name =
        sprintf "SUM_%s" name 
        |> SizeSymbol.ofName

    /// summation index size of given name
    static member sumIdx name =
        SizeSpec.Base (BaseSize.Sym (Expr.sumSymbol name))

    /// sum exprover given index (created by sumIdx) from first to last
    static member sum idx first last expr =
        match idx with
        | SizeSpec.Base (BaseSize.Sym (sumSym)) ->
            Unary (Sum (sumSym, first, last), expr) |> Expr.check
        | _ -> invalidArg "idx" "idx must be summation index obtained by calling sumIdx"

    /// If left=right, then thenExpr else elseExpr.
    static member ifThenElse left right thenExpr elseExpr =
        Binary (IfThenElse (left, right), thenExpr, elseExpr) |> Expr.check

    /// expr if first <= sym <= last, otherwise 0.
    static member kroneckerRng sym first last expr =
        Unary (KroneckerRng (sym, first, last), expr) |> Expr.check

    /// the element with index idx of the n-th argument
    [<RequiresExplicitTypeArguments>]
    static member argElem<'T> pos idx =
        Leaf (ArgElement ((Arg pos, idx), TypeName.ofType<'T>)) |> Expr.check

    /// the element with index idx of the n-th argument of given type
    static member argElemWithType typ pos idx =
        Leaf (ArgElement ((Arg pos, idx), TypeName.ofTypeInst typ)) |> Expr.check

    /// extract ArgElementSpec from element expression
    static member extractArg expr =
        match expr with
        | Leaf (ArgElement (argSpec, _)) -> argSpec
        | _ -> failwith "the provided element expression is not an argument"
   
    /// returns the required number of arguments of the element expression
    static member requiredNumberOfArgs expr =
        match expr with
        | Leaf (ArgElement ((Arg n, _), _)) -> 
            if n < 0 then failwith "argument index must be positive"
            n

        | Leaf _ -> 0
        | Unary (op, a) -> Expr.requiredNumberOfArgs a
        | Binary (op, a, b) -> max (Expr.requiredNumberOfArgs a) (Expr.requiredNumberOfArgs b)
    
    /// checks if the arguments' shapes are compatible with the result shape and that the types match
    static member checkCompatibility (expr: Expr) (argShapes: ShapeSpec list) (argTypes: TypeName list) 
                                        (resShape: ShapeSpec) =

        // check number of arguments
        let nArgs = List.length argShapes
        if argTypes.Length <> nArgs then
            failwith "argShapes and argTypes must be of same length"
        let nReqArgs = Expr.requiredNumberOfArgs expr       
        if nReqArgs > nArgs then
            failwithf "the element expression requires at least %d arguments but only %d arguments were specified"
                nReqArgs nArgs

        // check dimensionality of arguments
        let rec check expr =
            match expr with
            | Leaf (ArgElement ((Arg n, idx), tn)) ->
                if not (0 <= n && n < nArgs) then
                    failwithf "the argument with zero-based index %d used in the element \
                                expression does not exist" n
                let idxDim = ShapeSpec.nDim idx
                let argDim = ShapeSpec.nDim argShapes.[n]
                if idxDim <> argDim then
                    failwithf 
                        "the argument with zero-based index %d has %d dimensions but was used  \
                            with %d dimensions in the element expression" n argDim idxDim
                let argType = argTypes.[n]
                if argType <> tn then
                    failwithf 
                        "the argument with zero-based index %d has type %A but was used  \
                         as type %A in the element expression" n argType.Type tn.Type
            | Leaf _ -> ()

            | Unary (_, a) -> check a
            | Binary (_, a, b) -> check a; check b
        check expr

    /// substitutes the specified size symbols with their replacements 
    static member substSymSizes symSizes expr = 
        let sSub = Expr.substSymSizes symSizes
        let sSize = SymSizeEnv.subst symSizes
        let sShp = SymSizeEnv.substShape symSizes

        match expr with
        | Leaf (SizeValue (sc, tn)) -> Leaf (SizeValue ((sSize sc), tn))
        | Leaf (ArgElement ((arg, argIdxs), tn)) -> Leaf (ArgElement ((arg, sShp argIdxs), tn))
        | Leaf _ -> expr

        | Unary (Sum (sym, first, last), a) -> 
            Unary (Sum (sym, sSize first, sSize last), sSub a)
        | Unary (KroneckerRng (sym, first, last), a) ->
            Unary (KroneckerRng (sSize sym, sSize first, sSize last), sSub a)
        | Unary (op, a) -> Unary (op, sSub a)
        | Binary (IfThenElse (left, right), a, b) ->
            Binary (IfThenElse (sSize left, sSize right), sSub a, sSub b)
        | Binary (op, a, b) -> Binary (op, sSub a, sSub b)

    /// true if all size symbols can be evaluated to numeric values 
    static member canEvalAllSymSizes expr =
        let rec canEval expr =  
            match expr with
            | Leaf (SizeValue (sc, _)) -> SizeSpec.canEval sc
            | Leaf (ArgElement ((arg, argIdxs), _)) -> ShapeSpec.canEval argIdxs
            | Leaf _ -> true

            | Unary (Sum (sym, first, last), a) -> 
                // replace sum index by one before testing for evaluability
                let sumSymVals = Map [sym, SizeSpec.one]
                SizeSpec.canEval first && 
                SizeSpec.canEval last && 
                canEval (a |> Expr.substSymSizes sumSymVals)
            | Unary (KroneckerRng (sym, first, last), a) ->
                SizeSpec.canEval sym && 
                SizeSpec.canEval first && 
                SizeSpec.canEval last &&
                canEval a
            | Unary (op, a) -> canEval a
            | Binary (IfThenElse (left, right), a, b) ->
                SizeSpec.canEval left && 
                SizeSpec.canEval right &&
                canEval a && canEval b
            | Binary (op, a, b) -> canEval a && canEval b

        // replace output indices by ones before testing for evaluability
        let dummySymVals =
            seq {for dim=0 to 20 do yield (Expr.idxSymbol dim, SizeSpec.one)}
            |> Map.ofSeq
        expr |> Expr.substSymSizes dummySymVals |> canEval

    /// pretty string of an element expression
    member this.Pretty =
        // TODO: remove unnecessary brackets
        match this with
        | Leaf (op) -> 
            match op with
            | Const v -> sprintf "%A" v
            | SizeValue (ss, _) -> sprintf "%A" ss
            | ArgElement ((Arg a, idxs), _) -> sprintf "a%d%A" a idxs
        
        | Unary (op, a) ->
            match op with
            | Negate -> sprintf "-(%s)" (a.Pretty)
            | Abs -> sprintf "abs(%s)" (a.Pretty)
            | SignT -> sprintf "signt(%s)" (a.Pretty) 
            | Log -> sprintf "log(%s)" (a.Pretty)
            | Log10 -> sprintf "log10(%s)" (a.Pretty)
            | Exp -> sprintf "exp(%s)" (a.Pretty)
            | Sin -> sprintf "sin(%s)" (a.Pretty)
            | Cos -> sprintf "cos(%s)" (a.Pretty)
            | Tan -> sprintf "tan(%s)" (a.Pretty)
            | Asin -> sprintf "asin(%s)" (a.Pretty)
            | Acos -> sprintf "acos(%s)" (a.Pretty)
            | Atan -> sprintf "atan(%s)" (a.Pretty)
            | Sinh -> sprintf "sinh(%s)" (a.Pretty)
            | Cosh -> sprintf "cosh(%s)" (a.Pretty)
            | Tanh -> sprintf "tanh(%s)" (a.Pretty)
            | Sqrt -> sprintf "sqrt(%s)" (a.Pretty)
            | Ceil -> sprintf "ceil(%s)" (a.Pretty) 
            | Floor -> sprintf "floor(%s)" (a.Pretty) 
            | Round -> sprintf "round(%s)" (a.Pretty)
            | Truncate -> sprintf "truncate(%s)" (a.Pretty)
            | Sum (sumSym, first, last)-> 
                sprintf "sum(%A[%A..%A], %s)" sumSym first last (a.Pretty)
            | KroneckerRng (s, first, last) ->
                sprintf "%s[%A%s%A %s %A%s%A](%s)"  (String('\u03B4',1)) 
                                                    first
                                                    (String('\u2264',1))
                                                    s
                                                    (String('\u2227',1))
                                                    s
                                                    (String('\u2264',1))
                                                    last
                                                    (a.Pretty)
        | Binary(op, a, b) -> 
            match op with
            | Add -> sprintf "(%s + %s)" (a.Pretty) (b.Pretty)
            | Substract -> sprintf "(%s - %s)" (a.Pretty) (b.Pretty)
            | Multiply -> sprintf "(%s * %s)" (a.Pretty) (b.Pretty)
            | Divide -> sprintf "(%s / %s)" (a.Pretty) (b.Pretty)
            | Modulo -> sprintf "(%s %% %s)" (a.Pretty) (b.Pretty)
            | Power -> sprintf "(%s ** %s)" (a.Pretty) (b.Pretty)
            | IfThenElse (left,right)-> 
                sprintf "ifThenElse(%A = %A, %s, %s)" left right (a.Pretty) (b.Pretty)

module Expr =

    /// tuple of 1 index symbol
    let idx1 = Expr.idx 0
    /// tuple of 2 index symbols
    let idx2 = Expr.idx 0, Expr.idx 1
    /// tuple of 3 index symbols
    let idx3 = Expr.idx 0, Expr.idx 1, Expr.idx 2
    /// tuple of 4 index symbols
    let idx4 = Expr.idx 0, Expr.idx 1, Expr.idx 2, Expr.idx 3
    /// tuple of 5 index symbols
    let idx5 = Expr.idx 0, Expr.idx 1, Expr.idx 2, Expr.idx 3, Expr.idx 4
    /// tuple of 6 index symbols
    let idx6 = Expr.idx 0, Expr.idx 1, Expr.idx 2, Expr.idx 3, Expr.idx 4, Expr.idx 5
    /// tuple of 7 index symbols
    let idx7 = Expr.idx 0, Expr.idx 1, Expr.idx 2, Expr.idx 3, Expr.idx 4, Expr.idx 5, Expr.idx 6

    /// tuple of 1 argument
    [<RequiresExplicitTypeArguments>]
    let arg1<'T> = Expr.argElem<'T> 0
    /// tuple of 2 arguments
    [<RequiresExplicitTypeArguments>]
    let arg2<'T> = Expr.argElem<'T> 0, Expr.argElem<'T> 1
    /// tuple of 3 arguments
    [<RequiresExplicitTypeArguments>]
    let arg3<'T> = Expr.argElem<'T> 0, Expr.argElem<'T> 1, Expr.argElem<'T> 2
    /// tuple of 4 arguments
    [<RequiresExplicitTypeArguments>]
    let arg4<'T> = Expr.argElem<'T> 0, Expr.argElem<'T> 1, Expr.argElem<'T> 2, Expr.argElem<'T> 3
    /// tuple of 5 arguments
    [<RequiresExplicitTypeArguments>]
    let arg5<'T> = Expr.argElem<'T> 0, Expr.argElem<'T> 1, Expr.argElem<'T> 2, Expr.argElem<'T> 3, Expr.argElem<'T> 4
    /// tuple of 6 arguments
    [<RequiresExplicitTypeArguments>]
    let arg6<'T> = Expr.argElem<'T> 0, Expr.argElem<'T> 1, Expr.argElem<'T> 2, Expr.argElem<'T> 3, Expr.argElem<'T> 4, Expr.argElem<'T> 5
    /// tuple of 7 arguments
    [<RequiresExplicitTypeArguments>]
    let arg7<'T> = Expr.argElem<'T> 0, Expr.argElem<'T> 1, Expr.argElem<'T> 2, Expr.argElem<'T> 3, Expr.argElem<'T> 4, Expr.argElem<'T> 5, Expr.argElem<'T> 6

