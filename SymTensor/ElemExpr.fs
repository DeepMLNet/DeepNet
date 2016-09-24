namespace SymTensor

open Basics
open ArrayNDNS
open ShapeSpec
open VarSpec
open System

module ElemExpr =

    /// argument 
    type ArgT = Arg of int

    /// element of an argument
    type ArgElementSpecT = ArgT * ShapeSpecT
    
    type LeafOpT =
        | Const of ConstSpecT
        | SizeValue of SizeSpecT
        | ArgElement of ArgElementSpecT

    and UnaryOpT = 
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

    and BinaryOpT = 
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power        
        | IfThenElse of SizeSpecT * SizeSpecT
        
    /// an element expression
    and [<StructuredFormatDisplay("{PrettyString}")>]
        ElemExprT =
        | Leaf of LeafOpT
        | Unary of UnaryOpT * ElemExprT
        | Binary of BinaryOpT * ElemExprT * ElemExprT
        
    /// a constant value
    let scalar f =
        Leaf (Const (ConstSpec.ofValue f))
              
    type ElemExprT with

        // elementwise unary
        static member (~+) (a: ElemExprT) = a 
        static member (~-) (a: ElemExprT) = Unary(Negate, a)  
        static member Abs (a: ElemExprT) = Unary(Abs, a) 
        static member SignT (a: ElemExprT) = Unary(SignT, a) 
        static member Log (a: ElemExprT) = Unary(Log, a) 
        static member Log10 (a: ElemExprT) = Unary(Log10, a) 
        static member Exp (a: ElemExprT) = Unary(Exp, a) 
        static member Sin (a: ElemExprT) = Unary(Sin, a) 
        static member Cos (a: ElemExprT) = Unary(Cos, a) 
        static member Tan (a: ElemExprT) = Unary(Tan, a) 
        static member Asin (a: ElemExprT) = Unary(Asin, a) 
        static member Acos (a: ElemExprT) = Unary(Acos, a) 
        static member Atan (a: ElemExprT) = Unary(Atan, a) 
        static member Sinh (a: ElemExprT) = Unary(Sinh, a) 
        static member Cosh (a: ElemExprT) = Unary(Cosh, a) 
        static member Tanh (a: ElemExprT) = Unary(Tanh, a) 
        static member Sqrt (a: ElemExprT) = Unary(Sqrt, a) 
        static member Ceiling (a: ElemExprT) = Unary(Ceil, a) 
        static member Floor (a: ElemExprT) = Unary(Floor, a) 
        static member Round (a: ElemExprT) = Unary(Round, a) 
        static member Truncate (a: ElemExprT) = Unary(Truncate, a) 

        // elementwise binary
        static member (+) (a: ElemExprT, b: ElemExprT) = Binary(Add, a, b)
        static member (-) (a: ElemExprT, b: ElemExprT) = Binary(Substract, a, b)
        static member (*) (a: ElemExprT, b: ElemExprT) = Binary(Multiply, a, b)
        static member (/) (a: ElemExprT, b: ElemExprT) = Binary(Divide, a, b)
        static member (%) (a: ElemExprT, b: ElemExprT) = Binary(Modulo, a, b)
        static member Pow (a: ElemExprT, b: ElemExprT) = Binary(Power, a, b)
        static member ( *** ) (a: ElemExprT, b: ElemExprT) = a ** b

        // elementwise binary with basetype
        static member (+) (a: ElemExprT, b: 'T) = a + (scalar b)
        static member (-) (a: ElemExprT, b: 'T) = a - (scalar b)
        static member (*) (a: ElemExprT, b: 'T) = a * (scalar b)
        static member (/) (a: ElemExprT, b: 'T) = a / (scalar b)
        static member (%) (a: ElemExprT, b: 'T) = a % (scalar b)
        static member Pow (a: ElemExprT, b: 'T) = a ** (scalar b)
        static member ( *** ) (a: ElemExprT, b: 'T) = a ** (scalar b)

        static member (+) (a: 'T, b: ElemExprT) = (scalar a) + b
        static member (-) (a: 'T, b: ElemExprT) = (scalar a) - b
        static member (*) (a: 'T, b: ElemExprT) = (scalar a) * b
        static member (/) (a: 'T, b: ElemExprT) = (scalar a) / b
        static member (%) (a: 'T, b: ElemExprT) = (scalar a) % b
        static member Pow (a: 'T, b: ElemExprT) = (scalar a) ** b          
        static member ( *** ) (a: 'T, b: ElemExprT) = (scalar a) ** b          
          
    /// sign keeping type
    let signt (a: ElemExprT) =
        ElemExprT.SignT a 

    /// square root
    let sqrtt (a: ElemExprT) =
        ElemExprT.Sqrt a       
        
    /// scalar 0 of appropriate type
    let zero = scalar 0

    /// scalar 1 of appropriate type
    let one = scalar 1

    /// scalar 2 of appropriate type
    let two = scalar 2
           
    /// index symbol for given dimension of the result
    let idxSymbol dim =
        sprintf "R%d" dim
        |> SizeSymbol.ofName

    /// index size of given dimension of the result
    let idx dim =
        Base (Sym (idxSymbol dim))

    /// summation symbol of given name
    let sumSymbol name =
        sprintf "SUM_%s" name 
        |> SizeSymbol.ofName

    /// summation index size of given name
    let sumIdx name =
        Base (Sym (sumSymbol name))

    /// sum exprover given index (created by sumIdx) from first to last
    let sum idx first last expr =
        match idx with
        | Base (Sym (sumSym)) ->
            Unary (Sum (sumSym, first, last), expr) 
        | _ -> invalidArg "idx" "idx must be summation index obtained by calling sumIdx"

    /// If left=right, then thenExpr else elseExpr.
    let ifThenElse left right thenExpr elseExpr =
        Binary (IfThenElse (left, right), thenExpr, elseExpr)

    /// expr if first <= sym <= last, otherwise 0.
    let kroneckerRng sym first last expr =
        Unary (KroneckerRng (sym, first, last), expr)

    /// the element with index idx of the n-th argument
    let argElem pos idx =
        Leaf (ArgElement (Arg pos, idx))

    type Argument1D (pos: int) =       
        member this.Item with get (i0) : ElemExprT = argElem pos [i0]
    type Argument2D (pos: int) =       
        member this.Item with get (i0, i1) : ElemExprT = argElem pos [i0; i1]
    type Argument3D (pos: int) =       
        member this.Item with get (i0, i1, i2) : ElemExprT = argElem pos [i0; i1; i2]
    type Argument4D (pos: int) =       
        member this.Item with get (i0, i1, i2, i3) : ElemExprT = argElem pos [i0; i1; i2; i3]
    type Argument5D (pos: int) =       
        member this.Item with get (i0, i1, i2, i3, i4) : ElemExprT = argElem pos [i0; i1; i2; i3; i4]
    type Argument6D (pos: int) =       
        member this.Item with get (i0, i1, i2, i3, i4, i5) : ElemExprT = argElem pos [i0; i1; i2; i3; i4; i5]

    /// scalar argument at given position
    let arg0D pos = argElem pos []
    /// 1-dimensional argument at given position
    let arg1D pos = Argument1D pos
    /// 2-dimensional argument at given position
    let arg2D pos = Argument2D pos
    /// 3-dimensional argument at given position
    let arg3D pos = Argument3D pos
    /// 4-dimensional argument at given position
    let arg4D pos = Argument4D pos
    /// 5-dimensional argument at given position
    let arg5D pos = Argument5D pos
    /// 6-dimensional argument at given position
    let arg6D pos = Argument6D pos

    /// extract ArgElementSpec from element expression
    let extractArg expr =
        match expr with
        | Leaf (ArgElement argSpec) -> argSpec
        | _ -> failwith "the provided element expression is not an argument"
   
    /// returns the required number of arguments of the element expression
    let rec requiredNumberOfArgs expr =
        match expr with
        | Leaf (ArgElement (Arg n, _)) -> 
            if n < 0 then failwith "argument index must be positive"
            n
        | Leaf _ -> 0

        | Unary (op, a) -> requiredNumberOfArgs a
        | Binary (op, a, b) -> max (requiredNumberOfArgs a) (requiredNumberOfArgs b)
    
    /// checks if the arguments' shapes are compatible with the result shape        
    let checkArgShapes (expr: ElemExprT) (argShapes: ShapeSpecT list) (resShape: ShapeSpecT) =
        // check number of arguments
        let nArgs = List.length argShapes
        let nReqArgs = requiredNumberOfArgs expr       
        if nReqArgs > nArgs then
            failwithf "the element expression requires at least %d arguments but only %d arguments were specified"
                nReqArgs nArgs

        // check dimensionality of arguments
        let rec checkDims expr =
            match expr with
            | Leaf (ArgElement (Arg n, idx)) ->
                let idxDim = ShapeSpec.nDim idx
                let argDim = ShapeSpec.nDim argShapes.[n]
                if idxDim <> argDim then
                    failwithf 
                        "the argument with zero-based index %d has %d dimensions but was used  \
                         with %d dimensions in the element expression" n argDim idxDim
            | Leaf _ -> ()

            | Unary (_, a) -> checkDims a
            | Binary (_, a, b) -> checkDims a; checkDims b
        checkDims expr

    /// substitutes the specified size symbols with their replacements 
    let rec substSymSizes symSizes expr = 
        let sSub = substSymSizes symSizes
        let sSize = SymSizeEnv.subst symSizes
        let sShp = SymSizeEnv.substShape symSizes

        match expr with
        | Leaf (SizeValue sc) -> Leaf (SizeValue (sSize sc))
        | Leaf (ArgElement (arg, argIdxs)) -> Leaf (ArgElement (arg, sShp argIdxs))
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
    let canEvalAllSymSizes expr =
        let rec canEval expr =  
            match expr with
            | Leaf (SizeValue sc) -> SizeSpec.canEval sc
            | Leaf (ArgElement (arg, argIdxs)) -> ShapeSpec.canEval argIdxs
            | Leaf _ -> true

            | Unary (Sum (sym, first, last), a) -> 
                // replace sum index by one before testing for evaluability
                let sumSymVals = Map [sym, SizeSpec.one]
                SizeSpec.canEval first && 
                SizeSpec.canEval last && 
                canEval (a |> substSymSizes sumSymVals)
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
            seq {for dim=0 to 20 do yield (idxSymbol dim, SizeSpec.one)}
            |> Map.ofSeq
        expr |> substSymSizes dummySymVals |> canEval

    /// pretty string of an element expression
    let rec prettyString (expr: ElemExprT) =
        // TODO: delete unnecessary brackets
        match expr with
        | Leaf (op) -> 
            match op with
            | Const v -> sprintf "%A" v
            | SizeValue ss -> sprintf "%A" ss
            | ArgElement ((Arg a), idxs) -> sprintf "a%d%A" a idxs
        
        | Unary (op, a) ->
            match op with
            | Negate -> sprintf "-(%s)" (prettyString a)
            | Abs -> sprintf "abs(%s)" (prettyString a)
            | SignT -> sprintf "signt(%s)" (prettyString a) 
            | Log -> sprintf "log(%s)" (prettyString a)
            | Log10 -> sprintf "log10(%s)" (prettyString a)
            | Exp -> sprintf "exp(%s)" (prettyString a)
            | Sin -> sprintf "sin(%s)" (prettyString a)
            | Cos -> sprintf "cos(%s)" (prettyString a)
            | Tan -> sprintf "tan(%s)" (prettyString a)
            | Asin -> sprintf "asin(%s)" (prettyString a)
            | Acos -> sprintf "acos(%s)" (prettyString a)
            | Atan -> sprintf "atan(%s)" (prettyString a)
            | Sinh -> sprintf "sinh(%s)" (prettyString a)
            | Cosh -> sprintf "cosh(%s)" (prettyString a)
            | Tanh -> sprintf "tanh(%s)" (prettyString a)
            | Sqrt -> sprintf "sqrt(%s)" (prettyString a)
            | Ceil -> sprintf "ceil(%s)" (prettyString a) 
            | Floor -> sprintf "floor(%s)" (prettyString a) 
            | Round -> sprintf "round(%s)" (prettyString a)
            | Truncate -> sprintf "truncate(%s)" (prettyString a)
            | Sum (sumSym, first, last)-> 
                sprintf "sum(%A[%A..%A], %s)" sumSym first last (prettyString a)
            | KroneckerRng (s, first, last) ->
                sprintf "%s[%s%s%s %s %s%s%s](%s)"  (String('\u03B4',1)) 
                                                    (prettyString (Leaf (SizeValue first)))
                                                    (String('\u2264',1))
                                                    (prettyString (Leaf (SizeValue s)))
                                                    (String('\u2227',1))
                                                    (prettyString (Leaf (SizeValue s)))
                                                    (String('\u2264',1))
                                                    (prettyString (Leaf (SizeValue last)))
                                                    (prettyString a)
        | Binary(op, a, b) -> 
            match op with
            | Add -> sprintf "(%s + %s)" (prettyString a) (prettyString b)
            | Substract -> sprintf "(%s - %s)" (prettyString a) (prettyString b)
            | Multiply -> sprintf "(%s * %s)" (prettyString a) (prettyString b)
            | Divide -> sprintf "(%s / %s)" (prettyString a) (prettyString b)
            | Modulo -> sprintf "(%s %% %s)" (prettyString a) (prettyString b)
            | Power -> sprintf "(%s ** %s)" (prettyString a) (prettyString b)
            | IfThenElse (left,right)-> 
                sprintf "ifThenElse(%A = %A, %s, %s)" left right (prettyString a) (prettyString b)

    type ElemExprT with
        member this.PrettyString = prettyString this

[<AutoOpen>]
module ElemExprTypes =
    type ElemExprT = ElemExpr.ElemExprT
