namespace SymTensor

open System.Reflection
open System.Collections.Generic
open Expr



[<AutoOpen>]
module UExprTypes = 

    /// unified variable specification
    [<StructuredFormatDisplay("\"{Name}\" {Shape}")>]
    type UVarSpecT = {
        Name:      string; 
        Shape:     ShapeSpecT;
        TypeName:  TypeNameT;
    }

    // int holds the position of the subuexpr that has the dynamic value
    type UExprRngSpecT = SimpleRangeSpecT<int>
    type UExprRngsSpecT = SimpleRangesSpecT<int>

    type IUExtensionOp =
        inherit System.IComparable
        abstract Arity: ArityT with get        

    type ULeafOpT =
        | Identity of SizeSpecT
        | Zeros of ShapeSpecT                   
        | ScalarConst of System.IComparable
        | SizeValue of SizeSpecT
        | Var of UVarSpecT

    type UUnaryOpT =
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
        | Annotated of Annotation       

    type UBinaryOpT =
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide       
        | Modulo                 
        | Power                         
        | Dot                           
        | TensorProduct                 

    type UNaryOpT =
        | Discard        
        | Subtensor of UExprRngsSpecT 
        | SetSubtensor of UExprRngsSpecT
        | ExtensionOp of IUExtensionOp
             

    /// unified op of any arity and type
    type UOpT =
        | ULeafOp of ULeafOpT
        | UUnaryOp of UUnaryOpT
        | UBinaryOp of UBinaryOpT
        | UNaryOp of UNaryOpT

    /// unified expression (combines all arities and types and ops cannot have expressions as parameters)
    [<StructuralComparison; StructuralEquality; StructuredFormatDisplay("{PrettyString}")>]
    type UExprT = 
        | UExpr of UOpT * TypeNameT * ShapeSpecT * (UExprT list)

        member this.PrettyString =
            match this with
            | UExpr (ULeafOp uop, tn, ss, subs) -> sprintf "%A" uop 
            | UExpr (UUnaryOp uop, tn, ss, subs) -> sprintf "%A (%A)" uop subs.[0]
            | UExpr (UBinaryOp uop, tn, ss, subs) -> sprintf "%A (%A, %A)" uop subs.[0] subs.[1]
            | UExpr (UNaryOp uop, tn, ss, subs) -> sprintf "%A (%A)" uop subs


module UVarSpec =

    /// create variable specifation by name and shape
    let inline ofNameShapeAndTypeName name shape typeName : UVarSpecT =
        {Name=name; Shape=shape; TypeName=typeName;}

    let ofVarSpec (vs: #IVarSpec) =
        {Name=vs.Name; Shape=vs.Shape; TypeName=vs.TypeName}

    let ofExpr expr =
        expr |> Expr.extractVar |> ofVarSpec

    let toVarSpec (vs: UVarSpecT) : VarSpecT<'T> =
        {Name=vs.Name; Shape=vs.Shape;}

    let name (vs: UVarSpecT) =
        vs.Name

    let shape (vs: UVarSpecT) =
        vs.Shape

    let nDims vs =
        shape vs |> List.length

    let typ (vs: UVarSpecT) = 
        vs.TypeName |> TypeName.getType 

    let substSymSizes symSizes (vs: UVarSpecT) = 
        {vs with Shape=SymSizeEnv.substShape symSizes vs.Shape} 

    let tryFindByName (vs: UVarSpecT) map =
        map |> Map.tryPick 
            (fun cvs value -> 
                if name cvs = name vs then Some value
                else None)

    let findByName vs map =
        match tryFindByName vs map with
        | Some value -> value
        | None -> raise (KeyNotFoundException())


module UExprRngsSpec =

    // split into two ops:
    // one that does nothing, just changes the static layout
    // and another that does the copying if necessary
    // op1 : StaticSubtensor
    // op2 : DynamicSubtensor
    // but how does this work with SetSubtensor?

    /// converts a ExprRngsSpecT to a UExprRngSpecT
    let ofExprRngsSpec (sr: ExprRngsSpecT) =
        ([], sr)
        ||> List.mapFold (fun dynExprs rng ->
            let idx = List.length dynExprs 
            match rng with
            | SRSSymStartSymEnd  (s, fo)     -> SRSSymStartSymEnd (s, fo),       dynExprs
            | SRSDynStartSymSize (s, size)   -> SRSDynStartSymSize (idx, size),  dynExprs @ [s])

    /// converts a UExprRngSpecT to a ExprRngsSpecT
    let rec toExprRngsSpec (srs: UExprRngsSpecT) (drs: ExprT<int> list)  =
        match srs, drs with
        | SRSSymStartSymEnd (s, fo) :: srs, _         -> SRSSymStartSymEnd (s, fo)   :: toExprRngsSpec srs drs
        | SRSDynStartSymSize (_, f) :: srs, dr :: rdrs-> SRSDynStartSymSize (dr, f)  :: toExprRngsSpec srs rdrs
        | []                              , []        -> []
        | _                               , _         -> failwith "invalid unified subtensor spec"


module UExpr =

    /// converts an expression to a unified expression
    let rec toUExpr (expr: ExprT<'T>) =
        let tn = TypeName typeof<'T>.AssemblyQualifiedName
        let shp = Expr.shapeOf expr

        let leaf uop        = UExpr (ULeafOp uop, tn, shp, [])
        let unary uop a     = UExpr (UUnaryOp uop, tn, shp, [toUExpr a])
        let binary uop a b  = UExpr (UBinaryOp uop, tn, shp, [toUExpr a; toUExpr b])
        let nary uop se     = UExpr (UNaryOp uop, tn, shp, se |> List.map toUExpr)

        match expr with
        | Leaf (Expr.Identity ss)       -> leaf (Identity ss)
        | Leaf (Expr.Zeros ss)          -> leaf (Zeros ss)
        | Leaf (Expr.ScalarConst v)     -> leaf (ScalarConst (box v :?> System.IComparable))
        | Leaf (Expr.SizeValue sv)      -> leaf (SizeValue sv)
        | Leaf (Expr.Var vs)            -> leaf (Var (UVarSpec.ofVarSpec vs))

        | Unary (Expr.Negate, a)        -> unary Negate a
        | Unary (Expr.Abs, a)           -> unary Abs a
        | Unary (Expr.SignT, a)         -> unary SignT a
        | Unary (Expr.Log, a)           -> unary Log a
        | Unary (Expr.Log10, a)         -> unary Log10 a
        | Unary (Expr.Exp, a)           -> unary Exp a
        | Unary (Expr.Sin, a)           -> unary Sin a
        | Unary (Expr.Cos, a)           -> unary Cos a
        | Unary (Expr.Tan, a)           -> unary Tan a
        | Unary (Expr.Asin, a)          -> unary Asin a
        | Unary (Expr.Acos, a)          -> unary Acos a
        | Unary (Expr.Atan, a)          -> unary Atan a
        | Unary (Expr.Sinh, a)          -> unary Sinh a
        | Unary (Expr.Cosh, a)          -> unary Cosh a
        | Unary (Expr.Tanh, a)          -> unary Tanh a
        | Unary (Expr.Sqrt, a)          -> unary Sqrt a
        | Unary (Expr.Ceil, a)          -> unary Ceil a
        | Unary (Expr.Floor, a)         -> unary Floor a
        | Unary (Expr.Round, a)         -> unary Round a
        | Unary (Expr.Truncate, a)      -> unary Truncate a
        | Unary (Expr.Diag (ax1, ax2), a) -> unary (Diag (ax1, ax2)) a
        | Unary (Expr.DiagMat (ax1, ax2), a)  -> unary (DiagMat (ax1, ax2)) a
        | Unary (Expr.Invert, a)        -> unary Invert a
        | Unary (Expr.Sum, a)           -> unary Sum a
        | Unary (Expr.SumAxis ax, a)    -> unary (SumAxis ax) a
        | Unary (Expr.Reshape ss, a)    -> unary (Reshape ss) a
        | Unary (Expr.DoBroadcast ss, a)-> unary (DoBroadcast ss) a
        | Unary (Expr.SwapDim (ax1, ax2), a) -> unary (SwapDim (ax1, ax2)) a
        | Unary (Expr.Subtensor sr, a)  ->
            let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr    
            let dynUExprs = dynExprs |> List.map toUExprForInt               
            UExpr(UNaryOp (Subtensor usr), tn, shp, toUExpr a :: dynUExprs)
        | Unary (Expr.StoreToVar vs, a) -> unary (StoreToVar (UVarSpec.ofVarSpec vs)) a
        | Unary (Expr.Annotated ano, a) -> unary (Annotated ano) a

        | Binary (Expr.Add, a, b)       -> binary Add a b
        | Binary (Expr.Substract, a, b) -> binary Substract a b
        | Binary (Expr.Multiply, a, b)  -> binary Multiply a b                     
        | Binary (Expr.Divide, a, b)    -> binary Divide a b             
        | Binary (Expr.Modulo, a, b)    -> binary Modulo a b          
        | Binary (Expr.Power, a, b)     -> binary Power a b               
        | Binary (Expr.Dot, a, b)       -> binary Dot a b                   
        | Binary (Expr.TensorProduct, a, b) -> binary TensorProduct a b         
        | Binary (Expr.SetSubtensor sr, a, b) ->
            let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr    
            let dynUExprs = dynExprs |> List.map toUExprForInt 
            UExpr(UNaryOp (SetSubtensor usr), tn, shp, toUExpr a :: toUExpr b :: dynUExprs)

        | Nary (Expr.Discard, se)       -> nary Discard se
        | Nary (Expr.ExtensionOp eop, se) -> nary (ExtensionOp (eop :?> IUExtensionOp)) se

    and private toUExprForInt (expr: ExprT<int>) =
        toUExpr expr

    /// converts a unified expression to an expression of (known) type
    let rec toExprOfType (UExpr (uop, tn, ss, subUExprs) as uexpr) : ExprT<'T> =
        if TypeName.ofType<'T> <> tn then
            failwith "UExpr type does not match"

        let leaf op    = Expr.Leaf op
        let unary op   = Expr.Unary (op, toExprOfType subUExprs.[0])
        let binary op  = Expr.Binary (op, toExprOfType subUExprs.[0], toExprOfType subUExprs.[1])
        let nary op    = Expr.Nary (op, List.map toExprOfType subUExprs)

        match uop with
        | ULeafOp (Identity ss)             -> leaf (Expr.Identity ss)
        | ULeafOp (Zeros ss)                -> leaf (Expr.Zeros ss)
        | ULeafOp (ScalarConst v)           -> leaf (Expr.ScalarConst (box v :?> 'T))
        | ULeafOp (SizeValue sv)            -> leaf (Expr.SizeValue sv)
        | ULeafOp (Var vs)                  -> leaf (Expr.Var (UVarSpec.toVarSpec vs))

        | UUnaryOp Negate                   -> unary Expr.Negate
        | UUnaryOp Abs                      -> unary Expr.Abs
        | UUnaryOp SignT                    -> unary Expr.SignT
        | UUnaryOp Log                      -> unary Expr.Log
        | UUnaryOp Log10                    -> unary Expr.Log10
        | UUnaryOp Exp                      -> unary Expr.Exp                         
        | UUnaryOp Sin                      -> unary Expr.Sin
        | UUnaryOp Cos                      -> unary Expr.Cos
        | UUnaryOp Tan                      -> unary Expr.Tan
        | UUnaryOp Asin                     -> unary Expr.Asin
        | UUnaryOp Acos                     -> unary Expr.Acos
        | UUnaryOp Atan                     -> unary Expr.Atan
        | UUnaryOp Sinh                     -> unary Expr.Sinh
        | UUnaryOp Cosh                     -> unary Expr.Cosh
        | UUnaryOp Tanh                     -> unary Expr.Tanh
        | UUnaryOp Sqrt                     -> unary Expr.Sqrt
        | UUnaryOp Ceil                     -> unary Expr.Ceil
        | UUnaryOp Floor                    -> unary Expr.Floor
        | UUnaryOp Round                    -> unary Expr.Round
        | UUnaryOp Truncate                 -> unary Expr.Truncate
        | UUnaryOp (Diag (ax1, ax2))        -> unary (Expr.Diag (ax1, ax2))
        | UUnaryOp (DiagMat (ax1, ax2))     -> unary (Expr.DiagMat (ax1, ax2))
        | UUnaryOp Invert                   -> unary Expr.Invert
        | UUnaryOp Sum                      -> unary Expr.Sum                           
        | UUnaryOp (SumAxis a)              -> unary (Expr.SumAxis a)            
        | UUnaryOp (Reshape ss)             -> unary (Expr.Reshape ss)
        | UUnaryOp (DoBroadcast ss)         -> unary (Expr.DoBroadcast ss)
        | UUnaryOp (SwapDim (ax1, ax2))     -> unary (Expr.SwapDim (ax1, ax2))
        | UUnaryOp (StoreToVar vs)          -> unary (Expr.StoreToVar (UVarSpec.toVarSpec vs))
        | UUnaryOp (Annotated ano)          -> unary (Expr.Annotated ano)

        | UBinaryOp Add                     -> binary Expr.Add                         
        | UBinaryOp Substract               -> binary Expr.Substract                    
        | UBinaryOp Multiply                -> binary Expr.Multiply                     
        | UBinaryOp Divide                  -> binary Expr.Divide             
        | UBinaryOp Modulo                  -> binary Expr.Modulo          
        | UBinaryOp Power                   -> binary Expr.Power               
        | UBinaryOp Dot                     -> binary Expr.Dot                   
        | UBinaryOp TensorProduct           -> binary Expr.TensorProduct     
            
        | UNaryOp Discard                   -> nary Expr.Discard
        | UNaryOp (Subtensor usr)           ->
            let drs = subUExprs |> List.tail |> List.map toExprOfTypeInt
            unary (Expr.Subtensor (UExprRngsSpec.toExprRngsSpec usr drs))
        | UNaryOp (SetSubtensor usr)        ->
            let drs = subUExprs |> List.skip 2 |>  List.map toExprOfTypeInt
            binary (Expr.SetSubtensor (UExprRngsSpec.toExprRngsSpec usr drs))
        | UNaryOp (ExtensionOp eop)         -> nary (Expr.ExtensionOp (eop :?> IExtensionOp<'T>))

    and private toExprOfTypeInt uexpr : ExprT<int> =
        toExprOfType uexpr

    type private ToExprOfTypeT =
        static member ToExprOfType<'T> uexpr : ExprT<'T> =
            toExprOfType uexpr

    /// converts a unified expression to an expression of the correct type
    let toExpr (UExpr (_, tn, _, _) as uexpr) =
        let gm = typeof<ToExprOfTypeT>.GetMethod ("ToExprOfType", 
                                                  BindingFlags.NonPublic ||| 
                                                  BindingFlags.Public ||| 
                                                  BindingFlags.Static)
        let m = gm.MakeGenericMethod ([| TypeName.getType tn |])
        m.Invoke(null, [| uexpr |])

    /// the op of the given unified expression
    let inline opOf uexpr =
        match uexpr with UExpr(op, typ, shp, se) -> op

    /// the type of the given unified expression
    let inline typeOf uexpr =
        match uexpr with UExpr(op, TypeName tn, shp, se) -> System.Type.GetType(tn)

    /// the type of the given unified expression
    let inline typenameOf uexpr =
        match uexpr with UExpr(op, typ, shp, se) -> typ

    /// the shape of the given unified expression
    let inline shapeOf uexpr = 
        match uexpr with UExpr(op, typ, shp, se) -> shp

    /// counts how many times subExpr occurs in unified expression uexpr
    let subExprOccurrences uexpr =
        let cnt = Dictionary<UExprT, int>()
        let rec build expr =
            if cnt.ContainsKey(expr) then
                cnt.[expr] <- cnt.[expr] + 1
            else
                cnt.[expr] <- 1

            match expr with
            | UExpr (_, _, _, srcs) ->
                for src in srcs do
                    build src
        build uexpr

        fun subExpr ->
            if cnt.ContainsKey(subExpr) then cnt.[subExpr]
            else 0

