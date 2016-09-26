namespace SymTensor

open System.Collections.Generic
open Microsoft.FSharp.Reflection

open Basics
open ArrayNDNS
open ShapeSpec
open VarSpec


/// expression module
module Expr =
    open ArrayND

    /// boxes the contents of an option
    let inline boxOption (oo: #obj option) = 
        match oo with
        | Some o -> Some (o :> obj)
        | None -> None

    /// start plus the specified number of (symbolic elements)
    type PlusElems (elems: SizeSpecT) =
        new (intElems: int) = PlusElems (SizeSpec.fix intElems)
        member this.Elems = elems

    /// arity of an op
    type ArityT =
        | FixedArity of int
        | DynamicArity

    /// ops with no exprs as arguments
    [<StructuralComparison; StructuralEquality>]
    type LeafOpT =

        // ==== tensor creation ====
        /// tensor with 1 on diagonal of given shape
        | Identity of shape:SizeSpecT * typ:TypeNameT
        /// zero tensor of given shape       
        | Zeros of shape:ShapeSpecT * typ:TypeNameT                  
        /// scalar of given value
        | ScalarConst of ConstSpecT
        /// scalar of the given size
        | SizeValue of value:SizeSpecT * typ:TypeNameT

        // ==== variable access ====
        /// variable read
        | Var of VarSpecT      
        

    /// ops with one expr as argument
    and [<StructuralComparison; StructuralEquality>] 
        UnaryOpT =

        // ==== unary elementwise ==== 
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

        // ==== element-wise unary logic ====
        | Not

        // ==== tensor operations ====
        /// extract diagonal along given axes
        | Diag of int * int
        /// build diagonal matrix along given axes
        | DiagMat of int * int
        /// matrix inverse
        | Invert

        // ==== reductions ====
        /// summation of all elements
        | Sum                           
        /// summation over given dimension
        | SumAxis of int                

        // ==== shape operations ====
        /// reshape tensor; element count does not change
        | Reshape of ShapeSpecT         
        /// broadcast tensor; element count may change
        | DoBroadcast of ShapeSpecT       
        /// permutes the axes of the tensor
        | PermuteAxes of perm:int list
        /// subtensor 
        | Subtensor of ExprRngsSpecT

        // ==== variable storage ====
        /// variable write
        | StoreToVar of VarSpecT

        // ==== misc ====
        /// nullifies the Jacobian of its argument when calculating derivatives
        | NullifyJacobian
        /// assumes the specified Jacobian for its argument when calculating derivatives
        | AssumeJacobian of ExprT
        /// prints the value together with the given string
        | Print of string
        /// dumps the value into the given dataset in the active HDF5 dump file
        | Dump of string
        /// checks the value for NaNs and infinities, outputs their location and stops the computation
        | CheckFinite of string
        /// annotation (no influence on value)
        | Annotated of string       

    and ExprRngSpecT = SimpleRangeSpecT<ExprT>
    and ExprRngsSpecT = SimpleRangesSpecT<ExprT>

    /// ops with two exprs as arguments
    and [<StructuralComparison; StructuralEquality>] 
        BinaryOpT =

        // ==== binary elementwise ====
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power            
        | MaxElemwise
        | MinElemwise        
           
        // ==== element-wise binary comparison ====
        | Equal
        | Less
        | LessEqual
        | Greater
        | GreaterEqual
        | NotEqual     

        // ==== element-wise binary logic ====
        | And
        | Or

        // ==== element-wise conditional ====
        | IfThenElse of ExprT

        // ==== matrix/tensor operations ====
        /// matrix*matrix => matrix dot product
        | Dot                           
        /// tensor product 
        | TensorProduct        
        
        // ==== shape operations ====
        /// replace subtensor
        | SetSubtensor of ExprRngsSpecT                 

    /// ops with an arbitrary exprs as arguments
    and [<StructuralComparison; StructuralEquality>] 
        NaryOpT =

        /// evaluate all subexpressions but discard them
        | Discard        
        /// elementwise calculated tensor
        | Elements of ShapeSpecT * ElemExpr.ElemExprT
        /// elementwise interpolation
        | Interpolate of InterpolatorT
        /// extension op
        | ExtensionOp of IOp
   
   
    /// A mathematical operation in an expression.
    /// This models a mathematical function or operator that takes one or more tensors
    /// and returns one tensor.
    and IOp =
        inherit System.IComparable
       
        /// Should return the shape of the result, given the shape of the arguments.
        abstract Shape: argShapes: ShapeSpecT list -> ShapeSpecT      
        
        /// Should check if the shapes of the arguments are acceptable and,
        /// if not, raise an exception.
        abstract CheckArgs: argShapes: ShapeSpecT list -> unit      

        /// Should return the op with all symbolic sizes substituted using the specified
        /// substitution table.
        /// Return a *new* op with substitution applied. Do not apply the mapping in-place.
        abstract SubstSymSizes: symSizes: SymSizeEnvT -> IOp

        /// Should be true, if all symbolic sizes can be evaluated to numeric sizes.
        /// This is the case if the function ShapeSpec.canEval or SizeSpec.canEval respectively
        /// return true on all sizes used in this op.
        abstract CanEvalAllSymSizes: bool

        /// Should compute the derivative w.r.t. each argument given the derivative w.r.t. the op.
        /// The derivative is always an NxM matrix where N is the number of elements of the function
        /// the derivative of which is being taken and M is the number of elements of the argument
        /// w.r.t. which the derivative is being taken. 
        /// Thus, if dOp is an NxK matrix and an argument has M elements, the derivative matrix
        /// you return w.r.t. that argument must have NxM elements.
        abstract Deriv: dOp:ExprT -> args:ExprT list -> ExprT list

        /// Should evaluate the numerical value of this op given the numerical values of its arguments.
        /// This evaluation should be done on the host using the simplest means possible and is used
        /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
        /// implementations. This method may be omitted when no verification will be done.
        abstract EvalSimple: args:ArrayNDHostT<'T> list -> ArrayNDHostT<'T>

    /// an expression
    and [<StructuralComparison; StructuralEquality; StructuredFormatDisplay("{Pretty}")>] 
        ExprT =
        | Leaf of LeafOpT
        | Unary of UnaryOpT * ExprT
        | Binary of BinaryOpT * ExprT * ExprT
        | Nary of NaryOpT * (ExprT list)

        /// pretty string
        member this.Pretty =
            match this with
            | Leaf op -> sprintf "{%A}" op 
            | Unary (op, a) -> sprintf "{%A} (%A)" op a
            | Binary (op, a, b) -> sprintf "{%A} (%A, %A)" op a b
            | Nary (op, es) -> sprintf "{%A} (%A)" op es

    type FullExprRngSpecT = RangeSpecT<ExprT>
    type FullExprRngsSpecT = RangesSpecT<ExprT>

    /// matches all unary ops that work elementwise
    let (|UnaryElemwiseOp|_|) uop =
        match uop with
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
        | Not
            -> Some ()
        | _ -> None

    /// matches all binary ops that work elementwise
    let (|BinaryElemwiseOp|_|) bop =
        match bop with
        | Add
        | Substract
        | Multiply
        | Divide
        | Modulo
        | Power
        | MaxElemwise
        | MinElemwise
        | Equal
        | Less
        | LessEqual
        | Greater
        | GreaterEqual
        | NotEqual     
        | IfThenElse _
        | And
        | Or
            -> Some ()
        | _ -> None


    /// Traverses the op tree and for each op calls a function on its arguments and replaces 
    /// them by the function's return value(s).
    let rec mapOperands unaryMapping binaryMapping naryMapping expr =
        let subMap = mapOperands unaryMapping binaryMapping naryMapping
        match expr with
        | Unary(op, a) -> Unary(op, unaryMapping op (subMap a))
        | Binary(op, a, b) -> 
            let ma, mb = binaryMapping op (subMap a) (subMap b)
            Binary(op, ma, mb)
        | Nary(op, es) ->
            let mes = naryMapping op (es |> List.map subMap)
            Nary(op, mes)
        | _ -> expr

    /// returns true if subExpr is contained in expr
    let rec contains subExpr expr =
        if expr = subExpr then true
        else
            match expr with
            | Unary (_, a) -> contains subExpr a
            | Binary (_, a, b) -> contains subExpr a || contains subExpr b
            | Nary (_, es) -> List.exists (contains subExpr) es
            | _ -> false

    /// Produces an error message about incompatible shapes.
    let failshape op sa sb =
        failwithf "op %A was provided with arrays of incompatible shapes %A and %A" op sa sb

    /// Returns the type of the given expression.
    let rec typename expr =
        match expr with
        | Leaf (Identity (_, tn)) -> tn
        | Leaf (Zeros (_, tn)) -> tn
        | Leaf (ScalarConst cs) -> cs.TypeName
        | Leaf (SizeValue (_, tn)) -> tn
        | Leaf (Var vs) -> vs.TypeName

        | Binary (Equal, _, _)
        | Binary (Less, _, _)
        | Binary (LessEqual, _, _)
        | Binary (Greater, _, _)
        | Binary (GreaterEqual, _, _)
        | Binary (NotEqual, _, _)
            -> TypeName.ofType<bool>

        | Unary (_, a) -> typename a
        | Binary (_, a, b) -> typename a
        | Nary (_, es) -> typename (List.head es)

    /// Returns the shape of the given expression.
    let rec shapeOf expr =
        // We assume that all operands have compatible size. 
        // For elementwise operations we assume that a and b are already broadcasted
        // to have the *same* size.

        match expr with

        // tensor creation
        | Leaf(Identity (ss, _)) -> ShapeSpec.matrix ss ss
        | Leaf(Zeros (ss, _)) -> ss
        | Leaf(ScalarConst _) -> ShapeSpec.scalar
        | Leaf(SizeValue _) -> ShapeSpec.scalar

        // variable access
        | Leaf(Var vs) -> VarSpec.shape vs

        // unary elementwise
        | Unary (Negate, a)                       
        | Unary (Abs, a)
        | Unary (SignT, a)
        | Unary (Log, a)
        | Unary (Log10, a)                           
        | Unary (Exp, a)                           
        | Unary (Sin, a)
        | Unary (Cos, a)
        | Unary (Tan, a)
        | Unary (Asin, a)
        | Unary (Acos, a)
        | Unary (Atan, a)
        | Unary (Sinh, a)
        | Unary (Cosh, a)
        | Unary (Tanh, a)
        | Unary (Sqrt, a)
        | Unary (Ceil, a)
        | Unary (Floor, a)
        | Unary (Round, a)
        | Unary (Truncate, a)
        | Unary (Not, a)
        | Unary (NullifyJacobian, a)
        | Unary (AssumeJacobian _, a)
            -> shapeOf a

        // tensor operations
        | Unary(Diag(ax1, ax2), a) -> shapeOf a |> ShapeSpec.withoutAxis ax2
        | Unary(DiagMat(ax1, ax2), a) ->  shapeOf a |> List.insert ax2 (shapeOf a).[ax1]
        | Unary(Invert, a) -> shapeOf a

        // reductions
        | Unary(Sum, _) -> ShapeSpec.scalar
        | Unary(SumAxis(ax), a) -> shapeOf a |> ShapeSpec.withoutAxis ax

        // shape operations
        | Unary(Reshape(ss), _) -> ss
        | Unary(DoBroadcast(ss), _) -> ss
        | Unary(PermuteAxes perm, a) -> shapeOf a |> ShapeSpec.permuteAxes perm
        | Unary(Subtensor(srs), a) ->
            (srs, shapeOf a)
            ||> List.map2 (fun sr shp ->
                 match sr with
                 | SRSSymStartSymEnd (s, fo)    -> (fo |? (shp - SizeSpec.one)) + 1 - s
                 | SRSDynStartSymSize (_, size) -> size)

        // misc
        | Unary(StoreToVar _, a) -> ShapeSpec.emptyVector
        | Unary(Print _, a) -> shapeOf a
        | Unary(Dump _, a) -> shapeOf a
        | Unary(CheckFinite _, a) -> shapeOf a
        | Unary(Annotated(_), a) -> shapeOf a

        // binary elementwise
        | Binary (Add, a, _)                         
        | Binary (Substract, a, _)                     
        | Binary (Multiply, a, _)                      
        | Binary (Divide, a, _)                        
        | Binary (Modulo, a, _)
        | Binary (Power, a, _)     
        | Binary (MaxElemwise, a, _)                    
        | Binary (MinElemwise, a, _)       
        | Binary (Equal, a, _)             
        | Binary (Less, a, _)
        | Binary (LessEqual, a, _)
        | Binary (Greater, a, _)
        | Binary (GreaterEqual, a, _)
        | Binary (NotEqual, a, _)
        | Binary (IfThenElse _, a, _)
        | Binary (And, a, _)
        | Binary (Or, a, _)
            -> shapeOf a
            
        // matrix/tensor operations
        | Binary (Dot, a, b) -> 
            let sa, sb = shapeOf a, shapeOf b
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 2, 2 -> ShapeSpec.matrix sa.[0] sb.[1]
            | na, nb when na=nb -> sa.[0 .. na-2] @ [sb.[nb-1]]
            | _ -> failwithf "invalid dot product shapes: %A, %A" sa sb
        | Binary (TensorProduct, a, b) -> 
            let sa, sb = shapeOf a, shapeOf b
            List.map2 (*) sa sb

        // shape operations
        | Binary (SetSubtensor ss, a, b) ->
            shapeOf a

        // misc
        | Nary(Discard, _) -> ShapeSpec.emptyVector 
        | Nary(Elements (resShape, elemExpr), _) -> resShape
        | Nary(Interpolate _, es) -> shapeOf es.Head
        | Nary(ExtensionOp eop, es) -> eop.Shape (es |> List.map shapeOf)

    /// number of elements of given expression
    let nElems expr =
        expr |> shapeOf |> ShapeSpec.nElem

    /// number of dimensions of given expression
    let nDims expr =
        expr |> shapeOf |> ShapeSpec.nDim

    /// Wraps the given op in a Reshape op if its shape does not match ss.
    let reshapeIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(Reshape(ss), expr)

    /// Wraps the given op in a Broadcast op if its shape does not match ss.
    let broadcastIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(DoBroadcast(ss), expr)

    type ExprT with
        /// symbolic shape
        member this.Shape = shapeOf this

        /// number of dimensions
        member this.NDims = ShapeSpec.nDim this.Shape

        /// symbolic number of elements
        member this.NElems = nElems this

        /// type name of this expression
        member this.TypeName = typename this  

        /// type of this expression
        member this.Type = this.TypeName |> TypeName.getType 

    /// checks that given axis is valid for specified expression
    let checkAxis ax expr =
        if not (0 <= ax && ax < nDims expr) then
            failwithf "invalid axis %d for expression of shape %A" ax (shapeOf expr)

    /// expressions that were already checked for correctness
    let checkedExprs = HashSet<ExprT> (HashIdentity.Reference)

    /// Checks ops' arguments for compatible shapes.
    let rec checkExpr (expr: ExprT) =
        if not (checkedExprs.Contains expr) then
            let mutable shapesBeingChecked = []
            let mutable opBeingChecked = fun () -> ""
            let (.=) (ssa: SizeSpecT) (ssb: SizeSpecT) =
                if not (ssa .= ssb) then 
                    failwithf "%s is incompatiables with shapes %A" 
                        (opBeingChecked()) shapesBeingChecked
            let (..=) (sa: ShapeSpecT) (sb: ShapeSpecT) =
                List.iter2 (.=) sa sb
            let reqBool a =
                if typename a <> TypeName.ofType<bool> then
                    failwithf "%s requires data type bool but got %A" 
                        (opBeingChecked()) (typename a).Type

            if typename expr = TypeName.ofType<obj> then
                failwith "Expression type cannot be object."

            match expr with 
            | Leaf op -> ()           

            | Unary (op, a) ->
                checkExpr a
                let sa = shapeOf a
                let nda = ShapeSpec.nDim sa
                shapesBeingChecked <- [sa]
                opBeingChecked <- fun () -> sprintf "%A" op

                match op with
                | Not -> reqBool a
                | SumAxis(ax) when not (0 <= ax && ax < nda) ->
                    failwithf "cannot sum over non-existant axis %d of array with shape %A" ax sa
                | Reshape(ss) ->
                    (ShapeSpec.nElem sa) .= (ShapeSpec.nElem ss) 
                | DoBroadcast(ss) -> 
                    if ShapeSpec.nDim ss <> nda then
                        failwithf "array of shape %A does not have same number of dimesions as broadcast shape %A"
                            sa ss
                    for dim in 0 .. (ShapeSpec.nDim ss) - 1 do
                        match sa.[dim], ss.[dim] with
                        | SizeSpecT.Broadcast, _ -> ()
                        | ssa, ssb -> ssa .= ssb
                | PermuteAxes perm -> 
                    if nda <> List.length perm then
                        failwithf "permutation %A must have same rank as shape %A" perm sa
                    if not (Permutation.is perm) then
                        failwithf "%A is not a valid permutation of an %d-dimensional tensor" perm nda
                | StoreToVar vs ->
                    sa ..= (VarSpec.shape vs)
                | Diag(ax1, ax2) ->
                    if not (0 <= ax1 && ax1 < nda && 0 <= ax2 && ax2 < nda) then
                        failwithf "cannot extract diagonal from non-existant axis %d or %d of array with shape %A" 
                            ax1 ax2 sa
                    if not (ax1 < ax2) then 
                        failwith "first axis for extracting diagonal must come before second axis"
                    sa.[ax1] .= sa.[ax2]
                | DiagMat(ax1, ax2) ->
                    if not (0 <= ax1 && ax1 < nda && 0 <= ax2 && ax2 <= nda) then
                        failwithf "cannot build diagonal over non-existant axis %d or %d of array with shape %A" 
                            ax1 ax2 sa
                    if not (ax1 < ax2) then 
                        failwith "first axis for building diagonal matrix must come before second axis"
                | Invert ->
                    if nda < 2 then
                        failwithf "need at least a matrix to invert but got shape %A" sa
                    sa.[nda-2] .= sa.[nda-1]
                | AssumeJacobian jac ->
                    checkExpr jac
                    if typename jac <> typename expr then
                        failwithf "Jacobian type %A does not match expression type %A."
                            (typename jac).Type (typename expr).Type
                    if nDims jac <> 2 then
                        failwithf "Jacobian shape %A must be two-dimensional" (shapeOf jac)
                    if (shapeOf jac).[1] <> nElems expr then
                        failwithf "Jacobian shape %A must have %A elements in second dimension" 
                            (shapeOf jac) (nElems expr)
                | _ -> ()

            | Binary (op, a, b) ->
                checkExpr a
                checkExpr b
                let sa, sb = shapeOf a, shapeOf b
                let nda, ndb = ShapeSpec.nDim sa, ShapeSpec.nDim sb
                shapesBeingChecked <- [sa; sb]
                opBeingChecked <- fun () -> sprintf "%A" op

                let ta, tb = typename a, typename b
                if ta <> tb then
                    failwithf "cannot apply binary operation %s to expressions of \
                               different types %A and %A" (opBeingChecked()) ta.Type tb.Type

                match op with
                | And
                | Or ->
                    reqBool a
                    reqBool b
                | IfThenElse c ->
                    checkExpr c
                    if c.Type <> typeof<bool> then
                        failwith "condition of IfThenElse must be expression of type bool"
                    c.Shape ..= sa 
                    sa ..= sb
                | BinaryElemwiseOp ->
                    sa ..= sb 
                | Dot -> 
                    match nda, ndb with
                    | 2, 2 -> sa.[1] .= sb.[0] 
                    | na, nb when na = nb -> 
                        sa.[na-1] .= sb.[nb-2]
                        for n = 0 to na - 3 do sa.[n] .= sb.[n]
                    | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
                | TensorProduct when nda <> ndb ->
                    failwithf "cannot compute tensor product between arrays of shapes %A and %A" sa sb
                | _ -> ()

            | Nary (op, es) ->
                es |> List.iter checkExpr
                let ss = es |> List.map shapeOf
                shapesBeingChecked <- ss
                opBeingChecked <- fun () -> sprintf "%A" op

                if es |> List.exists (fun e -> typename e <> typename es.Head) then
                    failwithf "cannot apply n-ary operation %s to expressions of different types %A"
                        (opBeingChecked()) (es |> List.map (typename >> TypeName.getType))

                match op with
                | Elements (trgtShp, elemExpr) -> ElemExpr.checkArgShapes elemExpr ss trgtShp
                | Interpolate ip ->
                    let nDims = ip.MinArg.Length
                    if nDims < 1 then
                        failwith "interpolator must be at least one-dimensional"
                    if ip.MaxArg.Length <> nDims || ip.Outside.Length <> nDims || ip.Resolution.Length <> nDims then
                        failwith "MinArg, MaxArg, Resolution and Outside have inconsistent lengths"
                    if es.Length <> nDims then
                        failwith "number of arguments does not match dimensionality of interpolator"
                    if not ((ip.MinArg, ip.MaxArg) ||> List.forall2 (fun mi ma -> conv<float> mi < conv<float> ma)) then
                        failwith "MinArg of interpolator must be smaller than MaxArg"
                    if ip.Resolution |> List.exists ((>) 0.0) then
                        failwith "Resolution of interpolator must be positive"
                    match ip.Derivative with
                    | Some d when d.NDims <> nDims ->
                        failwith "Dimensionality of derivative interpolator must match dimensionality of interpolator"
                    | _ -> ()
                    for s in ss do 
                        s ..= ss.Head
                | ExtensionOp eop -> eop.CheckArgs ss
                | _ -> ()

            checkedExprs.Add expr |> ignore

    /// substitues the given symbol sizes into the expression
    let rec substSymSizes symSizes (expr: ExprT) =
        let sSub = substSymSizes symSizes
        let sSize = SymSizeEnv.subst symSizes
        let sShp = SymSizeEnv.substShape symSizes
        let sSrs = SymSizeEnv.substRange symSizes

        match expr with
        | Leaf (Identity (ss, tn)) -> Leaf (Identity (sSize ss, tn))
        | Leaf (Zeros (ss, tn)) -> Leaf (Zeros (sShp ss, tn))
        | Leaf (SizeValue (sc, tn)) -> Leaf (SizeValue (sSize sc, tn))
        | Leaf (Var vs) -> Leaf (Var {vs with Shape = sShp vs.Shape})
        | Leaf _ -> expr

        | Unary (Reshape ss, a) -> Unary (Reshape (sShp ss), sSub a)
        | Unary (DoBroadcast ss, a) -> Unary (DoBroadcast (sShp ss), sSub a)
        | Unary (StoreToVar vs, a) -> Unary (StoreToVar {vs with Shape = sShp vs.Shape}, sSub a)
        | Unary (Subtensor srs, a) -> Unary (Subtensor (sSrs srs), sSub a)
        | Unary (AssumeJacobian jac, a) -> Unary (AssumeJacobian (sSub jac), sSub a)
        | Unary (op, a) -> Unary (op, sSub a)

        | Binary (IfThenElse c, a, b) -> Binary (IfThenElse (sSub c), sSub a, sSub b)
        | Binary (SetSubtensor srs, a, b) -> Binary (SetSubtensor (sSrs srs), sSub a, sSub b)
        | Binary (op, a, b) -> Binary (op, sSub a, sSub b)

        | Nary (Elements (trgtShp, elemExpr), es) -> 
            Nary (Elements (sShp trgtShp, ElemExpr.substSymSizes symSizes elemExpr), List.map sSub es)
        | Nary (ExtensionOp eop, es) -> Nary (ExtensionOp (eop.SubstSymSizes symSizes), List.map sSub es)
        | Nary (op, es) -> Nary (op, List.map sSub es)

    /// true if all shapes in the expression can be evaluated to numeric shapes
    let rec canEvalAllSymSizes (expr: ExprT) =
        match expr with
        | Leaf (Identity (ss, tn)) -> SizeSpec.canEval ss
        | Leaf (Zeros (ss, tn)) -> ShapeSpec.canEval ss
        | Leaf (SizeValue (sc, tn)) -> SizeSpec.canEval sc
        | Leaf (Var vs) -> ShapeSpec.canEval (VarSpec.shape vs)
        | Leaf _ -> true

        | Unary (Reshape ss, a) -> ShapeSpec.canEval ss && canEvalAllSymSizes a
        | Unary (DoBroadcast ss, a) -> ShapeSpec.canEval ss && canEvalAllSymSizes a
        | Unary (StoreToVar vs, a) -> ShapeSpec.canEval (VarSpec.shape vs) && canEvalAllSymSizes a
        | Unary (Subtensor srs, a) -> SimpleRangesSpec.canEvalSymbols srs && canEvalAllSymSizes a
        | Unary (AssumeJacobian jac, a) -> canEvalAllSymSizes jac && canEvalAllSymSizes a
        | Unary (op, a) -> canEvalAllSymSizes a

        | Binary (SetSubtensor srs, a, b) -> 
            SimpleRangesSpec.canEvalSymbols srs && canEvalAllSymSizes a && canEvalAllSymSizes b
        | Binary (IfThenElse c, a, b) ->
            canEvalAllSymSizes c && canEvalAllSymSizes a && canEvalAllSymSizes b 
        | Binary (op, a, b) -> canEvalAllSymSizes a && canEvalAllSymSizes b

        | Nary (Elements (trgtShp, elemExpr), es) -> 
            ShapeSpec.canEval trgtShp && 
            ElemExpr.canEvalAllSymSizes elemExpr && 
            List.forall canEvalAllSymSizes es
        | Nary (ExtensionOp eop, es) -> eop.CanEvalAllSymSizes && List.forall canEvalAllSymSizes es
        | Nary (op, es) -> List.forall canEvalAllSymSizes es

    /// Traverses the expression and checks ops' arguments for compatible shapes.
    let check (expr: ExprT) : ExprT =
        checkExpr expr |> ignore
        expr

    /// Replaces all occurences of "part" in "expr" with "replacement".
    let subst part replacement expr =
        // TODO: currently does not substitues into Subtensor and SetSubtensor dyanmic range expression.
        let rec doSubst part replacement expr =       
            let subSubst = doSubst part replacement
            match expr with
            | _ when expr = part -> replacement
            | Leaf _ -> expr
            | Unary (op, a) -> Unary (op, subSubst a)
            | Unary (AssumeJacobian jac, a) ->
                Unary (AssumeJacobian (subSubst jac), subSubst a)
            | Binary (IfThenElse c, a, b) -> 
                Binary (IfThenElse (subSubst c), subSubst a, subSubst b)
            | Binary (op, a, b) -> Binary (op, subSubst a, subSubst b)
            | Nary (op, es) -> Nary (op, es |> List.map subSubst)

        doSubst part replacement expr |> check

    /// scalar constant of given value
    let scalar f = 
        Leaf (ScalarConst (ConstSpec.ofValue f)) |> check

    /// scalar of given value converted to same type as given expression
    let scalarOfSameType expr f = 
        let tn = typename expr
        let v = System.Convert.ChangeType (box f, TypeName.getType tn)
        scalar v

    /// scalar 0 of the same type as given expression
    let inline zeroOfSameType expr = scalarOfSameType expr 0

    /// scalar 1 of the same type as given expression
    let inline oneOfSameType expr = scalarOfSameType expr 1

    /// scalar 2 of the same type as given expression
    let inline twoOfSameType expr = scalarOfSameType expr 2

    /// scalar with value of given size converted to the given type
    [<RequiresExplicitTypeArguments>]
    let sizeValue<'T> size = 
        Leaf (SizeValue (size, TypeName.ofType<'T>)) |> check

    /// scalar with value of given size converted to the same type as given expression
    let sizeValueOfSameType expr size = 
        Leaf (SizeValue (size, typename expr)) |> check

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    let permuteAxes perm a =
        Unary (PermuteAxes perm, a) |> check

    /// swaps two dimensions of a tensor
    let swapDim ax1 ax2 a = 
        a |> checkAxis ax1
        a |> checkAxis ax2
        if ax1 = ax2 then a
        else
            let perm = 
                [0 .. nDims a - 1]
                |> List.map (function
                             | d when d=ax1 -> ax2
                             | d when d=ax2 -> ax1
                             | d -> d)
            a |> permuteAxes perm

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    let transpose a =
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 2 then invalidArg "a" "need at least a matrix to transpose"
        swapDim (nd-2) (nd-1) a

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    let constructElementwise op a b =
        let sa, sb = shapeOf a, shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> reshapeIfNecessary psa |> broadcastIfNecessary bsa
        let bb = b |> reshapeIfNecessary psb |> broadcastIfNecessary bsb    
        Binary (op, ba, bb) |> check


    // elementwise operators
    type ExprT with

        // elementwise unary
        static member (~+) (a: ExprT) = a |> check
        static member (~-) (a: ExprT) = Unary(Negate, a) |> check 
        static member Abs (a: ExprT) = Unary(Abs, a) |> check
        static member SignT (a: ExprT) = Unary(SignT, a) |> check
        static member Log (a: ExprT) = Unary(Log, a) |> check
        static member Log10 (a: ExprT) = Unary(Log10, a) |> check
        static member Exp (a: ExprT) = Unary(Exp, a) |> check
        static member Sin (a: ExprT) = Unary(Sin, a) |> check
        static member Cos (a: ExprT) = Unary(Cos, a) |> check
        static member Tan (a: ExprT) = Unary(Tan, a) |> check
        static member Asin (a: ExprT) = Unary(Asin, a) |> check
        static member Acos (a: ExprT) = Unary(Acos, a) |> check
        static member Atan (a: ExprT) = Unary(Atan, a) |> check
        static member Sinh (a: ExprT) = Unary(Sinh, a) |> check
        static member Cosh (a: ExprT) = Unary(Cosh, a) |> check
        static member Tanh (a: ExprT) = Unary(Tanh, a) |> check
        static member Sqrt (a: ExprT) = Unary(Sqrt, a) |> check
        static member Ceiling (a: ExprT) = Unary(Ceil, a) |> check
        static member Floor (a: ExprT) = Unary(Floor, a) |> check
        static member Round (a: ExprT) = Unary(Round, a) |> check
        static member Truncate (a: ExprT) = Unary(Truncate, a) |> check

        // element-wise unary logic
        static member (~~~~) (a: ExprT) = Unary(Not, a) |> check

        // elementwise binary
        static member (+) (a: ExprT, b: ExprT) = constructElementwise Add a b
        static member (-) (a: ExprT, b: ExprT) = constructElementwise Substract a b
        static member (*) (a: ExprT, b: ExprT) = constructElementwise Multiply a b
        static member (/) (a: ExprT, b: ExprT) = constructElementwise Divide a b
        static member (%) (a: ExprT, b: ExprT) = constructElementwise Modulo a b
        static member Pow (a: ExprT, b: ExprT) = constructElementwise Power a b    

        // element-wise binary logic
        static member (&&&&) (a: ExprT, b: ExprT) = constructElementwise And a b
        static member (||||) (a: ExprT, b: ExprT) = constructElementwise Or a b

        // element-wise binary comparison
        static member (====) (a: ExprT, b: ExprT) = constructElementwise Equal a b
        static member (<<<<) (a: ExprT, b: ExprT) = constructElementwise Less a b
        static member (<<==) (a: ExprT, b: ExprT) = constructElementwise LessEqual a b
        static member (>>>>) (a: ExprT, b: ExprT) = constructElementwise Greater a b
        static member (>>==) (a: ExprT, b: ExprT) = constructElementwise GreaterEqual a b
        static member (<<>>) (a: ExprT, b: ExprT) = constructElementwise NotEqual a b

        // elementwise binary with basetype
        static member (+) (a: ExprT, b: System.IComparable) = a + (scalar b)
        static member (-) (a: ExprT, b: System.IComparable) = a - (scalar b)
        static member (*) (a: ExprT, b: System.IComparable) = a * (scalar b)
        static member (/) (a: ExprT, b: System.IComparable) = a / (scalar b)
        static member (%) (a: ExprT, b: System.IComparable) = a % (scalar b)
        static member Pow (a: ExprT, b: System.IComparable) = a ** (scalar b)
        static member (====) (a: ExprT, b: System.IComparable) = constructElementwise Equal a (scalar b)
        static member (<<<<) (a: ExprT, b: System.IComparable) = constructElementwise Less a (scalar b)
        static member (<<==) (a: ExprT, b: System.IComparable) = constructElementwise LessEqual a (scalar b)
        static member (>>>>) (a: ExprT, b: System.IComparable) = constructElementwise Greater a (scalar b)
        static member (>>==) (a: ExprT, b: System.IComparable) = constructElementwise GreaterEqual a (scalar b)
        static member (<<>>) (a: ExprT, b: System.IComparable) = constructElementwise NotEqual a (scalar b)

        static member (+) (a: System.IComparable, b: ExprT) = (scalar a) + b
        static member (-) (a: System.IComparable, b: ExprT) = (scalar a) - b
        static member (*) (a: System.IComparable, b: ExprT) = (scalar a) * b
        static member (/) (a: System.IComparable, b: ExprT) = (scalar a) / b
        static member (%) (a: System.IComparable, b: ExprT) = (scalar a) % b
        static member Pow (a: System.IComparable, b: ExprT) = (scalar a) ** b
        static member (====) (a: System.IComparable, b: ExprT) = constructElementwise Equal (scalar a) b
        static member (<<<<) (a: System.IComparable, b: ExprT) = constructElementwise Less (scalar a) b
        static member (<<==) (a: System.IComparable, b: ExprT) = constructElementwise LessEqual (scalar a) b
        static member (>>>>) (a: System.IComparable, b: ExprT) = constructElementwise Greater (scalar a) b
        static member (>>==) (a: System.IComparable, b: ExprT) = constructElementwise GreaterEqual (scalar a) b
        static member (<<>>) (a: System.IComparable, b: ExprT) = constructElementwise NotEqual (scalar a) b

        // transposition
        member this.T = transpose this
    

    /// sign keeping type
    let signt (a: ExprT) =
        ExprT.SignT a 

    /// square root
    let sqrtt (a: ExprT) =
        ExprT.Sqrt a

    /// elementwise uses elements from ifTrue if cond is true, 
    /// otherwise elements from ifFalse
    let ifThenElse cond ifTrue ifFalse =
        let shps = [shapeOf cond; shapeOf ifTrue; shapeOf ifFalse]
        let pShps = ShapeSpec.padToSameMany shps
        let bcShps = ShapeSpec.broadcastToSameMany false pShps           
        match pShps, bcShps with
        | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
            let condBc = cond |> reshapeIfNecessary condPShp |> broadcastIfNecessary condBcShp
            let ifTrueBc = ifTrue |> reshapeIfNecessary ifTruePShp |> broadcastIfNecessary ifTrueBcShp
            let ifFalseBc = ifFalse |> reshapeIfNecessary ifFalsePShp |> broadcastIfNecessary ifFalseBcShp
            Binary (IfThenElse condBc, ifTrueBc, ifFalseBc) |> check
        | _ -> failwith "impossible"

    /// elementwise maximum
    let maxElemwise a b =
        constructElementwise MaxElemwise a b

    /// elementwise minimum
    let minElemwise a b =
        constructElementwise MinElemwise a b

    /// reshape (assuming C-continguous order) tensor; element count does not change
    let reshape ss a = Unary(Reshape(ss), a) |> check

    /// broadcast of SizeBroadcast dimensions
    let broadcast ss a = Unary(DoBroadcast(ss), a) |> check

    /// enables broadcasting in the given dimension, it must be of size one
    let enableBroadcast dim a = 
        a |> reshape (shapeOf a |> ShapeSpec.enableBroadcast dim)

    /// disables broadcasting in the given dimension
    let disableBroadcast dim a =
        a |> reshape (shapeOf a |> ShapeSpec.disableBroadcast dim)
  
    /// inserts a broadcast axis at the given dimension
    let insertBroadcastAxis dim a =
        a |> reshape (shapeOf a |> ShapeSpec.insertBroadcastAxis dim)

    /// Replicates the tensor the given number of repetitions along the given axis.
    let replicate dim reps a =
        a |> checkAxis dim

        // 1. insert axis of size one left to repetition axis
        // 2. broadcast along the new axis to number of repetitions
        // 3. reshape to result shape
        a 
        |> insertBroadcastAxis dim
        |> broadcast (a.Shape |> ShapeSpec.insertAxis dim reps)
        |> reshape (a.Shape |> List.set dim (reps * a.Shape.[dim]))

    /// summaiton of all elements
    let sum a = Unary(Sum, a) |> check

    /// summation over given dimension
    let sumAxis ax a = Unary(SumAxis(ax), a) |> check

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    let sumKeepingAxis ax a =
        a |> sumAxis ax |> insertBroadcastAxis ax

    /// mean over all elements
    let mean (a: ExprT) = 
        sum a / sizeValueOfSameType a a.NElems

    /// mean over given dimension
    let meanAxis ax (a: ExprT) =
        sumAxis ax a / sizeValueOfSameType a a.Shape.[ax]

    /// mean over given dimension, while keeping the axis with one (broadcastable) element
    let meanKeepingAxis ax a =
        a |> meanAxis ax |> insertBroadcastAxis ax

    /// identity matrix of given size
    [<RequiresExplicitTypeArguments>]
    let identity<'T> size = 
        Leaf(Identity(size, TypeName.ofType<'T>)) |> check

    /// identity matrix of given size and same type as given expression
    let identityOfSameType expr size =
        Leaf(Identity(size, typename expr)) |> check

    /// zero tensor of given shape
    [<RequiresExplicitTypeArguments>]
    let zeros<'T> ss =
        Leaf(Zeros(ss, TypeName.ofType<'T>)) |> check

    /// zero tensor of given shape and same type as given expression
    let zerosOfSameType expr ss =
        Leaf(Zeros(ss, typename expr)) |> check

    /// zero tensor with same shape and type as given tensor
    let zerosLike a = 
        Leaf (Zeros(shapeOf a, typename a)) |> check

    /// variable of given name and shape
    [<RequiresExplicitTypeArguments>]
    let var<'T> name (ss: ShapeSpecT) = 
        Leaf(Var({Name=name; Shape=ss; TypeName=TypeName.ofType<'T>})) |> check

    /// annotated expression
    let annotate ano a = 
        Unary(Annotated(ano), a) |> check

    /// adds one broadcastable dimension to the left
    let padLeft a =
        let sa = shapeOf a
        reshape (ShapeSpec.padLeft sa) a

    /// adds one broadcastable dimension to the right
    let padRight a =
        let sa = shapeOf a
        reshape (ShapeSpec.padRight sa) a

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    let dot (a: ExprT) (b: ExprT) =
        let sa, sb = shapeOf a, shapeOf b
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 -> 
                // vector-vector dot product
                sum (a * b)
            | 2, 1 -> 
                // matrix-vector dot product
                let bm = b |> reshape (ShapeSpec.padRight sb)
                Binary(Dot, a, bm) |> reshape [sa.[0]]
            | 2, 2 -> 
                // matrix-matrix dot product
                Binary(Dot, a, b)
            | na, nb when na = nb -> 
                // batched matrix-matrix dot product
                let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa sb
                let ba = a |> broadcastIfNecessary bsa
                let bb = b |> broadcastIfNecessary bsb    
                Binary(Dot, ba, bb)
            | na, nb when na = nb + 1 ->
                // batched matrix-vector dot product
                let psb = ShapeSpec.padRight sb
                let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa psb
                let ba = a |> broadcastIfNecessary bsa
                let bb = b |> reshapeIfNecessary psb |> broadcastIfNecessary bsb    
                Binary(Dot, ba, bb) |> reshape bsa.[0 .. na-2]
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        |> check

    /// tensor product
    let tensorProduct (a: ExprT) (b: ExprT) =
        let sa, sb = shapeOf a, shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let a, b = reshapeIfNecessary psa a, reshapeIfNecessary psb b
        Binary(TensorProduct, a, b) |> check

    type ExprT with
        // tensor binary

        /// Dot product.
        /// Behavior depends on the dimensionality of the arguments.
        /// Cases: 
        /// (1, 1) -> vector-vector dot product resulting in a scalar
        /// (2, 1) -> matrix-vector dot product resulting in a vector
        /// (2, 2) -> matrix-matrix dot product resulting in a matrix
        /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
        /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
        static member (.*) (a: ExprT, b: ExprT) = dot a b
        static member (%*) (a: ExprT, b: ExprT) = tensorProduct a b

    /// extract all variables from an expression
    let rec extractVars expr =
        match expr with
        | Leaf (Var vs) -> Set.singleton vs
        | Unary (StoreToVar vs, a) -> extractVars a |> Set.add vs
        | Binary (IfThenElse cond, a, b) -> 
            Set.unionMany [extractVars cond; extractVars a; extractVars b]

        | Leaf _ -> Set.empty
        | Unary (_, a) -> extractVars a
        | Binary (_, a, b) -> Set.union (extractVars a) (extractVars b)
        | Nary (_, es) -> Set.unionMany (es |> List.map extractVars)

    /// extract VarSpec from variable expression
    let extractVar expr = 
        match expr with
        | Leaf(Var(v)) -> v
        | _ -> invalidArg "expr" "not a expr consisting solely of a variable"

    /// make variable expression from VarSpec
    let makeVar vs =
        Leaf(Var(vs)) |> check

    /// store to variable
    let storeToVar ve a =
        let vs = extractVar ve
        Unary(StoreToVar(vs), a) |> check

    /// computes specified expressions, but discards the result
    let discard es =
        Nary(Discard, es) |> check

    /// expression a with the specified subtensor replaced with b
    let setSubtensor a b =
        match a with
        | Unary (Reshape _, (Unary (Subtensor srs, t) as st)) ->
            let stShp = shapeOf st
            Binary (SetSubtensor srs, t, Unary (Reshape stShp, b)) |> check
        | _ ->
            invalidArg "a" "the first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]"

    type ExprT with
        // item / slicing
        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =

            /// converts ints to SizeSpecTs
            let intToSizeSpec (arg: obj) =
                match arg with
                | :? int as f -> SizeSpec.fix f :> obj
                | :? (int option) as fo -> 
                    match fo with
                    | Some f -> Some (SizeSpec.fix f) :> obj
                    | None -> None :> obj
                | _ -> arg

            /// converts argument list to range specification
            let rec parseArgs (args: obj list) : FullExprRngsSpecT =
                match args with
                // direct range specification
                | [:? FullExprRngsSpecT as rngs] -> rngs

                // slices
                | (:? (SizeSpecT option) as so)  :: (:? (SizeSpecT option) as fo)    :: rest ->
                    RSSymStartSymEnd (so, fo) :: parseArgs rest
                | (:? (SizeSpecT option) as so)  :: null                             :: rest ->
                    RSSymStartSymEnd (so, None) :: parseArgs rest
                | null                           :: (:? (SizeSpecT option) as fo)    :: rest ->
                    RSSymStartSymEnd (None, fo) :: parseArgs rest
                | (:? (ExprT option) as so)      :: (:? (PlusElems option) as fo)    :: rest ->
                    if typename so.Value <> TypeName.ofType<int> then
                        failwith "need int expression for range start"
                    RSDynStartSymSize (so.Value, fo.Value.Elems) :: parseArgs rest
                | null                           :: null                             :: rest ->
                    RSSymStartSymEnd (None, None) :: parseArgs rest

                // items
                | (:? SizeSpecT as s)     :: rest -> RSSymElem s :: parseArgs rest
                | (:? SpecialAxisT as s)  :: rest -> match s with
                                                     | NewAxis -> RSNewAxis :: parseArgs rest
                                                     | Fill    -> RSAllFill :: parseArgs rest
                | (:? ExprT as e)         :: rest -> if typename e <> TypeName.ofType<int> then
                                                         failwith "need int expression for element"               
                                                     RSDynElem e :: parseArgs rest
                | []                              -> []
                | _                               -> failwithf "invalid item/slice specification: %A" allArgs

            /// converts a full range specification into a simple range specification
            let rec splitFRS (rngs: FullExprRngsSpecT) (shps: ShapeSpecT) (simpleRs: ExprRngsSpecT) (newShape: ShapeSpecT) =
                match rngs, shps with
                | RSSymElem e :: rngs, _::shps -> splitFRS rngs shps (SRSSymStartSymEnd (e, Some e)::simpleRs) newShape
                | RSDynElem e :: rngs, _::shps -> splitFRS rngs shps (SRSDynStartSymSize (e, SizeSpec.one)::simpleRs) newShape
                | RSSymStartSymEnd (so, fo) :: rngs, shp::shps -> 
                    let size = (fo |? shp) - (so |? SizeSpec.zero) + 1
                    splitFRS rngs shps (SRSSymStartSymEnd (so |? SizeSpec.zero, fo)::simpleRs) (size::newShape)
                | RSDynStartSymSize (s, size) :: rngs, _::shps ->
                    splitFRS rngs shps (SRSDynStartSymSize (s, size)::simpleRs) (size::newShape)
                | RSNewAxis :: rngs, _ ->
                    splitFRS rngs shps simpleRs (SizeSpec.broadcastable::newShape)
                | RSAllFill :: rrngs, _ ->
                    if List.length rngs <= List.length shps then splitFRS (RSAll::rngs) shps simpleRs newShape
                    else splitFRS rrngs shps simpleRs newShape
                | [], [] -> List.rev simpleRs, List.rev newShape
                | _ -> failwith "item/slice processing error"

            // build full range specificaton
            let argList = allArgs |> Array.toList |> List.map intToSizeSpec

            let srs, reshp = 
                match argList with
                | [:? ExprRngsSpecT as srs] -> 
                    // simplified range specification was specified, use directly
                    srs, shapeOf (Unary (Subtensor srs, this))
                | [:? FullExprRngsSpecT as frs] ->
                    // split into simplified range specification and reshape operation
                    splitFRS frs (shapeOf this) [] []
                | _ ->
                    // parse, then split into simplified range specification and reshape operation
                    splitFRS (argList |> parseArgs) (shapeOf this) [] []

            // emit expression
            Unary (Reshape reshp, Unary (Subtensor srs, this))  
            |> check

        member this.Item 
            with get ([<System.ParamArray>] allArgs: obj []) = 
                this.GetSlice (allArgs)
                   
    /// Extracts the diagonal along the given axes.
    let diagAxis ax1 ax2 a = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Unary(Diag (ax1, ax2), a) |> check
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    let diag a = 
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 2 then failwith "need at least a matrix to extract diagonal"
        diagAxis (nd-2) (nd-1) a

    /// Creates a diagonal matrix by duplicating the given dimension.
    let diagMatAxis ax1 ax2 a = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Unary(DiagMat (ax1, ax2), a) |> check

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    let diagMat a =
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 1 then failwith "need at least a vector to create diagonal matrix"
        diagMatAxis (nd-1) nd a

    /// Computes the traces along the given axes.
    let traceAxis ax1 ax2 a =
        let tax = if ax1 < ax2 then ax1 else ax1 + 1
        a |> diagAxis ax1 ax2 |> sumAxis tax

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    let trace a =
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 2 then
            failwith "need at least a two dimensional array for trace"      
        traceAxis (nd-2) (nd-1) a

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    let invert a =
        Unary(Invert, a) |> check

    /// calculates a tensor elementwise using the given element expression and
    /// result shape
    let elements trgtShp elemExpr args =
        Nary (Elements (trgtShp, elemExpr), args) |> check

    /// nullifies the Jacobian when calculating derivatives
    let assumeZeroDerivative expr =
        Unary (NullifyJacobian, expr) |> check

    /// assumes the specified Jacobian when calculating derivatives
    let assumeJacobian jac expr =
        Unary (AssumeJacobian jac, expr) |> check

    /// print the result with the given message when evaluated
    let print msg a =
        Unary (Print msg, a) |> check

    /// dumps the result into the active dump session HDF5 file
    let dump name a =
        Unary (Dump name, a) |> check

    /// checks the value for NaNs and infinities, outputs their location and stops the computation
    let checkFinite name a =
        if Debug.EnableCheckFinite then
            Unary (CheckFinite name, a) |> check
        else a |> check

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the createInterpolator function.
    let interpolate interpolator e =
        Nary (Interpolate interpolator, e) |> check

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the createInterpolator function.
    let interpolate1D interpolator a =
        interpolate interpolator [a]

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the createInterpolator function.
    let interpolate2D interpolator a b =
        interpolate interpolator [a; b]

   


[<AutoOpen>]
module ExprTypes =
    type ArityT = Expr.ArityT
    type LeafOpT = Expr.LeafOpT
    type UnaryOpT = Expr.UnaryOpT
    type BinaryOpT = Expr.BinaryOpT
    type NaryOpT = Expr.NaryOpT
    type IOp = Expr.IOp
    type ExprT = Expr.ExprT

    




