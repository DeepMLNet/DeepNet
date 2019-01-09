namespace rec SymTensor.Ops

open Tensor
open Tensor.Backend
open SymTensor
open DeepNet.Utils


module Check =
    let sameType (exprs: Expr2 list) =
        let types = exprs |> List.map Expr2.typeName
        if types |> List.exists ((<>) types.Head) then
            failwithf "All arguments are expected to be of same type, but types are: %A." types

    let sameShape (exprs: Expr2 list) =
        let shapes = exprs |> List.map Expr2.shape
        if shapes |> List.exists ((<>) shapes.Head) then
            failwithf "All arguments are expected to be of same shape, but shapes are: %A." shapes

    let bool (exprs: Expr2 list) =
        let types = exprs |> List.map Expr2.typeName
        if types |> List.exists ((<>) TypeName.ofType<bool>) then
            failwithf "All arguments are expected to be of type bool, but types are: %A." types

    let axis (ax: int) (expr: Expr2) =
        if not (0 <= ax && ax < ShapeSpec.nDim expr.Shape) then
            failwithf "Cannot apply reduction operation over non-existant axis %d of tensor with shape %A." 
                      ax expr.Shape

    let range (range: SimpleRangesSpec) (x: Expr2) =
        if range.Length <> x.NDims then
            failwithf "Invalid range specification for expression of shape %A." x.Shape                
        range |> List.iter (function 
            | SimpleRangeSpec.SymStartSymEnd _ -> ()
            | SimpleRangeSpec.DynStartSymSize (s, _) -> 
                if (s :?> Expr2).DataType <> typeof<int64> then
                    failwithf "Dynamic range start must be of type int64.")

module Args =
    //let leaf : ArgsMap = Map.empty

    let unary x : ArgsMap = Map ["X", x]
    let unaryX (am: Map<string, _>) = am.["X"]

    let binary x y : ArgsMap = Map ["X", x; "Y", y]
    let binaryX (am: Map<string, _>) = am.["X"]
    let binaryY (am: Map<string, _>) = am.["Y"]

    let nary xs : ArgsMap =
        xs |> List.indexed |> List.map (fun (i,v) -> i.ToString(), v) |> Map.ofList
    let naryXs (am: Map<string, _>) =
        let xs = 
            am 
            |> Map.toList 
            |> List.choose (fun (s,v) -> 
                s |> Int32.tryParse |> Option.map (fun i -> i,v))
            |> List.sortBy fst 
        if [0 .. xs.Length-1] <> List.map fst xs then
            failwithf "Cannot convert argument map to argument list: %A" am
        xs |> List.map snd

//[<AutoOpen>]
//module private DerivUtils =
//    /// Expands the second dimension of the Jacobian into the shape of this expression.
//    let expand (dOp: Expr2) (expr: Expr2) = 
//        let funElems = dOp.Shape.[0]
//        dOp |> Expr2.reshape (funElems :: expr.Shape)

//    /// Flattens all but the first dimension of the Jacobian into one dimension.
//    let collapse (g: Expr2) =
//        let funElems = g.Shape.[0]
//        let wrtElems = g.Shape.[1..] |> ShapeSpec.nElem
//        g |> Expr2.reshape [funElems; wrtElems]


[<AutoOpen>]
module LeafOps =

    /// Scalar constant value
    type ScalarConst = { Value: Const } with
        interface IOp2 with    
            member this.Check () = ()
            member this.TypeName = this.Value.TypeName
            member this.Shape = ShapeSpec.scalar
            member this.Args = Map.empty
            member this.ReplaceArgs args = this :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Map.empty
            member this.Eval env = this.Value.AsTensor env.Dev       
    let (|ScalarConst|_|) (expr: Expr2) =
        match expr.Op with
        | :? ScalarConst as this -> Some this.Value
        | _ -> None
    
    /// Value of the specified size
    type SizeValue = { Value: SizeSpec } with
        interface IOp2 with    
            member this.Check () = ()
            member this.TypeName = TypeName.ofType<int64>
            member this.Shape = ShapeSpec.scalar
            member this.Args = Map.empty
            member this.ReplaceArgs args = this :> _
            member this.SubstSymSizes env = { Value = SymSizeEnv.subst env this.Value } :> _
            member this.CanEvalAllSymSizes = SizeSpec.canEval this.Value
            member this.Deriv dOp = Map.empty
            member this.Eval env = 
                SizeSpec.eval this.Value |> Tensor.scalar env.Dev :> ITensor       
    let (|SizeValue|_|) (expr: Expr2) =
        match expr.Op with
        | :? SizeValue as this -> Some this.Value
        | _ -> None

    /// Identity matrix
    type Identity = { Size: SizeSpec; Type: TypeName} with     
        interface IOp2 with    
            member this.Check () = ()
            member this.TypeName = this.Type
            member this.Shape = ShapeSpec.matrix this.Size this.Size
            member this.Args = Map.empty
            member this.ReplaceArgs args = this :> _
            member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> _
            member this.CanEvalAllSymSizes = SizeSpec.canEval this.Size
            member this.Deriv dOp = Map.empty
            member this.Eval env = 
                (Generic<IdentityTyped<_>, IIdentityTyped> [this.Type.Type]).Eval this env
    type internal IIdentityTyped =
        abstract Eval: this:Identity -> env:EvalEnv -> ITensor
    type internal IdentityTyped<'T> () =     
        interface IIdentityTyped with
            member __.Eval this env =
                Tensor<'T>.identity env.Dev (SizeSpec.eval this.Size) :> ITensor       
    let (|Identity|_|) (expr: Expr2) =
        match expr.Op with
        | :? Identity as this -> Some this
        | _ -> None

    /// Counting vector of given size
    type Arange = { Size: SizeSpec; Type: TypeName} with
        interface IOp2 with    
            member this.Check () = ()
            member this.TypeName = this.Type
            member this.Shape = ShapeSpec.vector this.Size
            member this.Args = Map.empty
            member this.ReplaceArgs args = this :> _
            member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> _
            member this.CanEvalAllSymSizes = SizeSpec.canEval this.Size
            member this.Deriv dOp = Map.empty
            member this.Eval env = 
                (Generic<ArangeTyped<_>, IArangeTyped> [this.Type.Type]).Eval this env                
    type internal IArangeTyped =
        abstract Eval: this:Arange -> env:EvalEnv -> ITensor
    type internal ArangeTyped<'T> () =     
        interface IArangeTyped with
            member __.Eval this env =
                Tensor.counting env.Dev (SizeSpec.eval this.Size) :> ITensor       
    let (|Arange|_|) (expr: Expr2) =
        match expr.Op with
        | :? Arange as this -> Some this
        | _ -> None

    /// Argument (placeholder for a variable).
    type VarArg = { Var: Var } with
        interface IOp2 with       
            member this.Check () = ()
            member this.TypeName = this.Var.TypeName
            member this.Shape = this.Var.Shape
            member this.Args = Map.empty
            member this.ReplaceArgs args = this :> _
            member this.SubstSymSizes env = 
                {Var={this.Var with Shape=SymSizeEnv.substShape env this.Var.Shape}} :> _
            member this.CanEvalAllSymSizes = ShapeSpec.canEval this.Var.Shape
            member this.Deriv dOp = Map.empty
            member this.Eval env = 
                env.VarEnv |> VarEnv.get this.Var |> ITensor.transfer env.Dev       
    let (|VarArg|_|) (expr: Expr2) =
        match expr.Op with
        | :? VarArg as this -> Some this.Var
        | _ -> None


[<AutoOpen>]
module UnaryOps =

    /// Unary plus.
    type UnaryPlus = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp
            member this.Eval env = (Args.unaryX env.Args).UnaryPlus ()      
    let (|UnaryPlus|_|) (expr: Expr2) =
        match expr.Op with
        | :? UnaryPlus as this -> Some this.X
        | _ -> None

    /// Negation.
    type Negate = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp
            member this.Eval env = (Args.unaryX env.Args).UnaryMinus ()       
    let (|Negate|_|) (expr: Expr2) =
        match expr.Op with
        | :? Negate as this -> Some this.X
        | _ -> None

    /// Absolute value.
    type Abs = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = failwith "TODO" //Args.unary (dOp * Expr2.padLeft (Expr.signt dOp))
            member this.Eval env = (Args.unaryX env.Args).Abs ()       
    let (|Abs|_|) (expr: Expr2) =
        match expr.Op with
        | :? Abs as this -> Some this.X
        | _ -> None
    
    /// Sign.
    type SignT = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Sgn ()       
    let (|SignT|_|) (expr: Expr2) =
        match expr.Op with
        | :? SignT as this -> Some this.X
        | _ -> None

    /// Logarithm to base exp.
    type Log = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Log ()       
    let (|Log|_|) (expr: Expr2) =
        match expr.Op with
        | :? Log as this -> Some this.X
        | _ -> None

    /// Logarithm to base 10.
    type Log10 = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Log10 ()       
    let (|Log10|_|) (expr: Expr2) =
        match expr.Op with
        | :? Log10 as this -> Some this.X
        | _ -> None

    /// Exponential function.
    type Exp = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Exp ()       
    let (|Exp|_|) (expr: Expr2) =
        match expr.Op with
        | :? Exp as this -> Some this.X
        | _ -> None

    /// Sine.
    type Sin = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Sin ()       
    let (|Sin|_|) (expr: Expr2) =
        match expr.Op with
        | :? Sin as this -> Some this.X
        | _ -> None

    /// Cosine.
    type Cos = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Cos ()       
    let (|Cos|_|) (expr: Expr2) =
        match expr.Op with
        | :? Cos as this -> Some this.X
        | _ -> None

    /// Tangent.
    type Tan = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Tan ()       
    let (|Tan|_|) (expr: Expr2) =
        match expr.Op with
        | :? Tan as this -> Some this.X
        | _ -> None

    /// Inverse sine.
    type Asin = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Asin ()       
    let (|Asin|_|) (expr: Expr2) =
        match expr.Op with
        | :? Asin as this -> Some this.X
        | _ -> None

    /// Inverse cosine.
    type Acos = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Acos ()       
    let (|Acos|_|) (expr: Expr2) =
        match expr.Op with
        | :? Acos as this -> Some this.X
        | _ -> None

    /// Inverse tangent.
    type Atan = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Atan ()       
    let (|Atan|_|) (expr: Expr2) =
        match expr.Op with
        | :? Atan as this -> Some this.X
        | _ -> None

    /// Hyperbolic sine.
    type Sinh = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Sinh ()       
    let (|Sinh|_|) (expr: Expr2) =
        match expr.Op with
        | :? Sinh as this -> Some this.X
        | _ -> None

    /// Hyperbolic cosine.
    type Cosh = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Cosh ()       
    let (|Cosh|_|) (expr: Expr2) =
        match expr.Op with
        | :? Cosh as this -> Some this.X
        | _ -> None

    /// Hyperbolic tangent.
    type Tanh = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Tanh ()       
    let (|Tanh|_|) (expr: Expr2) =
        match expr.Op with
        | :? Tanh as this -> Some this.X
        | _ -> None
        
    /// Square root.
    type Sqrt = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Sqrt ()       
    let (|Sqrt|_|) (expr: Expr2) =
        match expr.Op with
        | :? Sqrt as this -> Some this.X
        | _ -> None

    /// Round towards positive infinity.
    type Ceiling = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Ceiling ()       
    let (|Ceiling|_|) (expr: Expr2) =
        match expr.Op with
        | :? Ceiling as this -> Some this.X
        | _ -> None

    /// Round towards negative infinity.
    type Floor = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Floor ()       
    let (|Floor|_|) (expr: Expr2) =
        match expr.Op with
        | :? Floor as this -> Some this.X
        | _ -> None

    /// Round towards nearest integer.
    type Round = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Round ()       
    let (|Round|_|) (expr: Expr2) =
        match expr.Op with
        | :? Round as this -> Some this.X
        | _ -> None

    /// Round towards zeros.
    type Truncate = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Truncate ()       
    let (|Truncate|_|) (expr: Expr2) =
        match expr.Op with
        | :? Truncate as this -> Some this.X
        | _ -> None

    /// (Batched) matrix inverse.
    type Invert = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = 
                if this.X.NDims < 2 then
                    failwithf "Need at least a matrix to invert but got shape %A" this.X.Shape
                if this.X.Shape.[this.X.NDims-2] .<> this.X.Shape.[this.X.NDims-1] then
                    failwithf "Cannot invert non-square matrix %A along last two axes." this.X.Shape
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Invert ()
    let (|Invert|_|) (expr: Expr2) =
        match expr.Op with
        | :? Invert as this -> Some this.X
        | _ -> None

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    let invert (x: Expr2) =
        {Invert.X=x} |> Expr2

    /// Logical not.
    type Not = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = Check.bool [this.X]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = ~~~~(Args.unaryX env.Args :?> Tensor<bool>) :> ITensor       
    let (|Not|_|) (expr: Expr2) =
        match expr.Op with
        | :? Not as this -> Some this.X
        | _ -> None

    /// Reshape
    type Reshape = { X: Expr2; Shape: ShapeSpec } with
        interface IOp2 with      
            member this.Check () = 
                if ShapeSpec.nElem this.X.Shape .<> ShapeSpec.nElem this.Shape then
                    failwithf "Cannot change number of elements while reshaping from %A to %A." 
                              this.X.Shape this.Shape
            member this.TypeName = this.X.TypeName
            member this.Shape = this.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = 
                { this with Shape = SymSizeEnv.substShape env this.Shape } :> _
            member this.CanEvalAllSymSizes = 
                ShapeSpec.canEval this.Shape
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.reshape (ShapeSpec.eval this.Shape)       
    let (|Reshape|_|) (expr: Expr2) =
        match expr.Op with
        | :? Reshape as this -> Some this
        | _ -> None

    /// Broadcast.
    type DoBroadcast = { X: Expr2; Shape: ShapeSpec } with
        interface IOp2 with      
            member this.Check () = 
                if ShapeSpec.nDim this.X.Shape <> ShapeSpec.nDim this.Shape then
                    failwithf "Tensor of shape %A does not have same number of dimesions as broadcast shape %A."
                              this.X.Shape this.Shape
                for dim in 0 .. (ShapeSpec.nDim this.Shape) - 1 do
                    match this.X.Shape.[dim], this.Shape.[dim] with
                    | SizeSpec.Broadcast, _ -> ()
                    | ssa, ssb when ssa .<> ssb -> 
                        failwithf "Cannot broadcast from %A to %A because non-broadcast dimensions must not change." 
                                  this.X.Shape this.Shape
                    | _ -> ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = 
                { this with Shape = SymSizeEnv.substShape env this.Shape } :> _
            member this.CanEvalAllSymSizes = 
                ShapeSpec.canEval this.Shape
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.broadcastTo (ShapeSpec.eval this.Shape)      
    let (|DoBroadcast|_|) (expr: Expr2) =
        match expr.Op with
        | :? DoBroadcast as this -> Some this
        | _ -> None

    /// Permute the axes.
    type PermuteAxes = {X: Expr2; Permutation: int list} with
        interface IOp2 with      
            member this.Check () = 
                if ShapeSpec.nDim this.X.Shape <> List.length this.Permutation then
                    failwithf "Permutation %A must have same rank as shape %A." this.Permutation this.X.Shape
                if not (Permutation.is this.Permutation) then
                    failwithf "%A is not a valid permutation of an %d-dimensional tensor." 
                              this.Permutation (ShapeSpec.nDim this.X.Shape)
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.permuteAxes this.Permutation
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.permuteAxes this.Permutation
    let (|PermuteAxes|_|) (expr: Expr2) =
        match expr.Op with
        | :? PermuteAxes as this -> Some this
        | _ -> None

    let private dynPrefix = "D"


    /// Read a slice from a tensor.
    type Subtensor = {X: Expr2; Range: SimpleRangesSpec} with
        interface IOp2 with      
            member this.Check () = 
                Check.range this.Range this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = 
                (this.Range, this.X.Shape)
                ||> List.map2 (fun sr shp ->
                    match sr with
                    | SimpleRangeSpec.SymStartSymEnd (s, fo)    -> (fo |? (shp - SizeSpec.one)) + 1L - s
                    | SimpleRangeSpec.DynStartSymSize (_, size) -> size)            
            member this.Args = 
                let xArgs = Args.unary this.X 
                let dynArgs = 
                    SimpleRangesSpec.dynElems dynPrefix this.Range
                    |> Map.map (fun _ v -> v :?> Expr2)
                Map.join xArgs dynArgs
            member this.ReplaceArgs args = 
                let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
                let range = this.Range |> SimpleRangesSpec.replaceDynElems dynPrefix dynArgs               
                {this with X=Args.unaryX args; Range=range} :> _
            member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> _
            member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = 
                // TODO: dynamic range is always copied to host
                let dynVals = 
                    env.Args 
                    |> Map.filter (fun k _ -> k.StartsWith dynPrefix)
                    |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> SizeSpec.fix)
                let range = 
                    this.Range 
                    |> SimpleRangesSpec.resolveDynElems dynPrefix dynVals 
                    |> SimpleRangesSpec.eval
                (Args.unaryX env.Args).[range]
    let (|Subtensor|_|) (expr: Expr2) =
        match expr.Op with
        | :? Subtensor as this -> Some this
        | _ -> None

    /// Replace a slice of a tensor with another tensor.
    type SetSubtensor = {X: Expr2; Y: Expr2; Range: SimpleRangesSpec} with
        interface IOp2 with      
            member this.Check () = 
                Check.sameType [this.X; this.Y]
                Check.range this.Range this.X
                if this.X.NDims <> this.Y.NDims then
                    failwith "Source and target of SetSubtensor must be of same dimensionality."
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape           
            member this.Args = 
                let xyArgs = Args.binary this.X this.Y
                let dynArgs = 
                    SimpleRangesSpec.dynElems dynPrefix this.Range
                    |> Map.map (fun _ v -> v :?> Expr2)
                Map.join xyArgs dynArgs
            member this.ReplaceArgs args = 
                let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
                let range = this.Range |> SimpleRangesSpec.replaceDynElems dynPrefix dynArgs               
                {this with X=Args.binaryX args; Y=Args.binaryY args; Range=range} :> _
            member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> _
            member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = 
                // TODO: dynamic range is always copied to host
                let dynVals = 
                    env.Args 
                    |> Map.filter (fun k _ -> k.StartsWith dynPrefix)
                    |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> SizeSpec.fix)
                let range = 
                    this.Range 
                    |> SimpleRangesSpec.resolveDynElems dynPrefix dynVals 
                    |> SimpleRangesSpec.eval
                let trgt = Args.binaryX env.Args |> ITensor.copy
                trgt.[range] <- Args.binaryY env.Args
                trgt
    let (|SetSubtensor|_|) (expr: Expr2) =
        match expr.Op with
        | :? SetSubtensor as this -> Some this
        | _ -> None

    let internal isSubtensor (expr: Expr2) =
        match expr with
        | Reshape {X=Subtensor {Range=range; X=trgtExpr} as subtensorExpr} ->
            Some (range, subtensorExpr, trgtExpr)
        | _ -> None

    /// Reverses the tensor in the specified dimension.
    type ReverseAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape 
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.reverseAxis this.Axis
    let (|ReverseAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ReverseAxis as this -> Some this
        | _ -> None   

    /// Reverses the tensor in the specified dimension.
    let reverseAxis axis (x: Expr2) =
        {ReverseAxis.Axis=axis; X=x} |> Expr2

    /// Extract the diagonal(s) along the given axes.
    type Diag = {X: Expr2; Axis1: int; Axis2: int} with
        interface IOp2 with      
            member this.Check () = 
                Check.axis this.Axis1 this.X
                Check.axis this.Axis2 this.X 
                if not (this.Axis1 < this.Axis2) then 
                    failwith "First axis for extracting diagonal must come before second axis."
                if this.X.Shape.[this.Axis1] .<> this.X.Shape.[this.Axis2] then
                    failwithf "Cannot extract diagonal along axes %d and %d from non-square tensor with shape %A" 
                              this.Axis1 this.Axis2 this.X.Shape
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis2
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).DiagAxis this.Axis1 this.Axis2
    let (|Diag|_|) (expr: Expr2) =
        match expr.Op with
        | :? Diag as this -> Some this
        | _ -> None   

    /// Extracts the diagonal along the given axes.
    let diagAxis ax1 ax2 (x: Expr2) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        {Diag.Axis1=ax1; Axis2=ax2; Diag.X=x} |> Expr2
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    let diag (x: Expr2) = 
        if x.NDims < 2 then 
            failwithf "Need at least a matrix to extract diagonal but got shape: %A" x.Shape
        x |> diagAxis (x.NDims-2) (x.NDims-1)

    /// Build a matrix with the specified diagonal.
    type DiagMat = {X: Expr2; Axis1: int; Axis2: int} with
        interface IOp2 with      
            member this.Check () = 
                Check.axis this.Axis1 this.X
                if not (0 <= this.Axis2 && this.Axis2 <= this.X.NDims) then
                    failwithf "Cannot build diagonal matrix over non-existant axis %d of tensor with shape %A." 
                              this.Axis2 this.X.Shape
                if not (this.Axis1 < this.Axis2) then 
                    failwith "First axis for building diagonal matrix must come before second axis."
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> List.insert this.Axis2 this.X.Shape.[this.Axis1]
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).DiagMatAxis this.Axis1 this.Axis2
    let (|DiagMat|_|) (expr: Expr2) =
        match expr.Op with
        | :? DiagMat as this -> Some this
        | _ -> None   

    /// Creates a diagonal matrix by duplicating the given dimension.
    let diagMatAxis ax1 ax2 (x: Expr2) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        {DiagMat.Axis1=ax1; Axis2=ax2; X=x} |> Expr2

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    let diagMat (x: Expr2) =
        if x.NDims < 1 then 
            failwithf "Need at least a vector to build diagonal matrix but got shape: %A" x.Shape
        x |> diagMatAxis (x.NDims-1) x.NDims

    /// Sum over specified axis.
    type SumAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).SumAxis this.Axis 
    let (|SumAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? SumAxis as this -> Some this
        | _ -> None

    /// summation over given dimension
    let sumAxis axis x = 
        {SumAxis.Axis=axis; X=x} |> Expr2

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    let sumKeepingAxis axis x =
        x |> sumAxis axis |> Expr2.insertBroadcastAxis axis

    /// summaiton of all elements
    let sum x = 
        x |> Expr2.flatten |> sumAxis 0

    /// Computes the traces along the given axes.
    let traceAxis ax1 ax2 x =
        let tax = if ax1 < ax2 then ax1 else ax1 + 1
        x |> diagAxis ax1 ax2 |> sumAxis tax

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    let trace (x: Expr2) =
        if x.NDims < 2 then
            failwithf "Need at least a matrix for trace but got shape: %A" x.Shape      
        x |> traceAxis (x.NDims-2) (x.NDims-1) 

    /// Product over specified axis.
    type ProductAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).ProductAxis this.Axis
    let (|ProductAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ProductAxis as this -> Some this
        | _ -> None

    /// product over given dimension
    let productAxis axis x = 
        {ProductAxis.Axis=axis; X=x} |> Expr2

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    let productKeepingAxis axis x =
        x |> productAxis axis |> Expr2.insertBroadcastAxis axis

    /// product of all elements
    let product x = 
        x |> Expr2.flatten |> productAxis 0

    /// Maximum over specified axis.
    type MaxAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).MaxAxis this.Axis
    let (|MaxAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? MaxAxis as this -> Some this
        | _ -> None

    /// Maximum over given dimension.
    let maxAxis axis x = 
        {MaxAxis.Axis=axis; X=x} |> Expr2

    /// Maximum over given dimension, while keeping the axis with one (broadcastable) element.
    let maxKeepingAxis axis x =
        x |> maxAxis axis |> Expr2.insertBroadcastAxis axis

    /// Maximum of all elements.
    let max x = 
        x |> Expr2.flatten |> maxAxis 0

    /// Minimum over specified axis.
    type MinAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).MinAxis this.Axis
    let (|MinAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? MinAxis as this -> Some this
        | _ -> None

    /// Minimum over given dimension.
    let minAxis axis x = 
        {MinAxis.Axis=axis; X=x} |> Expr2

    /// Minimum over given dimension, while keeping the axis with one (broadcastable) element.
    let minKeepingAxis axis x =
        x |> minAxis axis |> Expr2.insertBroadcastAxis axis

    /// Minimum of all elements.
    let min x = 
        x |> Expr2.flatten |> minAxis 0

    /// Maximum over specified axis.
    type ArgMaxAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = TypeName.ofType<int64>
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).ArgMaxAxis this.Axis
    let (|ArgMaxAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ArgMaxAxis as this -> Some this
        | _ -> None

    /// Index of maximum over given dimension.
    let argMaxAxis axis x = 
        {ArgMaxAxis.Axis=axis; X=x} |> Expr2

    /// Index of maximum over given dimension, while keeping the axis with one (broadcastable) element.
    let argMaxKeepingAxis axis x =
        x |> argMaxAxis axis |> Expr2.insertBroadcastAxis axis

    /// Minimum over specified axis.
    type ArgMinAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = TypeName.ofType<int64>
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).ArgMinAxis this.Axis
    let (|ArgMinAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ArgMinAxis as this -> Some this
        | _ -> None

    /// Index of minimum over given dimension.
    let argMinAxis axis x = 
        {MinAxis.Axis=axis; X=x} |> Expr2

    /// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
    let argMinKeepingAxis axis x =
        x |> minAxis axis |> Expr2.insertBroadcastAxis axis

    let private listToMap (list: 'a option list) =
        list 
        |> List.indexed 
        |> List.choose (function 
                        | i, Some v -> Some (sprintf "I%d" i, v)
                        | _, None -> None)
        |> Map.ofList

    let private mapToList length (map: Map<string, 'a>) =
        [0 .. length-1]
        |> List.map (fun i -> map |> Map.tryFind (sprintf "I%d" i))

    /// Select elements according to the specified index tensors
    type Gather = {X: Expr2; Indices: Expr2 option list} with
        interface IOp2 with      
            member this.Check () = 
                if this.X.NDims <> this.Indices.Length then
                    failwithf "Gather source has shape %A but %d index tensors were specified." 
                              this.X.Shape this.Indices.Length
                let trgtShape =
                    match this.Indices |> List.tryPick id with
                    | Some idx -> idx.Shape
                    | None -> failwith "Gather needs at least one specified (not None) index expression."  
                for dim, idx in List.indexed this.Indices do
                    match idx with
                    | Some idx when idx.DataType <> typeof<int64> ->
                        failwithf "All index tensors for gather must be of type int64, but got type %A." idx.DataType
                    | Some idx when idx.Shape <> trgtShape ->
                        failwithf "All gather indices must have equal shape, but got shapes %A."
                                  (this.Indices |> List.map (Option.map Expr2.shape))
                    | None when dim >= ShapeSpec.nDim trgtShape ->
                        failwithf "Gather index dimensions beyond the number of target dimensions \
                                   must not be None."
                    | _ -> ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.Indices |> List.pick id |> Expr2.shape
            member this.Args = 
                let idxArgs = this.Indices |> listToMap                
                let xArgs = Args.unary this.X
                Map.join idxArgs xArgs
            member this.ReplaceArgs args =                
                {this with X=Args.unaryX args; Indices=mapToList this.Indices.Length args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = 
                let vIndices = env.Args |> mapToList this.Indices.Length
                (Args.unaryX env.Args).Gather vIndices 
    let (|Gather|_|) (expr: Expr2) =
        match expr.Op with
        | :? Gather as this -> Some this
        | _ -> None    

    /// Select elements according to the specified index tensors.
    let gather (indices: Expr2 option list) (x: Expr2) =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "Gather needs at least one specified index tensor."
        let bcSomeIndices = Expr2.broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        {Gather.Indices=bcIndices; X=x} |> Expr2

    /// Disperses elements according to the specified index tensors.
    type Scatter = {X: Expr2; Indices: Expr2 option list; Shape: ShapeSpec} with
        interface IOp2 with      
            member this.Check () = 
                for dim, idx in List.indexed this.Indices do
                    match idx with
                    | Some idx when idx.DataType <> typeof<int64> ->
                        failwithf "All index tensors for scatter must be of type int64, but got type %A." idx.DataType
                    | Some idx when idx.Shape <> this.X.Shape ->
                        failwithf "All scatter indices must have shape of source %A, but got %A." 
                                  this.X.Shape (this.Indices |> List.map (Option.map Expr2.shape))
                    | None when dim >= this.X.NDims ->
                        failwithf "Scatter index dimensions beyond the number of source dimensions \
                                   must not be None."
                    | _ -> ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.Shape
            member this.Args = 
                let idxArgs = this.Indices |> listToMap                
                let xArgs = Args.unary this.X
                Map.join idxArgs xArgs
            member this.ReplaceArgs args =                
                {this with X=Args.unaryX args; Indices=mapToList this.Indices.Length args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = 
                let vIndices = env.Args |> mapToList this.Indices.Length
                (Args.unaryX env.Args).Scatter vIndices (ShapeSpec.eval this.Shape)
    let (|Scatter|_|) (expr: Expr2) =
        match expr.Op with
        | :? Scatter as this -> Some this
        | _ -> None   

    /// Disperses elements according to the specified index tensors.
    let scatter (indices: Expr2 option list) (trgtShp: ShapeSpec) (x: Expr2) =
        let indices = indices |> List.map (Option.map (Expr2.broadcastToShape x.Shape))
        {Scatter.Indices=indices; Shape=trgtShp; X=x} |> Expr2

    /// Store value to variable.
    type Store = {X: Expr2; Var: Var} with
        interface IOp2 with       
            member this.Check () = 
                if this.X.TypeName <> this.Var.TypeName then
                    failwithf "Cannot store expression of type %A into variable of type %A."
                              this.X.TypeName this.Var.TypeName
                if not (ShapeSpec.equalWithoutBroadcastability this.X.Shape this.Var.Shape) then
                    failwithf "Cannot store expression of shape %A into variable of shape %A." 
                              this.X.Shape this.Var.Shape                
            member this.TypeName = this.X.TypeName
            member this.Shape = ShapeSpec.emptyVector
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = 
                {this with X=Args.unaryX args} :> _
            member this.SubstSymSizes env = 
                {this with Var={this.Var with Shape=SymSizeEnv.substShape env this.Var.Shape}} :> _
            member this.CanEvalAllSymSizes = ShapeSpec.canEval this.Var.Shape
            member this.Deriv dOp = Map.empty
            member this.Eval env = 
                let tv = env.VarEnv |> VarEnv.get this.Var 
                let v = Args.unaryX env.Args                
                tv.CopyFrom (v.Transfer tv.Dev)
                v.ZerosOfSameType v.Dev [0L]
    let (|Store|_|) (expr: Expr2) =
        match expr.Op with
        | :? Store as this -> Some this
        | _ -> None

    /// Sets the Jacobian of its argument to zero when calculating derivatives.
    type AssumeZeroDeriv = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp // TODO
            member this.Eval env = Args.unaryX env.Args
    let (|AssumeZeroDeriv|_|) (expr: Expr2) =
        match expr.Op with
        | :? AssumeZeroDeriv as this -> Some this.X
        | _ -> None
    
    /// Nullifies the Jacobian of its argument when calculating derivatives.
    let assumeZeroDeriv x =
        {AssumeZeroDeriv.X=x} |> Expr2

    /// Sets the Jacobian of its argument to zero when calculating derivatives.
    type AssumeDeriv = {Deriv: Expr2; X: Expr2} with
        interface IOp2 with      
            member this.Check () = 
                Check.sameType [this.Deriv; this.X]
                if this.Deriv.NDims <> 2 then
                    failwithf "Jacobian shape %A must be two-dimensional." this.Deriv.Shape
                if this.Deriv.Shape.[1] <> this.X.NElems then
                    failwithf "Jacobian shape %A must have %A elements in second dimension." 
                        this.Deriv.Shape this.X.NElems
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args =                 
                Map.join (Args.unary this.X) (Map ["Deriv", this.Deriv])                
            member this.ReplaceArgs args = 
                {this with Deriv=args.["Deriv"]; X=Args.unaryX args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp // TODO
            member this.Eval env = Args.unaryX env.Args
    let (|AssumeDeriv|_|) (expr: Expr2) =
        match expr.Op with
        | :? AssumeDeriv as this -> Some this
        | _ -> None
    
    /// Assumes the specified Jacobian when calculating derivatives.
    let assumeDeriv deriv x =
        {AssumeDeriv.Deriv=deriv; X=x} |> Expr2

    /// Annotation (no influence on value).
    type Annotated = {Label: System.IComparable; X: Expr2} with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp 
            member this.Eval env = Args.unaryX env.Args                  
    let (|Annotated|_|) (expr: Expr2) =
        match expr.Op with
        | :? Annotated as this -> Some this
        | _ -> None

    /// Annotated expression (no influence on value).
    let annotate label x = 
        {Annotated.Label=label; X=x} |> Expr2
    
    /// Prints the value together with the given label.
    type Print = {Label: string; X: Expr2} with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp 
            member this.Eval env = 
                let v = Args.unaryX env.Args
                printfn "%s=\n%A\n" this.Label v
                v                            
    let (|Print|_|) (expr: Expr2) =
        match expr.Op with
        | :? Print as this -> Some this
        | _ -> None
    
    /// Print the result with the given label when evaluated.
    let print label x =
        {Print.Label=label; X=x} |> Expr2

    /// Dumps the result into the given dataset in the active HDF5 dump file.
    type Dump = {Dataset: string; X: Expr2} with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp 
            member this.Eval env = 
                let v = Args.unaryX env.Args
                Dump.dumpValue this.Dataset v
                v                            
    let (|Dump|_|) (expr: Expr2) =
        match expr.Op with
        | :? Dump as this -> Some this
        | _ -> None

    /// Dumps the result into the given dataset in the active HDF5 dump file.
    let dump dataset x =
        {Dump.Dataset=dataset; X=x} |> Expr2

    /// If the value contains NaNs or infinities, outputs their location and 
    /// stops the computation.
    type CheckFinite = {Label: string; X: Expr2} with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp 
            member this.Eval env = 
                let v = Args.unaryX env.Args
                if not (v.AllFinite ()) then
                    printfn "Infinity or NaN encountered in %s with value:\n%A" this.Label v
                    failwithf "Infinity or NaN encountered in %s." this.Label
                v                            
    let (|CheckFinite|_|) (expr: Expr2) =
        match expr.Op with
        | :? CheckFinite as this -> Some this
        | _ -> None

    /// If the value contains NaNs or infinities, outputs their location and 
    /// stops the computation.
    let checkFinite label x =
        {CheckFinite.Label=label; X=x} |> Expr2

    /// Accesses the specified channel of a multi-channnel expression.
    type Channel = {Channel: string; X: MultiChannelExpr} with
        interface IOp2 with      
            member this.Check () = 
                if not (this.X.Channels |> List.contains this.Channel) then
                    failwithf "Multi-channel expression does not contain channel %A, it contains channels %A." 
                              this.Channel this.X.Channels
            member this.TypeName = this.X.TypeNames.[this.Channel]
            member this.Shape = this.X.Shapes.[this.Channel]
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary dOp 
            member this.Eval env = 
                let v = Args.unaryX env.Args
                if not (v.AllFinite ()) then
                    printfn "Infinity or NaN encountered in %s with value:\n%A" this.Label v
                    failwithf "Infinity or NaN encountered in %s." this.Label
                v                            
    let (|Channel|_|) (expr: Expr2) =
        match expr.Op with
        | :? Channel as this -> Some this
        | _ -> None

[<AutoOpen>]
module BinaryOps =

    /// Addition.
    type Add = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Add (Args.binaryY env.Args)      
    let (|Add|_|) (expr: Expr2) =
        match expr.Op with
        | :? Add as this -> Some (this.X, this.Y)
        | _ -> None

    /// Subtraction.
    type Subtract = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Subtract (Args.binaryY env.Args)       
    let (|Subtract|_|) (expr: Expr2) =
        match expr.Op with
        | :? Subtract as this -> Some (this.X, this.Y)
        | _ -> None

    /// Multiplication.
    type Multiply = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Multiply (Args.binaryY env.Args)      
    let (|Multiply|_|) (expr: Expr2) =
        match expr.Op with
        | :? Multiply as this -> Some (this.X, this.Y)
        | _ -> None

    /// Division.
    type Divide = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Divide (Args.binaryY env.Args)       
    let (|Divide|_|) (expr: Expr2) =
        match expr.Op with
        | :? Divide as this -> Some (this.X, this.Y)
        | _ -> None

    /// Exponentiation.
    type Pow = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Pow (Args.binaryY env.Args)       
    let (|Pow|_|) (expr: Expr2) =
        match expr.Op with
        | :? Pow as this -> Some (this.X, this.Y)
        | _ -> None

    /// Modulo.
    type Modulo = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Modulo (Args.binaryY env.Args)       
    let (|Modulo|_|) (expr: Expr2) =
        match expr.Op with
        | :? Modulo as this -> Some (this.X, this.Y)
        | _ -> None

    /// Elementwise maximum.
    type MaxElemwise = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).MaxElemwise (Args.binaryY env.Args)       
    let (|MaxElemwise|_|) (expr: Expr2) =
        match expr.Op with
        | :? MaxElemwise as this -> Some (this.X, this.Y)
        | _ -> None

    /// Elementwise minimum.
    type MinElemwise = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).MinElemwise (Args.binaryY env.Args)       
    let (|MinElemwise|_|) (expr: Expr2) =
        match expr.Op with
        | :? MinElemwise as this -> Some (this.X, this.Y)
        | _ -> None

    /// Logical And.
    type And = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = 
                (Args.binaryX env.Args :?> Tensor<bool>) &&&& (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       
    let (|And|_|) (expr: Expr2) =
        match expr.Op with
        | :? And as this -> Some (this.X, this.Y)
        | _ -> None

    /// Logical Or.
    type Or = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = 
                (Args.binaryX env.Args :?> Tensor<bool>) |||| (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       
    let (|Or|_|) (expr: Expr2) =
        match expr.Op with
        | :? Or as this -> Some (this.X, this.Y)
        | _ -> None

    /// Logical Xor.
    type Xor = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = 
                (Args.binaryX env.Args :?> Tensor<bool>) ^^^^ (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       
    let (|Xor|_|) (expr: Expr2) =
        match expr.Op with
        | :? Xor as this -> Some (this.X, this.Y)
        | _ -> None

    /// Equal.
    type Equal = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = TypeName.ofType<bool>
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Equal (Args.binaryY env.Args)       
    let (|Equal|_|) (expr: Expr2) =
        match expr.Op with
        | :? Equal as this -> Some (this.X, this.Y)
        | _ -> None

    /// Not equal.
    type NotEqual = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = TypeName.ofType<bool>
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).NotEqual (Args.binaryY env.Args)       
    let (|NotEqual|_|) (expr: Expr2) =
        match expr.Op with
        | :? NotEqual as this -> Some (this.X, this.Y)
        | _ -> None

    /// Less than.
    type Less = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = TypeName.ofType<bool>
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Less (Args.binaryY env.Args)       
    let (|Less|_|) (expr: Expr2) =
        match expr.Op with
        | :? Less as this -> Some (this.X, this.Y)
        | _ -> None

    /// Less then or equal.
    type LessOrEqual = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = TypeName.ofType<bool>
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).LessOrEqual (Args.binaryY env.Args)       
    let (|LessOrEqual|_|) (expr: Expr2) =
        match expr.Op with
        | :? LessOrEqual as this -> Some (this.X, this.Y)
        | _ -> None

    /// Greater than.
    type Greater = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = TypeName.ofType<bool>
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Greater (Args.binaryY env.Args)       
    let (|Greater|_|) (expr: Expr2) =
        match expr.Op with
        | :? Greater as this -> Some (this.X, this.Y)
        | _ -> None

    /// Greater than or equal.
    type GreaterOrEqual = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = TypeName.ofType<bool>
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).GreaterOrEqual (Args.binaryY env.Args)       
    let (|GreaterOrEqual|_|) (expr: Expr2) =
        match expr.Op with
        | :? GreaterOrEqual as this -> Some (this.X, this.Y)
        | _ -> None

    /// Dot product.
    type Dot = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = 
                Check.sameType [this.X; this.Y]
                let sa, sb = this.X.Shape, this.Y.Shape
                match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 2, 2 -> 
                    if sa.[1] .<> sb.[0] then
                        failwithf "Incompatible shapes for dot product: %A and %A." sa sb
                | na, nb when na = nb -> 
                    if sa.[na-1] .<> sb.[nb-2] || 
                        [0 .. na-3] |> List.exists (fun n -> sa.[n] .<> sb.[n]) then
                            failwithf "Incompatible shapes for batched dot product: %A and %A." sa sb
                | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  
            member this.TypeName = this.X.TypeName
            member this.Shape =
                let sa, sb = this.X.Shape, this.Y.Shape
                match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 2, 2 -> ShapeSpec.matrix sa.[0] sb.[1]
                | na, nb when na=nb -> sa.[0 .. na-2] @ [sb.[nb-1]]
                | _ -> failwithf "Invalid dot product shapes: %A and %A." sa sb
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).Dot (Args.binaryY env.Args)       
    let (|Dot|_|) (expr: Expr2) =
        match expr.Op with
        | :? Dot as this -> Some (this.X, this.Y)
        | _ -> None

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    let dot (a: Expr2) (b: Expr2) =
        let sa, sb = a.Shape, b.Shape
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
        | 1, 1 -> 
            // vector-vector dot product
            sum (a * b)
        | 2, 1 -> 
            // matrix-vector dot product
            let bm = b |> Expr2.reshape (ShapeSpec.padRight sb)
            {Dot.X=a; Y=bm} |> Expr2 |> Expr2.reshape [sa.[0]]
        | 2, 2 -> 
            // matrix-matrix dot product
            {Dot.X=a; Y=b} |> Expr2
        | na, nb when na = nb -> 
            // batched matrix-matrix dot product
            let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa sb
            let ba = a |> Expr2.broadcast bsa
            let bb = b |> Expr2.broadcast bsb    
            {Dot.X=ba; Y=bb} |> Expr2
        | na, nb when na = nb + 1 ->
            // batched matrix-vector dot product
            let psb = ShapeSpec.padRight sb
            let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa psb
            let ba = a |> Expr2.broadcast bsa
            let bb = b |> Expr2.reshape psb |> Expr2.broadcast bsb    
            {Dot.X=ba; Y=bb} |> Expr2 |> Expr2.reshape bsa.[0 .. na-2]
        | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  

    /// Tensor product.
    type TensorProduct = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = 
                Check.sameType [this.X; this.Y]
                let sa, sb = this.X.Shape, this.Y.Shape
                if ShapeSpec.nDim sa <> ShapeSpec.nDim sb then
                    failwithf "Cannot compute tensor product between tensors of shapes %A and %A." sa sb
            member this.TypeName = this.X.TypeName
            member this.Shape = 
                List.map2 (*) this.X.Shape this.Y.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).TensorProduct (Args.binaryY env.Args)       
    let (|TensorProduct|_|) (expr: Expr2) =
        match expr.Op with
        | :? TensorProduct as this -> Some (this.X, this.Y)
        | _ -> None

    let tensorProduct (x: Expr2) (y: Expr2) =
        {TensorProduct.X=x; Y=y} |> Expr2


    /// Element-wise if-then-else.
    type IfThenElse = {Cond: Expr2; IfTrue: Expr2; IfFalse: Expr2} with
        interface IOp2 with       
            member this.Check () = 
                Check.sameType [this.IfTrue; this.IfFalse]
                Check.bool [this.Cond]
                Check.sameShape [this.Cond; this.IfTrue; this.IfFalse]
            member this.TypeName = this.IfTrue.TypeName
            member this.Shape = this.IfTrue.Shape
            member this.Args = 
                Map ["Cond", this.Cond
                     "IfTrue", this.IfTrue
                     "IfFalse", this.IfFalse]
            member this.ReplaceArgs args = 
                {this with Cond=args.["Cond"]
                           IfTrue=args.["IfTrue"]
                           IfFalse=args.["IfFalse"]} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = 
                env.Args.["IfTrue"].IfThenElse env.Args.["IfFalse"] env.Args.["Cond"]
    let (|IfThenElse|_|) (expr: Expr2) =
        match expr.Op with
        | :? IfThenElse as this -> Some this
        | _ -> None

    /// Elementwise uses elements from `ifTrue` if `cond` is true for that element, otherwise elements from `ifFalse`.
    let ifThenElse (cond: Expr2) (ifTrue: Expr2) (ifFalse: Expr2) =
        let shps = [cond.Shape; ifTrue.Shape; ifFalse.Shape]
        let pShps = ShapeSpec.padToSameMany shps
        let bcShps = ShapeSpec.broadcastToSameMany false pShps           
        match pShps, bcShps with
        | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
            let condBc = cond |> Expr2.reshape condPShp |> Expr2.broadcast condBcShp
            let ifTrueBc = ifTrue |> Expr2.reshape ifTruePShp |> Expr2.broadcast ifTrueBcShp
            let ifFalseBc = ifFalse |> Expr2.reshape ifFalsePShp |> Expr2.broadcast ifFalseBcShp
            {IfThenElse.Cond=condBc; IfTrue=ifTrueBc; IfFalse=ifFalseBc} |> Expr2
        | _ -> failwith "impossible"


[<AutoOpen>]
module NaryOps = 

    /// Discards the results of all arguments.
    type Discard = {Xs: Expr2 list} with
        interface IOp2 with       
            member this.Check () = ()
            member this.TypeName = TypeName.ofType<int32>
            member this.Shape = ShapeSpec.emptyVector
            member this.Args = Args.nary this.Xs
            member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp // TODO
            member this.Eval env = HostTensor.zeros<int32> [0L] :> ITensor
    let (|Discard|_|) (expr: Expr2) =
        match expr.Op with
        | :? Discard as this -> Some (this.Xs)
        | _ -> None

    /// Discards the results of all arguments.
    let discard (xs: Expr2 list) =
        {Discard.Xs=xs} |> Expr2

    /// Build tensor using numeric ranges.
    type BuildTensor = {Shape: ShapeSpec; Ranges: BaseRangesSpec list; Xs: Expr2 list} with
        interface IOp2 with       
            member this.Check () = 
                Check.sameType this.Xs
                if this.Ranges.Length <> this.Xs.Length then
                    failwithf "BuildTensor ranges must match arguments, but got %d ranges and %d arguments."
                               this.Ranges.Length this.Xs.Length
                match ShapeSpec.tryEval this.Shape with
                | Some shp ->
                    for rng, arg in List.zip this.Ranges this.Xs do
                        if rng.Length <> shp.Length then
                            failwithf "BuildTensor range %A has wrong dimensionality for shape %A." rng shp
                        for (start, stop), size, argSize in List.zip3 rng shp arg.Shape do
                            if argSize <> stop - start + 1L then
                                failwithf "BuildTensor range %A is invalid for argument of shape %A." rng arg.Shape
                            match SizeSpec.tryEval start, SizeSpec.tryEval stop with
                            | Some start, Some stop when not (0L <= start && start < size && 0L <= stop && 
                                                                stop < size && start <= stop) ->
                                failwithf "BuildTensor range %A is invalid for shape %A." rng shp
                            | _, _ -> ()
                | None -> ()       
            member this.TypeName = this.Xs.Head.TypeName
            member this.Shape = this.Shape
            member this.Args = Args.nary this.Xs
            member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
            member this.SubstSymSizes env = 
                let sSize = SizeSpec.substSymbols env
                {this with Shape=ShapeSpec.substSymbols env this.Shape
                           Ranges=this.Ranges |> List.map (List.map (fun (f,l) -> sSize f, sSize l))} :> _
            member this.CanEvalAllSymSizes = 
                ShapeSpec.canEval this.Shape &&
                List.forall BaseRangesSpec.canEval this.Ranges
            member this.Deriv dOp = Args.binary dOp dOp // TODO
            member this.Eval env = 
                let vs = Args.naryXs env.Args
                let trgt = vs.Head.ZerosOfSameType vs.Head.Dev (ShapeSpec.eval this.Shape)
                for rng, e in List.zip this.Ranges vs do                            
                    let aryRng = rng |> List.map (fun (first, last) -> 
                        Rng.Rng (Some (SizeSpec.eval first), Some (SizeSpec.eval last)))
                    trgt.[aryRng] <- e 
                trgt
    let (|BuildTensor|_|) (expr: Expr2) =
        match expr.Op with
        | :? BuildTensor as this -> Some this
        | _ -> None

    /// Build tensor from numeric ranges.
    let internal buildTensor shape ranges xs =
        {BuildTensor.Shape=shape; Ranges=ranges; Xs=xs} |> Expr2
    
    /// Elementwise calculated tensor.
    type Elements = {Shape: ShapeSpec; ElemExpr: Elem.Expr; Xs: Expr2 list} with
        interface IOp2 with       
            member this.Check () = 
                let tns = this.Xs |> List.map Expr2.typeName
                let ss = this.Xs |> List.map Expr2.shape
                Elem.Expr.check this.ElemExpr |> ignore
                Elem.Expr.checkCompatibility this.ElemExpr ss tns this.Shape   
            member this.TypeName = Elem.Expr.typeName this.ElemExpr
            member this.Shape = this.Shape
            member this.Args = Args.nary this.Xs
            member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
            member this.SubstSymSizes env = 
                let sSize = SizeSpec.substSymbols env
                {this with Shape=ShapeSpec.substSymbols env this.Shape
                           ElemExpr=Elem.Expr.substSymSizes env this.ElemExpr} :> _
            member this.CanEvalAllSymSizes = 
                ShapeSpec.canEval this.Shape &&
                Elem.Expr.canEvalAllSymSizes this.ElemExpr
            member this.Deriv dOp = Args.binary dOp dOp // TODO
            member this.Eval env = 
                let esv = Args.naryXs env.Args
                let nResShape = ShapeSpec.eval this.Shape
                Elem.Interpreter.evalUntyped this.ElemExpr esv nResShape 
    let (|Elements|_|) (expr: Expr2) =
        match expr.Op with
        | :? Elements as this -> Some this
        | _ -> None

    /// Calculates a tensor elementwise using the given element expression and result shape.
    let elements shape elemExpr xs =
        {Elements.Shape=shape; ElemExpr=elemExpr; Xs=xs} |> Expr2

    /// Elementwise interpolation using a value table.
    type Interpolate = {Interpolator: Interpolator; Xs: Expr2 list} with
        interface IOp2 with       
            member this.Check () = 
                Check.sameType this.Xs
                let nDims = this.Interpolator.MinArg.Length
                if nDims < 1 then
                    failwith "Interpolator must be at least one-dimensional."
                if this.Interpolator.MaxArg.Length <> nDims || this.Interpolator.Outside.Length <> nDims ||
                    this.Interpolator.Resolution.Length <> nDims then
                        failwith "MinArg, MaxArg, Resolution and Outside have inconsistent lengths."
                if this.Xs.Length <> nDims then
                    failwith "Number of arguments does not match dimensionality of interpolator."
                if not ((this.Interpolator.MinArg, this.Interpolator.MaxArg) 
                    ||> List.forall2 (fun mi ma -> conv<float> mi < conv<float> ma)) then
                        failwith "MinArg of interpolator must be smaller than MaxArg."
                if this.Interpolator.Resolution |> List.exists ((>) 0.0) then
                    failwith "Resolution of interpolator must be positive."
                for x in this.Xs do 
                    if not (ShapeSpec.equalWithoutBroadcastability x.Shape this.Xs.Head.Shape) then
                        failwithf "All arguments to interpolator must have equal shape but got shapes %A and %A."
                                  this.Xs.Head.Shape x.Shape
            member this.TypeName = this.Xs.Head.TypeName
            member this.Shape = this.Xs.Head.Shape
            member this.Args = Args.nary this.Xs
            member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp // TODO
            member this.Eval env = 
                let esv = Args.naryXs env.Args
                Interpolator.interpolateUntyped this.Interpolator esv
    let (|Interpolate|_|) (expr: Expr2) =
        match expr.Op with
        | :? Interpolate as this -> Some this
        | _ -> None

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate interpolator xs =
        let xs = Expr2.broadcastToSameMany xs
        {Interpolate.Interpolator=interpolator; Xs=xs} |> Expr2

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate1D interpolator x =
        interpolate interpolator [x]

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate2D interpolator x y =
        interpolate interpolator [x; y]

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate3D interpolator x y z =
        interpolate interpolator [x; y; z]


[<AutoOpen>]
module LoopOps = 

    /// a slice of an argument to the loop
    type SequenceArgSlice = {
        /// the index of the argument
        ArgIdx:     int
        /// the dimension the loop is performed over
        SliceDim:   int
    }

    /// references a loop channel of a previous iteration
    type PreviousChannel = {
        /// the channel to use
        Channel:       string
        /// the delay, must be at least one
        Delay:         SizeSpec
        /// the index of the argument specifying the initial values
        InitialArg:    int
    }

    /// a loop variable value specification
    type LoopInput = 
        /// provides the loop argument to all loop iterations
        | ConstArg of argIdx:int
        /// provides a slice of the loop argument to each loop iteration
        | SequenceArgSlice of SequenceArgSlice
        /// provides the value of a loop channel from a previous loop iteration
        | PreviousChannel of PreviousChannel
        /// provides the index of the current loop iteration (zero-based)
        | IterationIndex
        /// provides the number of remaining loop iterations after this iteration
        | IterationsRemaining

    /// the value of a loop channel
    type LoopValue = {
        /// the expression to compute the loop channel;
        /// it may only use variables defined in LoopSpecT.Vars
        Expr:       Expr2
        /// the dimension to concatenate the results along to produce the loop output
        SliceDim:   int
    }


    /// Elementwise interpolation using a value table.
    type Loop = {
        /// number of loop iterations
        Length:     SizeSpec
        /// specifies the values of the variables used in the channel value expressions,
        /// i.e. LoopValueT.Expr
        Vars:       Map<Var, LoopInput>   
        /// specifies the values of the loop channels
        Channels:   Map<string, LoopValue>
        /// inputs
        Xs:         Expr2 list
    } with
        interface IMultiChannelOp with       
            member this.Check () = ()
            member this.Channels = 
                this.Channels |> Map.toList |> List.map fst
            member this.TypeNames = 
                this.Channels |> Map.map (fun _ lv -> lv.Expr.TypeName)
            member this.Shapes = 
                this.Channels |> Map.map (fun _ lv -> lv.Expr.Shape)
            member this.Args = Args.nary this.Xs
            member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
            member this.SubstSymSizes env = this :> _
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = failwith "TODO" // TODO
            member this.Eval env = 
                failwith "TODO"
    let (|Loop|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Loop as this -> Some this
        | _ -> None


