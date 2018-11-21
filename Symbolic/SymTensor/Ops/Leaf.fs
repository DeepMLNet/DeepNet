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

module Args =
    //let leaf : ArgsMap = Map.empty

    let unary x : ArgsMap = Map ["X", x]
    let unaryX (am: Map<string, _>) = am.["X"]

    let binary x y : ArgsMap = Map ["X", x; "Y", y]
    let binaryX (am: Map<string, _>) = am.["X"]
    let binaryY (am: Map<string, _>) = am.["Y"]


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
            member this.ReplaceArgs args = this :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = this :> IOp2
            member this.SubstSymSizes env = { Value = SymSizeEnv.subst env this.Value } :> IOp2
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
            member this.ReplaceArgs args = this :> IOp2
            member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> IOp2
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
            member this.ReplaceArgs args = this :> IOp2
            member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> IOp2
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
            member this.ReplaceArgs args = this :> IOp2
            member this.SubstSymSizes env = 
                {Var={this.Var with Shape=SymSizeEnv.substShape env this.Var.Shape}} :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Truncate ()       
    let (|Truncate|_|) (expr: Expr2) =
        match expr.Op with
        | :? Truncate as this -> Some this.X
        | _ -> None

    /// Logical not.
    type Not = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = Check.bool [this.X]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = 
                { this with Shape = SymSizeEnv.substShape env this.Shape } :> IOp2
            member this.CanEvalAllSymSizes = 
                ShapeSpec.canEval this.Shape
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.reshape (ShapeSpec.eval this.Shape)       
    let (|Reshape|_|) (expr: Expr2) =
        match expr.Op with
        | :? Reshape as this -> Some this.X
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = 
                { this with Shape = SymSizeEnv.substShape env this.Shape } :> IOp2
            member this.CanEvalAllSymSizes = 
                ShapeSpec.canEval this.Shape
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.broadcastTo (ShapeSpec.eval this.Shape)      
    let (|DoBroadcast|_|) (expr: Expr2) =
        match expr.Op with
        | :? DoBroadcast as this -> Some this.X
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.permuteAxes this.Permutation
    let (|PermuteAxes|_|) (expr: Expr2) =
        match expr.Op with
        | :? PermuteAxes as this -> Some this
        | _ -> None

    /// Read a slice from a tensor.
    type Subtensor = {X: Expr2; Range: SimpleRangesSpec} with
        interface IOp2 with      
            member this.Check () = () // TODO?
            member this.TypeName = this.X.TypeName
            member this.Shape = 
                (this.Range, this.X.Shape)
                ||> List.map2 (fun sr shp ->
                    match sr with
                    | SimpleRangeSpec.SymStartSymEnd (s, fo)    -> (fo |? (shp - SizeSpec.one)) + 1L - s
                    | SimpleRangeSpec.DynStartSymSize (_, size) -> size)            
            member this.Args = Args.unary this.X // TODO: expose dynamic range expressions
            member this.ReplaceArgs args = {this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> IOp2
            member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = 
                // TODO: use argument values to evaluate SRS
                let rng = this.Range |> SimpleRangesSpec.eval (fun _ -> failwith "TODO")
                (Args.unaryX env.Args).[rng]
    let (|Subtensor|_|) (expr: Expr2) =
        match expr.Op with
        | :? Subtensor as this -> Some this
        | _ -> None

    /// Reverses the tensor in the specified dimension.
    type ReverseAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape 
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = {this with X = Args.unaryX args} :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
    type GetDiag = {X: Expr2; Axes: int * int} with
        interface IOp2 with      
            member this.Check () = 
                let ax1, ax2 = this.Axes
                Check.axis ax1 this.X
                Check.axis ax2 this.X 
                if not (ax1 < ax2) then 
                    failwith "First axis for extracting diagonal must come before second axis."
                if this.X.Shape.[ax1] .<> this.X.Shape.[ax2] then
                    failwithf "Cannot extract diagonal along axes %d and %d from non-square tensor with shape %A" 
                              ax1 ax2 this.X.Shape
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis (snd this.Axes)
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = {this with X = Args.unaryX args} :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args) |> ITensor.diagAxis this.Axis
    let (|ReverseAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ReverseAxis as this -> Some this
        | _ -> None   

    /// Sum over specified axis.
    type SumAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).SumAxis this.Axis 
    let (|SumAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? SumAxis as this -> Some this.X
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

    /// Product over specified axis.
    type ProductAxis = {X: Expr2; Axis: int} with
        interface IOp2 with      
            member this.Check () = Check.axis this.Axis this.X
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).ProductAxis this.Axis
    let (|ProductAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ProductAxis as this -> Some this.X
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).MaxAxis this.Axis
    let (|MaxAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? MaxAxis as this -> Some this.X
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).MinAxis this.Axis
    let (|MinAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? MinAxis as this -> Some this.X
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).ArgMaxAxis this.Axis
    let (|ArgMaxAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ArgMaxAxis as this -> Some this.X
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
            member this.ReplaceArgs args = { this with X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).ArgMinAxis this.Axis
    let (|ArgMinAxis|_|) (expr: Expr2) =
        match expr.Op with
        | :? ArgMinAxis as this -> Some this.X
        | _ -> None

    /// Index of minimum over given dimension.
    let argMinAxis axis x = 
        {MinAxis.Axis=axis; X=x} |> Expr2

    /// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
    let argMinKeepingAxis axis x =
        x |> minAxis axis |> Expr2.insertBroadcastAxis axis


    

[<AutoOpen>]
module BinaryOps =

    /// Addition.
    type Add = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
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
            member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.binary dOp dOp
            member this.Eval env = (Args.binaryX env.Args).TensorProduct (Args.binaryY env.Args)       
    let (|TensorProduct|_|) (expr: Expr2) =
        match expr.Op with
        | :? TensorProduct as this -> Some (this.X, this.Y)
        | _ -> None

    let tensorProduct (x: Expr2) (y: Expr2) =
        {TensorProduct.X=x; Y=y} |> Expr2

