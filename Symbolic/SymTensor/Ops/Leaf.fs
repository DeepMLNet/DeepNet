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



module Args =
    //let leaf : ArgsMap = Map.empty

    let unary x : ArgsMap = Map ["X", x]
    let unaryX (am: Map<string, _>) = am.["X"]

    let binary x y : ArgsMap = Map ["X", x; "Y", y]
    let binaryX (am: Map<string, _>) = am.["X"]
    let binaryY (am: Map<string, _>) = am.["Y"]


[<AutoOpen>]
module private DerivUtils =
        /// Expands the second dimension of the the Jacobian into the shape of this expression
        let expand (dOp: Expr2) (expr: Expr2) = 
            let funElems = dOp.Shape.[0]
            dOp |> Expr2.reshape (funElems :: expr.Shape)

        /// Flattens all but the first dimension into one dimension
        let collapse (g: Expr2) =
            let wrtElems = g.Shape.[1..] |> ShapeSpec.nElem
            g |> Expr2.reshape [funElems; wrtElems]


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

    /// Negation.
    type Negate = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.ReplaceArgs args = { Negate.X = Args.unaryX args } :> IOp2
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
            member this.ReplaceArgs args = { Abs.X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary (dOp * Expr2.padLeft (Expr.signt dOp))
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
            member this.ReplaceArgs args = { SignT.X = Args.unaryX args } :> IOp2
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
            member this.ReplaceArgs args = { Log.X = Args.unaryX args } :> IOp2
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
            member this.ReplaceArgs args = { Log10.X = Args.unaryX args } :> IOp2
            member this.SubstSymSizes env = this :> IOp2
            member this.CanEvalAllSymSizes = true
            member this.Deriv dOp = Args.unary -dOp // TODO
            member this.Eval env = (Args.unaryX env.Args).Log10 ()
       
    let (|Log10|_|) (expr: Expr2) =
        match expr.Op with
        | :? Log10 as this -> Some this.X
        | _ -> None

[<AutoOpen>]
module BinaryOps =

    type Add = { X: Expr2; Y: Expr2 } with
        interface IOp2 with       
            member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.binary this.X this.Y
            member this.Deriv dOp = Args.binary dOp dOp
       
    let (|Add|_|) (expr: Expr2) =
        match expr.Op with
        | :? Add as this -> Some (this.X, this.Y)
        | _ -> None



 
