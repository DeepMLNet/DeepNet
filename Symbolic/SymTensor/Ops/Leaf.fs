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
    let binary x y : ArgsMap = Map ["X", x; "Y", y]


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
            member this.Eval dev args = this.Value.AsTensor dev
       
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
            member this.Eval dev args = SizeSpec.eval this.Value |> Tensor.scalar dev :> ITensor
       
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
            member this.Eval dev args = 
                Generic.callGeneric<IdentityTyped<_>, ITensor> "Eval" [this.Type.Type] (dev, this)

    type internal IdentityTyped<'T>() =


        static member Eval (op: Identity) (dev: ITensorDevice) =
            let typeLambda<'R> (a: string) =
                a

            Tensor<'T>.identity dev (SizeSpec.eval op.Size) :> ITensor

    // will work but introduces reflection call in Eval.
    // can this be avoided using an interface?
    // how often will this occur?
    // could it possibly be handled using a lambda?
    // would perhaps be a clean idea but has to be investigated...
    // creating a lambda to call a generic funciton
    // how could this look?
        

    ///// Identity matrix
    //type Identity<'T> = { Size: SizeSpec } with
    //    interface IOp2 with    
    //        member this.Check () = ()
    //        member this.TypeName = TypeName.ofType<'T>
    //        member this.Shape = ShapeSpec.matrix this.Size this.Size
    //        member this.Args = Map.empty
    //        member this.ReplaceArgs args = this :> IOp2
    //        member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> IOp2
    //        member this.CanEvalAllSymSizes = SizeSpec.canEval this.Size
    //        member this.Deriv dOp = Map.empty
    //        member this.Eval dev args = Tensor<'T>.identity dev (SizeSpec.eval this.Size) :> ITensor
    // contra: 
    // 1. unclear how pattern match should work
    // 2. how to handle n-ary expressions with different input types, e.g. a loop?

       
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
            member this.Args = Args.leaf
            member this.Deriv dOp = Map.empty
       
    let (|Arange|_|) (expr: Expr2) =
        match expr.Op with
        | :? Arange as this -> Some this
        | _ -> None


    /// Argument (placeholder for a variable).
    type Arg = { Var: Var } with
        interface IOp2 with       
            member this.Check () = ()
            member this.TypeName = this.Var.TypeName
            member this.Shape = this.Var.Shape
            member this.Args = Args.leaf
            member this.Deriv dOp = Map.empty
       
    let (|Arg|_|) (expr: Expr2) =
        match expr.Op with
        | :? Arg as this -> Some this.Var
        | _ -> None


[<AutoOpen>]
module UnaryOps =

    type Negate = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.Deriv dOp = Args.unary -dOp
       
    let (|Negate|_|) (expr: Expr2) =
        match expr.Op with
        | :? Negate as this -> Some this.X
        | _ -> None


    type Abs = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.Deriv dOp = failwith "TODO"
       
    let (|Abs|_|) (expr: Expr2) =
        match expr.Op with
        | :? Abs as this -> Some this.X
        | _ -> None


    type SignT = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.Deriv dOp = failwith "TODO"
       
    let (|SignT|_|) (expr: Expr2) =
        match expr.Op with
        | :? SignT as this -> Some this.X
        | _ -> None


    type Log = { X: Expr2 } with
        interface IOp2 with      
            member this.Check () = ()
            member this.TypeName = this.X.TypeName
            member this.Shape = this.X.Shape
            member this.Args = Args.unary this.X
            member this.Deriv dOp = failwith "TODO"
       
    let (|Log|_|) (expr: Expr2) =
        match expr.Op with
        | :? Log as this -> Some this.X
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



 
