namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr


/// Scalar constant value
type Scalar = { Value: Const; Dev: ITensorDevice } with
    interface IOp with    
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.Value.TypeName |> Ch.only
        member this.Devs = this.Dev |> Ch.only
        member this.Shapes = ShapeSpec.scalar |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            this.Value.AsTensor this.Dev |> Ch.only

    interface IOpFormat with
        member this.Text =
            sprintf "%A<@%A>" this.Value this.Dev

    
/// Value of the specified size
type SizeValue = { Value: SizeSpec; Dev: ITensorDevice } with
    interface IOp with    
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.Dev |> Ch.only
        member this.Shapes = ShapeSpec.scalar |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Value = SymSizeEnv.subst env this.Value} :> _
        member this.CanEvalAllSymSizes = SizeSpec.canEval this.Value
        member this.Eval env argVals = 
            SizeSpec.eval this.Value |> Tensor.scalar this.Dev :> ITensor |> Ch.only     

    interface IOpFormat with
        member this.Text =
            sprintf "%A<@%A>" this.Value this.Dev


/// Identity matrix
type Identity = { Size: SizeSpec; Type: TypeName; Dev: ITensorDevice } with     
    interface IOp with    
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.Type |> Ch.only
        member this.Devs = this.Dev |> Ch.only
        member this.Shapes = ShapeSpec.matrix this.Size this.Size |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> _
        member this.CanEvalAllSymSizes = SizeSpec.canEval this.Size
        member this.Eval env argVals = 
            (Generic<IdentityTyped<_>, IIdentityTyped> [this.Type.Type]).Eval this env
            |> Ch.only

    interface IOpFormat with
        member this.Text =
            sprintf "Id<%A@%A>(%A x %A)" this.Type this.Dev this.Size this.Size

and internal IIdentityTyped =
    abstract Eval: this:Identity -> env:EvalEnv -> ITensor

and internal IdentityTyped<'T> () =     
    interface IIdentityTyped with
        member __.Eval this env =
            Tensor<'T>.identity this.Dev (SizeSpec.eval this.Size) :> ITensor       


/// Counting vector of given size
type Arange = { Size: SizeSpec; Type: TypeName; Dev: ITensorDevice } with
    interface IOp with    
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.Type |> Ch.only
        member this.Devs = this.Dev |> Ch.only
        member this.Shapes = ShapeSpec.vector this.Size |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> _
        member this.CanEvalAllSymSizes = SizeSpec.canEval this.Size
        member this.Eval env argVals = 
            (Generic<ArangeTyped<_>, IArangeTyped> [this.Type.Type]).Eval this env  
            |> Ch.only

    interface IOpFormat with
        member this.Text =
            sprintf "0 .. %A<%A@%A>" this.Size this.Type this.Dev

and internal IArangeTyped =
    abstract Eval: this:Arange -> env:EvalEnv -> ITensor

and internal ArangeTyped<'T> () =     
    interface IArangeTyped with
        member __.Eval this env =
            Tensor.counting this.Dev (SizeSpec.eval this.Size) :> ITensor       


/// Argument (placeholder for a variable).
type VarArg = { Var: BaseVar } with
    interface IOp with       
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.Devs = this.Var.Dev |> Ch.only
        member this.TypeNames = this.Var.TypeName |> Ch.only
        member this.Shapes = this.Var.Shape |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = 
            {Var={this.Var with Shape=SymSizeEnv.substShape env this.Var.Shape}} :> _
        member this.CanEvalAllSymSizes = ShapeSpec.canEval this.Var.Shape
        member this.Eval env argVals = 
            env.VarEnv.[this.Var.Name] |> Ch.only       

    interface IVarContainingOp with
        member this.Vars =
            Set [this.Var]

    interface IOpFormat with
        member this.Text =
            sprintf "%A" this.Var

