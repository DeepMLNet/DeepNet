namespace SymTensor.Ops

open DeepNet.Utils
open Tensor
open Tensor.Backend
open SymTensor


/// Scalar constant value
type Scalar = { Value: Const } with
    interface IOp with    
        member this.Check () = ()
        member this.TypeName = this.Value.TypeName
        member this.Shape = ShapeSpec.scalar
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = this.Value.AsTensor env.Dev       
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty

    
/// Value of the specified size
type SizeValue = { Value: SizeSpec } with
    interface IOp with    
        member this.Check () = ()
        member this.TypeName = TypeName.ofType<int64>
        member this.Shape = ShapeSpec.scalar
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = { Value = SymSizeEnv.subst env this.Value } :> _
        member this.CanEvalAllSymSizes = SizeSpec.canEval this.Value
        member this.Eval env = 
            SizeSpec.eval this.Value |> Tensor.scalar env.Dev :> ITensor       
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty


/// Identity matrix
type Identity = { Size: SizeSpec; Type: TypeName} with     
    interface IOp with    
        member this.Check () = ()
        member this.TypeName = this.Type
        member this.Shape = ShapeSpec.matrix this.Size this.Size
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> _
        member this.CanEvalAllSymSizes = SizeSpec.canEval this.Size
        member this.Eval env = 
            (Generic<IdentityTyped<_>, IIdentityTyped> [this.Type.Type]).Eval this env
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty

and internal IIdentityTyped =
    abstract Eval: this:Identity -> env:EvalEnv -> ITensor

and internal IdentityTyped<'T> () =     
    interface IIdentityTyped with
        member __.Eval this env =
            Tensor<'T>.identity env.Dev (SizeSpec.eval this.Size) :> ITensor       


/// Counting vector of given size
type Arange = { Size: SizeSpec; Type: TypeName} with
    interface IOp with    
        member this.Check () = ()
        member this.TypeName = this.Type
        member this.Shape = ShapeSpec.vector this.Size
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Size = SymSizeEnv.subst env this.Size} :> _
        member this.CanEvalAllSymSizes = SizeSpec.canEval this.Size
        member this.Eval env = 
            (Generic<ArangeTyped<_>, IArangeTyped> [this.Type.Type]).Eval this env                
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty

and internal IArangeTyped =
    abstract Eval: this:Arange -> env:EvalEnv -> ITensor

and internal ArangeTyped<'T> () =     
    interface IArangeTyped with
        member __.Eval this env =
            Tensor.counting env.Dev (SizeSpec.eval this.Size) :> ITensor       


/// Argument (placeholder for a variable).
type VarArg = { Var: Var } with
    interface IOp with       
        member this.Check () = ()
        member this.TypeName = this.Var.TypeName
        member this.Shape = this.Var.Shape
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = 
            {Var={this.Var with Shape=SymSizeEnv.substShape env this.Var.Shape}} :> _
        member this.CanEvalAllSymSizes = ShapeSpec.canEval this.Var.Shape
        member this.Eval env = 
            env.VarEnv |> VarEnv.get this.Var |> ITensor.transfer env.Dev       
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty

