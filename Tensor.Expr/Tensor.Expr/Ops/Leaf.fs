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
        member this.Shapes = Shape.scalar |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            this.Value |> Const.asITensor this.Dev |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs data 
        member this.Actions data =
            let valueTensor = this.Value |> Const.asITensor this.Dev 
            CompileTools.simpleAction (fun chVals argVals ->
                chVals.[Ch.Default].CopyFrom valueTensor)

    interface IOpFormat with
        member this.Text =
            sprintf "%A<@%A>" this.Value this.Dev

    
/// Value of the specified size
type SizeValue = { Value: Size; Dev: ITensorDevice } with
    interface IOp with    
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.Dev |> Ch.only
        member this.Shapes = Shape.scalar |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Value = Size.subst env this.Value} :> _
        member this.CanEvalAllSymSizes = Size.canEval this.Value
        member this.Eval env argVals = 
            Size.eval this.Value |> Tensor.scalar this.Dev :> ITensor |> Ch.only     

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs data 
        member this.Actions data =
            let valueTensor = Size.eval this.Value |> Tensor.scalar this.Dev
            CompileTools.simpleAction (fun chVals argVals ->
                chVals.[Ch.Default].CopyFrom valueTensor)

    interface IOpFormat with
        member this.Text =
            sprintf "Size<int64@%A>(%A)" this.Dev this.Value


/// Identity matrix
type Identity = { Size: Size; Type: TypeName; Dev: ITensorDevice } with     
    interface IOp with    
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.Type |> Ch.only
        member this.Devs = this.Dev |> Ch.only
        member this.Shapes = Shape.matrix this.Size this.Size |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Size = Size.subst env this.Size} :> _
        member this.CanEvalAllSymSizes = Size.canEval this.Size
        member this.Eval env argVals = 
            (Generic<IdentityTyped<_>, IIdentityTyped> [this.Type.Type]).Eval this env
            |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs data 
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                chVals.[Ch.Default].FillIdentity ())

    interface IOpFormat with
        member this.Text =
            sprintf "Id<%A@%A>(%A x %A)" this.Type this.Dev this.Size this.Size

and internal IIdentityTyped =
    abstract Eval: this:Identity -> env:EvalEnv -> ITensor

and internal IdentityTyped<'T> () =     
    interface IIdentityTyped with
        member __.Eval this env =
            Tensor<'T>.identity this.Dev (Size.eval this.Size) :> ITensor       


/// Counting vector of given size
type Counting = { Size: Size; Dev: ITensorDevice } with
    interface IOp with    
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.Dev |> Ch.only
        member this.Shapes = Shape.vector this.Size |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = {this with Size = Size.subst env this.Size} :> _
        member this.CanEvalAllSymSizes = Size.canEval this.Size
        member this.Eval env argVals = 
            Tensor.counting this.Dev (Size.eval this.Size) :> ITensor  
            |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs data 
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                let t = chVals.[Ch.Default] :?> Tensor<int64>
                t.FillIncrementing (0L, 1L))

    interface IOpFormat with
        member this.Text =
            sprintf "0 .. %A<@%A>" this.Size this.Dev


/// Argument (placeholder for a variable).
type VarArg = { Var: Var } with
    interface IOp with       
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.Devs = this.Var.Dev |> Ch.only
        member this.TypeNames = this.Var.TypeName |> Ch.only
        member this.Shapes = this.Var.Shape |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = 
            {Var={this.Var with Shape=Shape.subst env this.Var.Shape}} :> _
        member this.CanEvalAllSymSizes = Shape.canEval this.Var.Shape
        member this.Eval env argVals = 
            env.VarEnv.[this.Var.Name] |> Ch.only       

    interface IVarOp with
        member this.Var = this.Var

    interface ICompilableOp with
        member this.ChStubs data =
            Ch.only {
                Shape = Shape.eval this.Var.Shape
                TypeName = this.Var.TypeName
                Dev = this.Var.Dev
                OffsetStride = data.Env.VarOffsetStrides |> Map.tryFind this.Var.Name 
                Storage = StorageStub.VarStorage this.Var.Name
            }
        member this.Actions data =
            // Variable value is read directly from its storage, thus 
            // no actions need to be performed.
            []

    interface IOpFormat with
        member this.Text =
            sprintf "%A" this.Var


/// A reference to a data tensor.
type DataArg = { Data: OrdRef<ITensor> } with
    interface IOp with       
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.Devs = 
            this.Data.Value.Dev |> Ch.only
        member this.TypeNames = 
            TypeName.ofTypeInst this.Data.Value.DataType |> Ch.only
        member this.Shapes = 
            this.Data.Value.Shape |> List.map Size.fix |> Ch.only
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals =
            this.Data.Value |> Ch.only       

    interface ICompilableOp with
        member this.ChStubs data =
            Ch.only {
                Shape = this.Data.Value.Shape
                TypeName = TypeName.ofTypeInst this.Data.Value.DataType
                Dev = this.Data.Value.Dev
                OffsetStride = Some (this.Data.Value.Layout.Offset, this.Data.Value.Layout.Stride) 
                Storage = StorageStub.Fixed this.Data.Value.Storage
            }
        member this.Actions data =
            // Data value is read directly from its storage, thus 
            // no actions need to be performed.
            []

    interface IOpFormat with
        member this.Text = 
            sprintf "%A" this.Data


