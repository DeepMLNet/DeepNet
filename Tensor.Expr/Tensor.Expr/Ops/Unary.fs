namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor.Expr
open Tensor
open Tensor.Backend


/// Unary plus.
type UnaryPlus = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).UnaryPlus () |> Ch.only     

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillUnaryPlus (ArgValue.unaryX argVals))


/// Negation.
type Negate = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).UnaryMinus () |> Ch.only      

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillUnaryMinus (ArgValue.unaryX argVals))


/// Absolute value.
type Abs = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Abs () |> Ch.only   
        
    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAbs (ArgValue.unaryX argVals))

    
/// Sign.
type SignT = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only 
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sgn () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSgn (ArgValue.unaryX argVals))


/// Logarithm to base exp.
type Log = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Log () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillLog (ArgValue.unaryX argVals))


/// Logarithm to base 10.
type Log10 = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Log10 () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillLog10 (ArgValue.unaryX argVals))


/// Exponential function.
type Exp = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Exp () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillExp (ArgValue.unaryX argVals))


/// Sine.
type Sin = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sin () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSin (ArgValue.unaryX argVals))


/// Cosine.
type Cos = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Cos () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillCos (ArgValue.unaryX argVals))


/// Tangent.
type Tan = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Tan () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillTan (ArgValue.unaryX argVals))


/// Inverse sine.
type Asin = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Asin () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAsin (ArgValue.unaryX argVals))


/// Inverse cosine.
type Acos = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Acos () |> Ch.only     
        
    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAcos (ArgValue.unaryX argVals))


/// Inverse tangent.
type Atan = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Atan () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAtan (ArgValue.unaryX argVals))


/// Hyperbolic sine.
type Sinh = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sinh () |> Ch.only      

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSinh (ArgValue.unaryX argVals))


/// Hyperbolic cosine.
type Cosh = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Cosh () |> Ch.only  
        
    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillCosh (ArgValue.unaryX argVals))


/// Hyperbolic tangent.
type Tanh = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Tanh () |> Ch.only   
        
    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillTanh (ArgValue.unaryX argVals))
        

/// Square root.
type Sqrt = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sqrt () |> Ch.only    
        
    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSqrt (ArgValue.unaryX argVals))


/// Round towards positive infinity.
type Ceiling = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Ceiling () |> Ch.only
        
    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillCeiling (ArgValue.unaryX argVals))


/// Round towards negative infinity.
type Floor = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Floor () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillFloor (ArgValue.unaryX argVals))


/// Round towards nearest integer.
type Round = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Round () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillRound (ArgValue.unaryX argVals))


/// Round towards zeros.
type Truncate = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Truncate () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillTruncate (ArgValue.unaryX argVals))


/// (Batched) matrix inverse.
type Invert = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = 
            if this.X.NDims < 2 then
                failwithf "Need at least a matrix to invert but got shape %A" this.X.Shape
            if not (Size.equalIgnoringBc this.X.Shape.[this.X.NDims-2] this.X.Shape.[this.X.NDims-1]) then
                failwithf "Cannot invert non-square matrix %A along last two axes." this.X.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Invert () |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            // TODO: So far temporary storage is allocated at runtime.
            //       This should be changed to also perform preallocations.
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillInvert (ArgValue.unaryX argVals))
                

/// Logical not.
type Not = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = Check.bool [this.X]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ~~~~(ArgValue.unaryX argVals :?> Tensor<bool>) :> ITensor |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals :?> Tensor<bool>).FillNegate (ArgValue.unaryX argVals :?> Tensor<bool>))


/// Reshape
type Reshape = { X: BaseExprCh; Shape: Shape } with
    interface IOp with      
        member this.Check () = 
            if not (Size.equalIgnoringBc (Shape.nElem this.X.Shape) (Shape.nElem this.Shape)) then
                failwithf "Cannot change number of elements while reshaping from %A to %A." 
                            this.X.Shape this.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = 
            { this with Shape = Shape.subst env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            Shape.canEval this.Shape
        member this.Eval env argVals =
            (ArgValue.unaryX argVals) |> ITensor.reshape (Shape.eval this.Shape) |> Ch.only

    interface ITensorStubWishPropagatingOp with
        member this.PropagateWishes chWishes =
            chWishes |> CompileTools.propUnaryWish (fun wish ->
                wish |> TensorStub.tryReshape (Shape.eval this.Shape))

    interface ICompilableOp with
        member this.ChStubs data =
            // We return a view of X, when layout is in row-major order.
            // Otherwise, a copy is required.
            failwith "TODO"
            CompileTools.chStubs (data, tryInplace=true)
        member this.Actions data =
            failwith "TODO"

    interface IOpFormat with
        member this.Text =
            sprintf "Reshape%A" this.Shape


/// Broadcast.
type DoBroadcast = { X: BaseExprCh; Shape: Shape } with
    interface IOp with      
        member this.Check () = 
            if Shape.nDim this.X.Shape <> Shape.nDim this.Shape then
                failwithf "Tensor of shape %A does not have same number of dimesions as broadcast shape %A."
                            this.X.Shape this.Shape
            for dim in 0 .. (Shape.nDim this.Shape) - 1 do
                match this.X.Shape.[dim], this.Shape.[dim] with
                | Size.Broadcast, _ -> ()
                | ssa, ssb when not (Size.equalIgnoringBc ssa ssb) -> 
                    failwithf "Cannot broadcast from %A to %A because non-broadcast dimensions must not change." 
                                this.X.Shape this.Shape
                | _ -> ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = 
            { this with Shape = Shape.subst env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            Shape.canEval this.Shape
        member this.Eval env argVals = (ArgValue.unaryX argVals) |> ITensor.broadcastTo (Shape.eval this.Shape) |> Ch.only

    interface IOpFormat with
        member this.Text =
            sprintf "DoBroadcast%A" this.Shape


/// Permute the axes.
type PermuteAxes = {X: BaseExprCh; Permutation: int list} with
    interface IOp with      
        member this.Check () = 
            if Shape.nDim this.X.Shape <> List.length this.Permutation then
                failwithf "Permutation %A must have same rank as shape %A." this.Permutation this.X.Shape
            if not (Permutation.is this.Permutation) then
                failwithf "%A is not a valid permutation of an %d-dimensional tensor." 
                            this.Permutation (Shape.nDim this.X.Shape)
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.permuteAxes this.Permutation |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals) |> ITensor.permuteAxes this.Permutation |> Ch.only

    interface ITensorStubWishPropagatingOp with
        member this.PropagateWishes chWishes =
            chWishes |> CompileTools.propUnaryWish (fun wish ->
                wish |> TensorStub.tryPermuteAxes (Permutation.invert this.Permutation))

    interface IOpFormat with
        member this.Text =
            sprintf "PermuteAxes%A" this.Permutation


/// Read a slice from a tensor.
type Subtensor = {X: BaseExprCh; Range: SimpleRanges} with
    interface IOp with      
        member this.Check () = 
            Check.range this.Range this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = 
            (this.Range, this.X.Shape)
            ||> List.map2 (fun sr shp ->
                match sr with
                | SimpleRange.SymStartSymEnd (s, fo)    -> (fo |? (shp - Size.one)) + 1L - s
                | SimpleRange.DynStartSymSize (_, size) -> size)            
            |> Ch.only
        member this.Args = 
            let xArgs = Args.unary this.X 
            let dynArgs = 
                SimpleRangesArgs.toArgs this.Range
                |> Map.map (fun _ v -> v :?> BaseExprCh)
            Map.join xArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesArgs.replaceFromArgs dynArgs               
            {this with X=Args.unaryX args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SimpleRanges.subst env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRanges.canEvalSymbols this.Range
        member this.Eval env argVals = 
            // TODO: dynamic range is always copied to host
            let dynVals = 
                argVals 
                |> Map.filter (fun arg _ -> 
                    match arg with
                    | Arg.N _ -> true
                    | _ -> false)
                |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> Size.fix)
            let range = 
                this.Range 
                |> SimpleRangesArgs.resolveDynElems dynVals 
                |> SimpleRanges.eval
            (ArgValue.unaryX argVals).[range] |> Ch.only

    interface IOpFormat with
        member this.Text =
            sprintf "Subtensor%A" this.Range


/// Reverses the tensor in the specified dimension.
type ReverseAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals) |> ITensor.reverseAxis this.Axis |> Ch.only

    interface ITensorStubWishPropagatingOp with
        member this.PropagateWishes chWishes =
            chWishes |> CompileTools.propUnaryWish (fun wish ->
                wish |> TensorStub.tryReverseAxis this.Axis)

    interface IOpFormat with
        member this.Text =
            sprintf "ReverseAxis<%d>" this.Axis


/// Extract the diagonal(s) along the given axes.
type Diag = {X: BaseExprCh; Axis1: int; Axis2: int} with
    interface IOp with      
        member this.Check () = 
            Check.axis this.Axis1 this.X
            Check.axis this.Axis2 this.X 
            if not (this.Axis1 < this.Axis2) then 
                failwith "First axis for extracting diagonal must come before second axis."
            if not (Size.equalIgnoringBc this.X.Shape.[this.Axis1] this.X.Shape.[this.Axis2]) then
                failwithf "Cannot extract diagonal along axes %d and %d from non-square tensor with shape %A" 
                            this.Axis1 this.Axis2 this.X.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis2 |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).DiagAxis this.Axis1 this.Axis2 |> Ch.only


/// Build a matrix with the specified diagonal.
type DiagMat = {X: BaseExprCh; Axis1: int; Axis2: int} with
    interface IOp with      
        member this.Check () = 
            Check.axis this.Axis1 this.X
            if not (0 <= this.Axis2 && this.Axis2 <= this.X.NDims) then
                failwithf "Cannot build diagonal matrix over non-existant axis %d of tensor with shape %A." 
                            this.Axis2 this.X.Shape
            if not (this.Axis1 < this.Axis2) then 
                failwith "First axis for building diagonal matrix must come before second axis."
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> List.insert this.Axis2 this.X.Shape.[this.Axis1] |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).DiagMatAxis this.Axis1 this.Axis2 |> Ch.only


/// Sum over specified axis.
type SumAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).SumAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSumAxis this.Axis (ArgValue.unaryX argVals))


/// Product over specified axis.
type ProductAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ProductAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillProductAxis this.Axis (ArgValue.unaryX argVals))


/// Maximum over specified axis.
type MaxAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).MaxAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillMaxAxis this.Axis (ArgValue.unaryX argVals))


/// Minimum over specified axis.
type MinAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).MinAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillMinAxis this.Axis (ArgValue.unaryX argVals))


/// Maximum over specified axis.
type ArgMaxAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ArgMaxAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillArgMaxAxis this.Axis (ArgValue.unaryX argVals))


/// Minimum over specified axis.
type ArgMinAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ArgMinAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                (ChValue.onlyX chVals).FillArgMinAxis this.Axis (ArgValue.unaryX argVals))


/// Select elements according to the specified index tensors
type Gather = {X: BaseExprCh; Indices: BaseExprCh option list} with
    interface IOp with      
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
                                (this.Indices |> List.map (Option.map (fun e -> e.Shape)))
                | None when dim >= Shape.nDim trgtShape ->
                    failwithf "Gather index dimensions beyond the number of target dimensions \
                                must not be None."
                | _ -> ()
        member this.Channels = Ch.onlyOne   
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = (this.Indices |> List.pick id).Shape |> Ch.only
        member this.Args = 
            let idxArgs = Args.naryOpt this.Indices
            let xArgs = Args.unary this.X
            Map.join idxArgs xArgs
        member this.ReplaceArgs args =                
            {this with 
                X = Args.unaryX args
                Indices = Args.naryOptXs this.Indices.Length args
            } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
            (ArgValue.unaryX argVals).Gather vIndices |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
                (ChValue.onlyX chVals).FillGather vIndices (ArgValue.unaryX argVals))


/// Disperses elements according to the specified index tensors.
type Scatter = {X: BaseExprCh; Indices: BaseExprCh option list; Shape: Shape} with
    interface IOp with      
        member this.Check () = 
            for dim, idx in List.indexed this.Indices do
                match idx with
                | Some idx when idx.DataType <> typeof<int64> ->
                    failwithf "All index tensors for scatter must be of type int64, but got type %A." idx.DataType
                | Some idx when idx.Shape <> this.X.Shape ->
                    failwithf "All scatter indices must have shape of source %A, but got %A." 
                                this.X.Shape (this.Indices |> List.map (Option.map (fun e -> e.Shape)))
                | None when dim >= this.X.NDims ->
                    failwithf "Scatter index dimensions beyond the number of source dimensions \
                                must not be None."
                | _ -> ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = 
            let idxArgs = Args.naryOpt this.Indices            
            let xArgs = Args.unary this.X
            Map.join idxArgs xArgs
        member this.ReplaceArgs args =                
            {this with 
                X = Args.unaryX args
                Indices = Args.naryOptXs this.Indices.Length args
            } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
            (ArgValue.unaryX argVals).Scatter vIndices (Shape.eval this.Shape) |> Ch.only

    interface ICompilableOp with
        member this.ChStubs data =
            CompileTools.chStubs (data)
        member this.Actions data =
            CompileTools.simpleAction (fun chVals argVals ->
                let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
                (ChValue.onlyX chVals).FillScatter vIndices (ArgValue.unaryX argVals))


/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeZeroDeriv = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ArgValue.unaryX argVals |> Ch.only
    

/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeDeriv = {Deriv: BaseExprCh; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = 
            Check.sameType [this.Deriv; this.X]
            if this.Deriv.NDims <> 2 then
                failwithf "Jacobian shape %A must be two-dimensional." this.Deriv.Shape
            if this.Deriv.Shape.[1] <> this.X.NElems then
                failwithf "Jacobian shape %A must have %A elements in second dimension." 
                    this.Deriv.Shape this.X.NElems
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args =                 
            Map.join (Args.unary this.X) (Map [Arg.Custom "Deriv", this.Deriv])                
        member this.ReplaceArgs args = 
            {this with 
                Deriv = args.[Arg.Custom "Deriv"] 
                X = Args.unaryX args
            } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ArgValue.unaryX argVals |> Ch.only
    

/// Annotation (no influence on value).
type Annotated = {Label: System.IComparable; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ArgValue.unaryX argVals |> Ch.only                 

    
/// Prints the value together with the given label.
type Print = {Label: string; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            printfn "%s=\n%A\n" this.Label v
            v |> Ch.only                          
    

/// Dumps the result into the given dataset in the active HDF5 dump file.
type Dump = {Dataset: string; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            Dump.dumpValue this.Dataset v
            v |> Ch.only                            


/// A non-finite value was encountered.
exception NonFiniteValueException of msg:string with 
    /// <summary>Detailed error message.</summary>    
    override __.Message = __.msg


/// If the value contains NaNs or infinities, outputs their location and 
/// stops the computation.
type CheckFinite = {Label: string; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            if not (v.AllFinite ()) then
                let msg = 
                    sprintf "Non-finite value encountered in %s with value:\n%A" this.Label v
                raise (NonFiniteValueException msg)
            v |> Ch.only                            


/// Converts the data to the specified type.
type Convert = {ToType: TypeName; X: BaseExprCh} with
    interface IOp with
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.ToType |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            v |> ITensor.convertToType this.ToType.Type |> Ch.only     


/// Transfers the data to the specified device.
type Transfer = {ToDev: ITensorDevice; X: BaseExprCh} with
    interface IOp with
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.ToDev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            v |> ITensor.transfer this.ToDev |> Ch.only     


/// Returns the specified channel of a multi-channnel as its only channel.
type Channel = {X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X=Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            ArgValue.unaryX argVals |> Ch.only



