namespace Tensor.Expr.Compiler

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.Base



/// Data for compilation of an expression.
type CompileEnv = {
    //VarOffsetStrides: Map<VarName, int64 * int64 list> 
    /// Root expression channels that should have their values stored in
    /// tensors passed in during execution.
    ExternalTargets:    Set<BaseExprCh>
}



/// Data for execution of a compiled expression.
type ExecuteEnv = {
    /// Values for each variable.
    VarValues:          Map<VarName, ITensor>
    /// Tensors to store the resulting values of the root expressions.
    /// Only for results that were specified to use external values during compilation.
    ResultTargets:      Map<BaseExprCh, ITensor>
    /// Number of execution threads.
    /// If None, one thread is used per processor core.
    ThreadCount:        int option
    /// If true, results can use allocated or external storage, which might
    /// be shared with other data (variable values) or be overwritten when
    /// the workspace is executed again.
    /// If false, results are guaranteed to have an exclusive storage and
    /// may be overwritten by the caller.
    RawResults:         bool
}



/// Data passed to an action for execution.
type ActionData = {
    /// Returns the value of a tensor stub.
    StubValue:          TensorStub -> ITensor
    /// The execution environment.
    Env:                ExecuteEnv
}

/// Result of the execution of an action.
and ActionResult = {
    /// Values for channels that have dynamic tensor stubs.
    RuntimeChValues:    Map<Ch, ITensor>  
}

/// An action to execute for a compiled op.
type IAction =
    /// Execute actions and return resulting tensor for dynamic stubs.
    /// A tensor returned for a dynamic tensor stub must not use a storage
    /// with a static allocation.
    /// TODO: verify and enforce.
    abstract Execute: ActionData -> ActionResult
    /// Device that executes this action (primarily).
    /// TODO: Extend to multiple devices, when copying data between devices.
    abstract Dev: ITensorDevice


/// Base interface for device-specific data for an action.
type IActionDeviceData =
    interface end


/// Action that marks the end of execution.
/// This action is put into an ActionGroup that depends on all ActionGroup without dependants.
type FinishAction () =
    interface IAction with
        member __.Execute _ = failwith "Not to be executed."
        member __.Dev = HostTensor.Dev


/// Node containing an action for execution.
[<ReferenceEquality>]
type ActionNode = {
    /// Corresponding expression.
    Expr:               BaseExpr option
    /// ActionNodes this nodes depends on.
    DependsOn:          HashSet<ActionNode>
    /// ActionNodes that depend on this node.
    Dependants:         HashSet<ActionNode>
    /// Action to execute.
    Action:             IAction 
    /// Channel stubs.
    ChStubs:            Map<Ch, TensorStub>
    /// Device-specific action data.
    DevData:            IActionDeviceData option
} with
    /// Primary execution device.
    member this.Dev = this.Action.Dev


/// Results of tensor stub and action assignment.
type ExecutionRecipe = {
    /// All storage allocations.
    Allocs:             AllocStub list
    /// Stubs for all expression channels in the expression tree.
    ChStubs:            Map<BaseExprCh, TensorStub>
    /// All action nodes.
    ActionNodes:        ActionNode list
    /// Stubs for all expression channels of root expressions.
    ResultStubs:        Map<BaseExprCh, TensorStub>
}


/// Data for compilation of an op.
type CompileOpArgs = {
    /// Storage allocator.
    Alloc:              AllocReq -> AllocStub
    /// Compile environment.
    Env:                CompileEnv
    /// Expression containing the op to compile.
    Expr:               BaseExpr
    /// Tensor stubs for the arguments.
    ArgStubs:           Map<Arg, TensorStub> 
    /// Arguments with data that may be overwritten.
    OverwritableArgs:   Set<Arg>
    /// Wishes for the tensor stubs of the channels.
    ChStubWishes:       Map<Ch, TensorStub>
    /// Pre-assigned channel stubs.
    ChStubs:            Map<Ch, TensorStub>
    /// Wishes for the tensor stubs of the arguments.
    ArgStubWishes:      Map<Arg, TensorStub>
}

/// Result of the compilation of an op.
type CompileOpResult = {
    /// Tensor stubs for each channel.
    ChStubs:            Map<Ch, TensorStub>
    /// Action to perform for this op during execution.
    Actions:            IAction 
}

/// Interface for ops that can be compiled.
type ICompilableOp =
    /// Should compute the channel stubs given the argument stubs.
    abstract Compile: CompileOpArgs -> CompileOpResult



type NonCompilableOpAction (expr: BaseExpr, argStubs: Map<Arg, TensorStub>) =
    interface IAction with
        member this.Execute data =
            failwith "TODO"
        member this.Dev =
            failwith "TODO"



/// Arguments for IStubWishingOp.WishStubs.
type WishStubsArgs = {
    /// Storage allocator.
    Alloc:              AllocReq -> AllocStub
    /// Compile environment.
    Env:                CompileEnv    
    /// Expression containing the op.
    Expr:               BaseExpr
    /// Wishes for the tensor stubs of the channels.
    ChStubWishes:       Map<Ch, TensorStub>
}

/// Results from IStubWishingOp.WishStubs.
type WishStubsResult = {
    /// Accepeted channel stub wishes.
    ChStubs:            Map<Ch, TensorStub>
    /// Wishes for tensor stubs of arguments.
    ArgStubWishes:      Map<Arg, TensorStub>
}

type IStubWishingOp =
    /// Should compute argument stub wishes given channel stub wishes.
    abstract WishStubs: WishStubsArgs -> WishStubsResult