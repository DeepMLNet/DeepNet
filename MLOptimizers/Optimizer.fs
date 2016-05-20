namespace Optimizers

open ArrayNDNS
open SymTensor


[<AutoOpen>]
module OptimizerTypes =

    /// Optimizer.
    type IOptimizer<'T, 'OptCfg, 'OptState> =

        /// Expression performing one optimization step.
        abstract member OptStepExpr: ExprT<'T>

        abstract member Use: (VarEnvT -> 'A) -> (VarEnvT -> 'OptCfg -> 'OptState -> 'A)

        /// Returns the configuration with the learning rate set to the specified value.
        abstract member CfgWithLearningRate: learningRate:float -> cfg:'OptCfg -> 'OptCfg

        /// Initial optimizer state.
        abstract member InitialState: cfg:'OptCfg -> parameterValues:IArrayNDT -> 'OptState

        /// Loads the optimizer state from disk.
        abstract member LoadState: path:string -> 'OptState

        /// Saves the optimizer state to disk.
        abstract member SaveState: path:string -> state:'OptState -> unit


        

