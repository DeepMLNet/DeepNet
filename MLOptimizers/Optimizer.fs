namespace Optimizers

open Basics
open Tensor
open SymTensor


[<AutoOpen>]
module OptimizerTypes =

    /// Optimizer.
    type IOptimizer<'T, 'OptCfg, 'OptState> =

        /// Expression performing one optimization step.
        abstract member OptStepExpr: ExprT

        abstract member Use: (VarEnvT -> 'A) -> (VarEnvT -> 'OptCfg -> 'OptState -> 'A)

        /// Returns the configuration with the learning rate set to the specified value.
        abstract member CfgWithLearningRate: learningRate:float -> cfg:'OptCfg -> 'OptCfg

        /// Initial optimizer state.
        abstract member InitialState: cfg:'OptCfg -> parameterValues:ITensor -> 'OptState

        /// Loads the optimizer state from a HDF5 file with given prefix.
        abstract member LoadState: hdf:HDF5 -> prefix:string -> 'OptState

        /// Saves the optimizer state to a HDF5 file with given prefix.
        abstract member SaveState: hdf:HDF5 -> prefix:string -> state:'OptState -> unit


        

