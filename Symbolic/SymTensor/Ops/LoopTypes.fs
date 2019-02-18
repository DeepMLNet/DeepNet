namespace SymTensor.Loop

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


/// a slice of an argument to the loop
type SequenceArgSlice = {
    /// the index of the argument
    ArgIdx:     int
    /// the dimension the loop is performed over
    SliceDim:   int
}


/// references a loop channel of a previous iteration
type PreviousChannel = {
    /// the channel to use
    Channel:       string
    /// the delay, must be at least one
    Delay:         SizeSpec
    /// the index of the argument specifying the initial values
    InitialArg:    int
}


/// a loop variable value specification
type Input = 
    /// provides the loop argument to all loop iterations
    | ConstArg of argIdx:int
    /// provides a slice of the loop argument to each loop iteration
    | SequenceArgSlice of SequenceArgSlice
    /// provides the value of a loop channel from a previous loop iteration
    | PreviousChannel of PreviousChannel
    /// provides the index of the current loop iteration (zero-based)
    | IterationIndex
    /// provides the number of remaining loop iterations after this iteration
    | IterationsRemaining


/// the value of a loop channel
type Value = {
    /// the expression to compute the loop channel;
    /// it may only use variables defined in LoopSpecT.Vars
    Expr:       BaseExprCh
    /// the dimension to concatenate the results along to produce the loop output
    SliceDim:   int
}


