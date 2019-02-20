namespace Tensor.Expr


module Debug = 

    /// If true, then information about function instantiations for a particular set
    /// of variable sizes and locations is printed.
    let mutable PrintInstantiations = false

    /// If true, expressions are not optimized during function creation.
    let mutable DisableOptimizer = false

    /// If true, individual operations are not combined into an element expression
    /// during optimization.
    let mutable DisableCombineIntoElementsOptimization = false

    /// If ture, optimizer statistics are printed.
    let mutable PrintOptimizerStatistics = false

    /// if true, prints compilation step messages
    let mutable TraceCompile = false

    /// if true, prints compilation times
    let mutable Timing = false

    /// if false, Expr.checkFinite is doing nothing
    let mutable EnableCheckFinite = true

    /// if true, Deriv.ofVar fails if specified variable was not present in derived expression
    let mutable FailIfVarNotInDerivative = false

    /// if true, a graph of the expression tree is shown during compilation
    let mutable VisualizeUExpr = false

    /// if true, ExecItems are included in the visualization of the expression tree
    let mutable VisualizeExecItems = false

    /// terminates the program after an expression is compiled
    let mutable TerminateAfterCompilation = false
