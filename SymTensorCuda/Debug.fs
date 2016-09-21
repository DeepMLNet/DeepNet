namespace SymTensor.Compiler.Cuda


module Debug =
    /// redirects all stream calls to the null stream
    let mutable DisableStreams = false

    /// outputs messages when a function / kernel is launched
    let mutable TraceCalls = false

    /// compiles kernels with debug information and no optimizations
    let mutable DebugCompile = false

    /// prints timing information during CUDA function compilation
    let mutable Timing = false

    /// prints CUDA memory usage during CUDA function compilation
    let mutable MemUsage = false

    /// prints each compile step during compilation
    let mutable TraceCompile = false

    /// dumps kernel code before it is compiled
    let mutable DumpCode = false

    /// terminates the program when a non-finite tensor was found by the CheckFinite op
    let mutable TerminateWhenNonFinite = true
