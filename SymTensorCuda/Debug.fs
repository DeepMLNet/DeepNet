namespace SymTensor.Compiler.Cuda


module Debug =
    /// redirects all stream calls to the null stream
    [<Literal>]
    let DisableStreams = false

    /// outputs messages when a function / kernel is launched
    let mutable TraceCalls = false

    /// compiles kernels with debug information and no optimizations
    [<Literal>]
    let DebugCompile = false

    /// prints timing information during CUDA function compilation
    let mutable Timing = false

    /// prints CUDA memory usage during CUDA function compilation
    let mutable MemUsage = false

    /// prints each compile step during compilation
    let mutable TraceCompile = false

    /// dumps kernel code before it is compiled
    let mutable DumpCode = true