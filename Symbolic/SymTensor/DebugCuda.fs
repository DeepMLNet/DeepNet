namespace SymTensor.Compiler.Cuda


module Debug =
    /// redirects all stream calls to the null stream
    let mutable DisableStreams = true

    /// disables all events
    let mutable DisableEvents = true

    /// synchronizes the CUDA context after each call to detect errors
    let mutable SyncAfterEachCudaCall = false

    /// outputs messages when a function / kernel is launched
    let mutable TraceCalls = false

    /// compiles kernels with debug information and no optimizations
    let mutable DebugCompile = false

    /// enable the use of fast math operations in kernels
    let mutable FastKernelMath = false

    /// tells nvrtc that all pointers are restricted
    let mutable RestrictKernels = true

    /// compiles kernels with source line-number information (for nVidia profiler)
    let mutable GenerateLineInfo = false

    /// keeps the compile temporary directory 
    let mutable KeepCompileDir = false

    /// disables the caching of CUDA kernels
    let mutable DisableKernelCache = false

    /// prints timing information during CUDA function compilation
    let mutable Timing = false

    /// prints CUDA memory usage during CUDA function compilation
    let mutable ResourceUsage = false

    /// prints each compile step during compilation
    let mutable TraceCompile = false

    /// prints ptxas verbose information during compilation
    let mutable PtxasInfo = false

    /// dumps kernel code before it is compiled
    let mutable DumpCode = false

    /// terminates the program when a non-finite tensor was found by the CheckFinite op
    let mutable TerminateWhenNonFinite = true

    /// disables ordering of element expressions work to increase coalesced memory access
    let mutable DisableElementsWorkOrdering = false
