module CudaBasics

open ManagedCuda

/// CUDA context
let cudaCntxt = 
    try
        new CudaContext(createNew=false)
    with
    e ->
        printfn "Cannot create CUDA context: %s" e.Message
        exit 10








