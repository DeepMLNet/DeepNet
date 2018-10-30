namespace SymTensor

open SymTensor.Compiler

module DiagnosticsVisualizer =

    // Visualization function
    let mutable visualize: (CompileDiagnosticsT -> unit) = fun _ -> ()

