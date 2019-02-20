namespace Tensor.Expr

open Tensor.Expr.Compiler

module DiagnosticsVisualizer =

    // Visualization function
    let mutable visualize: (CompileDiagnosticsT -> unit) = fun _ -> ()

