namespace Tensor.Expr.Compiler

open DeepNet.Utils
open Tensor.Expr



/// Performs a unique allocation for each allocation stub without reusing memory.
module IndividualAllocationRealizer =
    let perform (_env: CompileEnv) (actNodes: ActionNode list) : AllocPlan =
        // Extract AllocStubs from action nodes.
        let stubs = actNodes |> List.collect (fun actNode -> actNode.Allocs)        
        // Create one block per allocation.
        let blocks =
            stubs
            |> List.map (fun stub -> {
                TypeName = stub.TypeName
                Dev = stub.Dev
                Size = stub.Size
            })
        // Create realizations referencing blocks.
        let realizations =
            stubs
            |> List.mapi (fun idx stub ->
                stub, {
                    BlockIndex = idx
                    Offset = 0L
                })
            |> Map.ofList
        {
            Blocks = blocks
            Allocs = realizations
        }


