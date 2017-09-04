#r "../Tensor/bin/Debug/ManagedCuda.dll"
#r "../Tensor/bin/Debug/Tensor.dll"


open Tensor
open Tensor.Algorithms

let M = HostTensor.ofList2D [[0;   0; 0]
                             [-3; -1; 4]
                             [-2;  1; 5]]
let Mr = M |> Tensor.convert<Rat>

let E, Mi = RowEchelonForm.computeAugmented Mr (HostTensor.identity 3L)
printfn "E=\n%A\nMi=\n%A" E Mi
