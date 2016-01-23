open Xunit

open Shape
open Op
open OpGrad

[<Fact>]
let ``Gradient of linear regression`` () =
    let A = var "A" [symbol "M"; symbol "N"]
    let b = var "b" [symbol "M"]
    let x = var "x" [symbol "N"]
    let t = var "t" [symbol "M"]

    let pred = A.*x + b
    let smplLoss = (pred - t)**2.0
    let loss = sum smplLoss

    printfn "loss = %A" loss
    printfn "shape of loss: %A" (shapeOf loss)

    let avar = 
        match A with
        | Var avar -> avar
        | _ -> failwith "not happen"

    ()
    let lossWrtA = grad loss avar
    printfn "dloss / dA = %A" lossWrtA

[<EntryPoint>]
let main argv = 
    ``Gradient of linear regression`` ()
    0
