open Tensor
open Tensor.Expr
open Tensor.Expr.ML.Opt
open Tensor.Expr.ML
open Tensor.Backend

open MBrace.FsPickler
open MBrace.FsPickler.Json

open System.IO


let createModel dev =
    let rng = System.Random 1
        
    // make training data
    let x = HostTensor.linspace -2.0f 2.0f 100L
    let y1 = 3.0f + 7.0f * x ** 2.0f
    let y2 = 1.0f + 2.0f * x ** 3.0f + 4.0f * x ** 4.0f
    let y = Tensor.concat 1 [y1.[*, NewAxis]; y2.[*, NewAxis]]
    let exps = HostTensor.arange 0.0f 1.0f 10.0f
    let f = x.[*, NewAxis] ** exps.[NewAxis, *]

    // context
    let ctx = Context.root dev
        
    // symbolic sizes
    let nSamples = SizeSym "nSamples"
    let nFeatures = SizeSym "nFeatures"
    let nOutputs = SizeSym "nOutputs"

    // model
    let inputVar = Var<float32> (ctx / "input", [Size.sym nSamples; Size.sym nFeatures])
    let targetVar = Var<float32> (ctx / "target", [Size.sym nSamples; Size.sym nOutputs])
    let input = Expr inputVar
    let target = Expr targetVar
    let hyperPars = NeuralLayer.HyperPars.standard (Size.sym nFeatures) (Size.sym nOutputs)
    let pars = NeuralLayer.pars ctx rng hyperPars
    let pred = NeuralLayer.pred pars input
    let loss = LossLayer.loss LossLayer.MSE pred target
    //printfn "loss: %s\n" (loss.ToString())

    // parameter set
    let parSet = ParSet.fromExpr ContextPath.root loss
    let sizeEnv = Map [
        nFeatures, Size.fix f.Shape.[1]
        nOutputs, Size.fix y.Shape.[1]
    ]
    let parSetInst = ParSet.inst ContextPath.root sizeEnv parSet
    let loss = parSetInst.Use loss 

    // use optimizer
    let optCfg = Adam.Cfg.standard
    let opt = Adam.make (optCfg, loss, parSetInst)
    let minStep = opt.Step
    let minLossStep = minStep |> EvalUpdateBundle.addExpr loss

    loss, minLossStep



let testSerTrip (name: string) (data: 'T) =
    printfn "\nbefore:\n%s\n" (data.ToString())

    let json = Json.serialize data
    File.WriteAllText (sprintf "%s.json" name, json)

    let deser: 'T = Json.deserialize json
    printfn "\nafter:\n%s\n" (deser.ToString())

    let eq = data = deser
    printfn "%s equal after serialization round trip: %A" name eq
    eq


[<EntryPoint>]
let main argv =
    printfn "On host:"
    let loss, minLossStep = createModel HostTensor.Dev
    testSerTrip "host_loss" loss |> ignore
    testSerTrip "host_minLossStep" minLossStep |> ignore


    printfn "On CUDA:"
    let loss, minLossStep = createModel CudaTensor.Dev
    testSerTrip "cuda_loss" loss |> ignore
    testSerTrip "cuda_minLossStep" minLossStep |> ignore

    0
