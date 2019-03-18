open Tensor
open Tensor.Expr
open Tensor.Expr.ML.Opt
open Tensor.Expr.ML
open Tensor.Backend


let simplePerceptron () =
    let rng = System.Random 1
        
    // make training data
    let x = HostTensor.linspace -2.0f 2.0f 100L
    let y1 = 3.0f + 7.0f * x ** 2.0f
    let y2 = 1.0f + 2.0f * x ** 3.0f + 4.0f * x ** 4.0f
    let y = Tensor.concat 1 [y1.[*, NewAxis]; y2.[*, NewAxis]]
    printfn "x: %A" x
    printfn "y: %A" y
    let exps = HostTensor.arange 0.0f 1.0f 10.0f
    let f = x.[*, NewAxis] ** exps.[NewAxis, *]
    printfn "f: %A" f.Shape

    // context
    let ctx = Context.root HostTensor.Dev
        
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
    printfn "loss: %s\n" (loss.ToString())

    // parameter set
    let parSet = ParSet.fromExpr ContextPath.root loss
    let sizeEnv = Map [
        nFeatures, Size.fix f.Shape.[1]
        nOutputs, Size.fix y.Shape.[1]
    ]
    let parSetInst = ParSet.inst ContextPath.root sizeEnv parSet
    let loss = parSetInst.Use loss 
    printfn "with ParSet: %s\n" (loss.ToString())

    // use optimizer
    let optCfg = Adam.Cfg.standard
    let opt = Adam.make (optCfg, loss, parSetInst)
    let minStep = opt.Step
    let minLossStep = minStep |> EvalUpdateBundle.addExpr loss
    printfn "Minimiziation step: %A\n" minStep

    // evaluate using training data
    let varEnv = 
        VarEnv.empty
        |> VarEnv.add inputVar f
        |> VarEnv.add targetVar y
        |> parSetInst.Use
    let lossVal = loss |> Expr.eval varEnv
    printfn "loss value: %A" lossVal

    // perform training step
    printfn "training..."
    for i in 1..200 do
        let results = minLossStep |> EvalUpdateBundle.exec varEnv
        printf "step %d loss value: %f             \r" i (results.Get loss).Value
    printfn ""





[<EntryPoint>]
let main argv =
    simplePerceptron()
    0
