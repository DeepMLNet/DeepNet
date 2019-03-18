open Tensor
open Tensor.Expr
open Tensor.Expr.ML
open Tensor.Expr.ML.Opt
open Tensor.Expr.ML.Train
open Tensor.Backend



type Data = {
    // [smpl, feature]
    Inp: Tensor<float32>
    // [smpl, target]
    Trgt: Tensor<float32>
}


[<EntryPoint>]
let main argv =
    let rng = System.Random 1
    
    // make training data
    let x = HostTensor.linspace -2.0f 2.0f 100L
    let exps = HostTensor.arange 0.0f 1.0f 10.0f
    let inp = x.[*, NewAxis] ** exps.[NewAxis, *]
    let trgt1 = 3.0f + 7.0f * x ** 2.0f
    let trgt2 = 1.0f + 2.0f * x ** 3.0f + 4.0f * x ** 4.0f
    let trgt = Tensor.concat 1 [trgt1.[*, NewAxis]; trgt2.[*, NewAxis]]
    
    // create dataset
    let data = {Trgt=trgt; Inp=inp}
    printfn "data: trgt: %A   inp: %A" data.Trgt.Shape data.Inp.Shape
    let dataset = Dataset.ofData data
    printfn "dataset: %A" dataset
    let datasetParts = TrnValTst.ofDataset dataset
    printfn "dataset parts: %A" datasetParts

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
    let loss = loss |> Expr<float>.convert
    printfn "loss: %s\n" (loss.ToString())

    // parameter set
    let parSet = ParSet.fromExpr ContextPath.root loss
    let sizeEnv = Map [
        nFeatures, Size.fix inp.Shape.[1]
        nOutputs, Size.fix trgt.Shape.[1]
    ]
    let parSetInst = ParSet.inst ContextPath.root sizeEnv parSet

    // train
    printfn "Training..."
    let trainable = 
        Trainable (
            primaryLoss=loss, 
            pars=parSetInst, 
            optCfg=Adam.Cfg.standard,
            varEnvForSmpl=fun (smpl: Data) -> 
                VarEnv.ofSeq [inputVar, smpl.Inp; targetVar, smpl.Trgt])
    let cfg = {
        Cfg.standard with
            Termination = Forever
            MaxIters = Some 2000
    }
    let res = trainable.Train datasetParts cfg
    printfn "Result:\n%A" res
    0
