open Tensor
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers


[<EntryPoint>]
let main argv =

    //let device = DevHost
    let device = DevCuda

    Debug.VisualizeUExpr <- true


    let mnist = Mnist.load ("../../../../Data/MNIST") 0.1
    let mnist = if device = DevCuda then TrnValTst.toCuda mnist else mnist

    // define symbolic sizes
    let mb = ModelBuilder<single> "NeuralNetModel"
    let nBatch  = mb.Size "nBatch"
    let nInput  = mb.Size "nInput"
    let nClass  = mb.Size "nClass"

    // define model parameters
    let mlp = 
        MLP.pars (mb.Module "MLP") 
            { Layers = [{NeuralLayer.defaultHyperPars with NInput=nInput; NOutput=nClass; TransferFunc=ActivationFunc.Tanh}]
              LossMeasure = LossLayer.MSE }

    // define variables
    let input1 : ExprT = mb.Var<single> "Input1" [nBatch; nInput]
    let input2 : ExprT = mb.Var<single> "Input2" [nBatch; nInput]
    let target : ExprT = mb.Var<single> "Target" [nBatch; nClass]


    let inputB = Expr.sumKeepingAxis 1 input1
    let a0, a1 = ElemExpr.arg2<single>
    let i0, i1 = ElemExpr.idx2
    let input3 = 
        Expr.elements [nBatch; nInput] (a0[i0; i1] * a1[i0; SizeSpec.fix 0L]) [input1; inputB]

    // instantiate model
    mb.SetSize nInput mnist.Trn.[0L].Input.Shape.[0]
    mb.SetSize nClass mnist.Trn.[0L].Target.Shape.[0]
    let mi = mb.Instantiate device

    let pred = MLP.pred mlp input3
    let loss = MLP.loss mlp input3 target

    let smplVarEnv (smpl: InputTargetSampleT) =
        VarEnv.empty
        |> VarEnv.add input1 smpl.Input
        |> VarEnv.add input2 smpl.Input
        |> VarEnv.add target smpl.Target
    let trainable =
        Train.trainableFromLossExpr mi loss smplVarEnv GradientDescent.New GradientDescent.DefaultCfg
        
    let trainCfg : Train.Cfg = {    
        Train.defaultCfg with
            Seed               = 100   
            //BatchSize          = 10 
            BatchSize          = 10000L
            LossRecordInterval = 10                                   
            Termination        = Train.ItersWithoutImprovement 100
            MinImprovement     = 1e-7  
            TargetLoss         = None  
            MinIters           = Some 100 
            MaxIters           = None  
            LearningRates      = [1e-3; 1e-4; 1e-5]                               
    } 

    //Debug.Timing <- true
    //Debug.TraceCompile <- true
    //Debug.VisualizeUExpr <- true
    //Debug.DisableCombineIntoElementsOptimization <- true
    Debug.VisualizeExecItems <- true
    //Debug.TerminateAfterCompilation <- true

    let lossFn = mi.Func loss |> arg3 input1 input2 target
    let initialLoss = lossFn mnist.Trn.All.Input mnist.Trn.All.Input mnist.Trn.All.Target |> Tensor.value
    printfn "Initial training loss: %f" initialLoss

    printfn "Computing dLoss / dInput1."
    Debug.VisualizeUExpr <- true
    let dLoss = Deriv.compute loss |> Deriv.ofVar input1
    let dLossFn = mi.Func dLoss |> arg3 input1 input2 target
    let dLossVal = 
        dLossFn (mnist.Trn.Part(0L,10L).All.Input) (mnist.Trn.Part(0L,10L).All.Input) (mnist.Trn.Part(0L,10L).All.Target)
    printfn "Done."

    let result = Train.train trainable mnist trainCfg


    0

