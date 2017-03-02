open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers


[<EntryPoint>]
let main argv =

    //let device = DevHost
    let device = DevCuda

    let mnist = Mnist.load ("../../../../Data/MNIST") 0.1

    let mnist = if device = DevCuda then TrnValTst.toCuda mnist else mnist

    let mb = ModelBuilder<single> "NeuralNetModel"

    // define symbolic sizes
    let nBatch  = mb.Size "nBatch"
    let nInput  = mb.Size "nInput"
    let nClass  = mb.Size "nClass"
    let nHidden = mb.Size "nHidden"

    // define model parameters
    let mlp = 
        MLP.pars (mb.Module "MLP") 
            { Layers = [{NeuralLayer.defaultHyperPars with NInput=nInput; NOutput=nHidden; TransferFunc=ActivationFunc.Tanh}
                        {NeuralLayer.defaultHyperPars with NInput=nHidden; NOutput=nClass; TransferFunc=ActivationFunc.SoftMax}]
              LossMeasure = LossLayer.CrossEntropy }

    // define variables
    let input  : ExprT = mb.Var<single> "Input"  [nBatch; nInput]
    let target : ExprT = mb.Var<single> "Target" [nBatch; nClass]

    // instantiate model
    mb.SetSize nInput mnist.Trn.[0L].Input.Shape.[0]
    mb.SetSize nClass mnist.Trn.[0L].Target.Shape.[0]
    mb.SetSize nHidden 100L
    let mi = mb.Instantiate device

    // loss expression
    let loss = MLP.loss mlp input target
    //let loss = loss |> Expr.checkFinite "loss"

    let smplVarEnv (smpl: InputTargetSampleT) =
        VarEnv.empty
        |> VarEnv.add input smpl.Input
        |> VarEnv.add target smpl.Target
    let trainable =
        //Train.trainableFromLossExpr mi loss smplVarEnv GradientDescent.New GradientDescent.DefaultCfg
        Train.trainableFromLossExpr mi loss smplVarEnv Adam.New Adam.DefaultCfg

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

//    let ts = Trace.startSession "LearnMnist"

    let result = Train.train trainable mnist trainCfg

//    let ts = ts.End ()
//    ts |> Trace.dumpToFile "LearnMNIST.txt"

    0

