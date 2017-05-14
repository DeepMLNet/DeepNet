open Tensor
open Tensor.Utils
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets
open Optimizers


[<EntryPoint>]
let main argv =

    //Debug.VisualizeUExpr <- true
    Debug.DisableCombineIntoElementsOptimization <- true
    //Debug.DisableOptimizer <- true

    let device = DevHost
    //let device = DevCuda

    let mnist = Mnist.load (Util.assemblyDirectory + "/../../../../Data/MNIST") 0.1

    let mnist = if device = DevCuda then TrnValTst.toCuda mnist else mnist

    printfn "Building model..."
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

    // prediction
    let pred = MLP.pred mlp input
    let predFn = mi.Func pred |> arg1 input

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

    let misclassificationRate inputs targets =
        let preds = predFn inputs |> HostTensor.transfer |> Tensor.argMaxAxis 1
        let targets = targets |> HostTensor.transfer |> Tensor.argMaxAxis 1
        let isErr = Tensor.ifThenElse (preds ==== targets) (Tensor.zerosLike preds) (Tensor.onesLike preds)
        let errCnt = Tensor.sum isErr |> Tensor.value
        let errRate = float errCnt / float inputs.Shape.[0]
        errRate        

    let userQuality iter =
        let qual = {
            TrnQuality = misclassificationRate mnist.Trn.All.Input mnist.Trn.All.Target
            ValQuality = misclassificationRate mnist.Val.All.Input mnist.Val.All.Target
            TstQuality = misclassificationRate mnist.Tst.All.Input mnist.Tst.All.Target
        }
        Map ["misclassificationRate", qual]        

    let trainCfg : Train.Cfg = {    
        Train.defaultCfg with
            Seed               = 100   
            //BatchSize          = 10 
            BatchSize          = 10000L
            LossRecordInterval = 10               
            UserQualityFunc    = userQuality
            Termination        = Train.ItersWithoutImprovement 100
            MinImprovement     = 1e-7  
            TargetLoss         = None  
            MinIters           = Some 100 
            MaxIters           = None  
            LearningRates      = [1e-3; 1e-4; 1e-5]       
            CheckpointFile     = Some "MNIST-%ITER%.h5"
            CheckpointInterval = Some 50
            DiscardCheckpoint  = true
    } 

    //Debug.Timing <- true
    //Debug.TraceCompile <- true
    //Debug.VisualizeUExpr <- true
    //Debug.VisualizeExecItems <- true
    //Debug.TerminateAfterCompilation <- true
    //let ts = Trace.startSession "LearnMnist"

    let lossFn = mi.Func loss |> arg2 input target
    printfn "Calculating initial loss..."
    let initialLoss = lossFn mnist.Trn.All.Input mnist.Trn.All.Target |> Tensor.value
    printfn "Initial training loss: %f" initialLoss
    let result = Train.train trainable mnist trainCfg

    //let ts = ts.End ()
    //ts |> Trace.dumpToFile "LearnMNIST.txt"

    0

