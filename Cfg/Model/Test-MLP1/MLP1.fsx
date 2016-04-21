#if !CONFIG
#I "../../../CurveFollow/bin/Debug"
#endif 

#r "Models.dll"
#r "SymTensor.dll"
#r "CurveFollow.exe"

open SymTensor
open Models
open Controller
open Models.Train

let mlpControllerCfg = {
    MLP = 
        {Layers = 
          [ {NInput = Controller.nBiotac; NOutput = SizeSpec.fix 100;     TransferFunc=NeuralLayer.Tanh}
            {NInput = SizeSpec.fix 100;   NOutput = Controller.nTarget;   TransferFunc=NeuralLayer.Identity} ]
         LossMeasure = LossLayer.MSE}
}

let trainCfg = {
    Models.Train.defaultCfg with
        Seed        = 1
        BatchSize   = 10000
        MinIters    = Some 2000
        LearningRates = [1e-2f; 1e-3f; 1e-4f; 1e-5f; 1e-6f]
}


let cfg = {
    TrnDirs        = [Config.baseDir + "/Data/DeepBraille/Movements/test1/ManSet/trn"]
    ValDirs        = [Config.baseDir + "/Data/DeepBraille/Movements/test1/ManSet/val"]
    TstDirs        = [Config.baseDir + "/Data/DeepBraille/Movements/test1/ManSet/tst"]
    DatasetCache   = Some (Config.baseDir + "/Data/DeepBraille/Movements/test1/ManSet/Cache")
    DownsampleFactor = 20
    MLPControllerCfg = mlpControllerCfg
    TrainCfg       = trainCfg
    ModelFile      = Config.cfgDir + "/model.h5"
}
