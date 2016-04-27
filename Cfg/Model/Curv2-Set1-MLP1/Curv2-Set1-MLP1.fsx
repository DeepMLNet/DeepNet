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
        LearningRates = [1e-2; 1e-3; 1e-4; 1e-5; 1e-6]
        CheckpointDir = Some (Config.cfgDir + "/checkpoint")
}


let mDir = Config.baseDir + "/Data/DeepBraille/Movements/Curv2-Set1"

let cfg = {
    TrnDirs        = [for i=0 to 7 do yield sprintf "%s/%02d.cur" mDir i]
    ValDirs        = [mDir + "/08.cur"]
    TstDirs        = [mDir + "/09.cur"]
    DatasetCache   = Some (mDir + "/Cache")
    DownsampleFactor = 20
    MLPControllerCfg = mlpControllerCfg
    TrainCfg       = trainCfg
    ModelFile      = Config.cfgDir + "/model.h5"
    TrnResultFile  = Config.cfgDir + "/result.json"
}
