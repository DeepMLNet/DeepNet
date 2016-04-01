#r "../../../CurveFollow/bin/Debug/CurveFollow.exe"
#r "../../../CurveFollow/bin/Debug/Models.dll"
#r "../../../CurveFollow/bin/Debug/SymTensor.dll"

open Controller
open Program
open SymTensor
open Models


let cfg = {
    DatasetDir       = "../../../Data/DeepBraille/curv2"
    MLPControllerCfg = 
      {MLP = 
        {Layers = 
          [ {NInput = Controller.nBiotac; NOutput = SizeSpec.fix 100;     TransferFunc=NeuralLayer.Tanh}
            {NInput = SizeSpec.fix 100;   NOutput = Controller.nVelocity; TransferFunc=NeuralLayer.Identity} ]
         LossMeasure = LossLayer.MSE}
       BatchSize      = 1000
       Seed           = 1
       Iters          = 1000
       StepSize       = 1e-3f}
}

