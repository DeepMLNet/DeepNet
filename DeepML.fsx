#if RELEASE

#r "Basics/bin/Release/Basics.dll"
#r "ArrayND/bin/Release/ArrayND.dll"
#r "SymTensor/bin/Release/SymTensor.dll"
#r "SymTensorCuda/bin/Release/SymTensorCuda.dll"
#r "Optimizers/bin/Release/Optimizers.dll"
#r "Models/bin/Release/Models.dll"
#r "Datasets/bin/Release/Datasets.dll"

#else

#r "Tensor/bin/Debug/ManagedCuda.dll"
#r "Tensor/bin/Debug/Tensor.dll"
#r "SymTensor/bin/Debug/SymTensor.dll"
#r "SymTensorCuda/bin/Debug/SymTensorCuda.dll"
#r "MLOptimizers/bin/Debug/MLOptimizers.dll"
#r "MLModels/bin/Debug/MLModels.dll"
#r "MLDatasets/bin/Debug/MLDatasets.dll"

#endif

#r "packages/Argu.2.1/lib/net40/Argu.dll"
#load "packages/FSharp.Charting.0.90.13/FSharp.Charting.fsx"

