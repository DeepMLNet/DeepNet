#if RELEASE

#r "Tensor/bin/Release/ManagedCuda.dll"
#r "Tensor/bin/Release/Tensor.dll"
#r "SymTensor/bin/Release/SymTensor.dll"
#r "SymTensorCuda/bin/Release/SymTensorCuda.dll"
#r "MLOptimizers/bin/Release/MLOptimizers.dll"
#r "MLModels/bin/Release/MLModels.dll"
#r "MLDatasets/bin/Release/MLDatasets.dll"

#else

#r "Tensor/bin/Debug/ManagedCuda.dll"
#r "Tensor/bin/Debug/Tensor.dll"
#r "SymTensor/bin/Debug/SymTensor.dll"
#r "SymTensorCuda/bin/Debug/SymTensorCuda.dll"
#r "MLOptimizers/bin/Debug/MLOptimizers.dll"
#r "MLModels/bin/Debug/MLModels.dll"
#r "MLDatasets/bin/Debug/MLDatasets.dll"

#endif


