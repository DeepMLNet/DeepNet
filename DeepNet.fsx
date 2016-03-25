#if RELEASE

#r "Basics/bin/Release/Basics.dll"
#r "ArrayND/bin/Release/ArrayND.dll"
#r "SymTensor/bin/Release/SymTensor.dll"
#r "SymTensorCuda/bin/Release/SymTensorCuda.dll"
#r "Optimizers/bin/Release/Optimizers.dll"
#r "Models/bin/Release/Models.dll"
#r "Datasets/bin/Release/Datasets.dll"

#else

#r "Basics/bin/Debug/ManagedCuda.dll"
#r "Basics/bin/Debug/Basics.dll"
#r "ArrayND/bin/Debug/ArrayND.dll"
#r "SymTensor/bin/Debug/SymTensor.dll"
#r "SymTensorCuda/bin/Debug/SymTensorCuda.dll"
#r "Optimizers/bin/Debug/Optimizers.dll"
#r "Models/bin/Debug/Models.dll"
#r "Datasets/bin/Debug/Datasets.dll"

#endif

#r "packages/Argu.2.1/lib/net40/Argu.dll"
#load "packages/FSharp.Charting.0.90.13/FSharp.Charting.fsx"

