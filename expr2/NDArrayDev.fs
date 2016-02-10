module NDArrayDev

open Util
open ManagedCuda
open NDArray


/// an NDArray on the GPU device
type NDArrayDev = 
    {Shape: int list;
     Stride: int list; 
     Offset: int;
     Data: CudaDeviceVariable<single>}

