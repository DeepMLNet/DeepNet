namespace SymTensor.Compiler.Cuda

open System
open System.Runtime.InteropServices
open ManagedCuda

open Util
open ArrayNDNS
open SymTensor
open SymTensor.Compiler



[<AutoOpen>]
module CudaTypes =

    /// dimensionality of parallel work to perform
    type WorkDimT = int * int * int

    /// device memory pointer
    type DevMemPtrT = {
        /// base memory
        Base: MemManikinT;
        /// offset in elements
        Offset: int}

    /// pre-allocated host memory 
    type HostExternalMemT = {Name: string}
    /// host memory pointer
    type HostMemPtrT = {Base: HostExternalMemT;
                        Offset: int}


    /// variable storage location
    type VarStorLocT =
        | LocDev
        | LocHost

    /// additional environment informations for CUDA
    type CudaEnvT = {VarStorLoc: Map<IVarSpec, VarStorLocT>}

    /// function domain (kernel only or host code that may call kernels)
    type FuncDomainT =
        | KernelFunc
        | CPPFunc

    /// template instantiation specification
    type TmplInstT = {FuncName: string; Domain: FuncDomainT; 
                      TmplArgs: string list; RetType: string; ArgTypes: string list;}


    /// Actual CUDA internal memory allocations and external device and host references
    type CudaExecEnvT = 
        {InternalMem: Dictionary<MemAllocManikinT, CudaDeviceVariable<byte>>;
         ExternalVar: Map<IVarSpec, ArrayNDCuda.IArrayNDCudaT>;
         HostVar:     Map<IVarSpec, ArrayNDHost.IArrayNDHostT>}


    /// CUDA C++ argument template
    type ICudaArgTmpl =
        abstract member CPPTypeName : string
        abstract member GetArg : CudaExecEnvT -> obj 

    /// CUDA device memory range
    type DevMemRngT = 
        {DeviceVar: CudaDeviceVariable<byte>;
         OffsetInBytes: int;
         LengthInBytes: int;}

    /// CUDA device memory range template
    type IDevMemRngTmpl =
        abstract member GetRng : CudaExecEnvT -> DevMemRngT

    /// CUDA host memory range
    type HostMemRngT = 
        {HostVar: CudaRegisteredHostMemory<byte>;
         OffsetInBytes: int;
         LengthInBytes: int;}

    /// CUDA host memory range template
    type IHostMemRngTmpl =
        abstract member GetRng : CudaExecEnvT -> HostMemRngT


    /// BLAS transpose operation
    type BlasOpT =
        | BlasId
        | BlasTranspose

        member this.CudaBlasOperation =
            match this with
            | BlasId -> CudaBlas.Operation.NonTranspose
            | BlasTranspose -> CudaBlas.Operation.Transpose



module CudaExecEnv = 

    /// gets device memory for an internal allocation or external reference
    let getDevVar (env: CudaExecEnvT) (manikin: ArrayNDManikinT) =
        match manikin.Storage with
        | MemAlloc im -> env.InternalMem.[im]
        | MemExternal vs ->
            let ev = env.ExternalVar.[vs]
            if (ArrayND.shape ev) = (ArrayND.shape manikin) && 
                    (ArrayND.stride ev) = (ArrayND.stride manikin) && 
                    (ArrayND.offset ev) = (ArrayND.offset manikin) then
                (ev :?> ArrayNDCuda.IDeviceStorage).ByteData
            else
                failwithf "external variable is of form %A but form %A was expected" ev manikin

    /// gets host memory for an external reference
    let getHostVar (env: CudaExecEnvT) (manikin: ArrayNDManikinT) =
        match manikin.Storage with
        | MemExternal vs ->
            let hv = env.HostVar.[vs]
            if (ArrayND.shape hv) = (ArrayND.shape manikin) && 
                    (ArrayND.stride hv) = (ArrayND.stride manikin) && 
                    (ArrayND.offset hv) = (ArrayND.offset manikin) then
                ArrayNDHostReg.getCudaRegisteredMemory hv
            else
                failwithf "host variable is of form %A but form %A was expected" hv manikin
        | _ -> failwithf "host variable must be of type ExternalMem"


[<AutoOpen>]
module CudaArgTemplates =


    // All CUBLAS calls use Fortran matrices. This means:
    // - one-based indexing
    // - column major
    // For NDArray this translates to:
    // CUBLAS #columns    = Shape.[0]
    // CUBLAS #rows       = Shape.[1]
    // CUBLAS leading dim = Stride.[0] >= 1 (no broadcasting)
    // Stride.[1] must be 1.

    /// BLAS view of NDArray. The NDArray is implicitly transposed and exposed as a "float *"
    type BlasTransposedMatrixTmpl (view: ArrayNDManikinT) =
        do
            match ArrayND.stride view with
            | [0; _] -> failwithf "ArrayND for use with BLAS cannot be broadcasted in first dimension"
            | [_; n] when n <> 1 -> failwithf "ArrayND for use with BLAS must be continguous in last dimension but has stride %d" n
            | [_; _] -> ()
            | _ -> failwith "ArrayND for use with BLAS must be 2-dimensional"         

        member this.GetLeadingDimension env =
            (ArrayND.stride view).[0] 

        member this.GetColumns env =
            (ArrayND.shape view).[0]

        member this.GetRows env =
            (ArrayND.shape view).[1]

        member this.GetColumnsForOp env op =
            match op with 
            | CudaBlas.Operation.NonTranspose -> this.GetColumns env
            | CudaBlas.Operation.Transpose 
            | CudaBlas.Operation.ConjugateTranspose -> this.GetRows env
            | _ -> failwithf "unknown CudaBlas.Operation %A" op

        member this.GetRowsForOp env op =
            match op with 
            | CudaBlas.Operation.NonTranspose -> this.GetRows env
            | CudaBlas.Operation.Transpose 
            | CudaBlas.Operation.ConjugateTranspose -> this.GetColumns env
            | _ -> failwithf "unknown CudaBlas.Operation %A" op

        interface ICudaArgTmpl with
            member this.CPPTypeName = "float"
            member this.GetArg env = 
                let devVar = CudaExecEnv.getDevVar env view
                // need to adjust by offset
                let offsetBytes = view.Offset * 4
                new CudaDeviceVariable<single>(devVar.DevicePointer + BasicTypes.SizeT(offsetBytes), 
                                               devVar.SizeInBytes - offsetBytes) :> obj

