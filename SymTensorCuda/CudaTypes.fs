namespace SymTensor.Compiler.Cuda

open System
open System.Runtime.InteropServices
open ManagedCuda
open ManagedCuda.BasicTypes

open Basics
open Basics.Cuda
open ArrayNDNS
open SymTensor
open SymTensor.Compiler


#nowarn "9"


[<AutoOpen>]
module Types =

    /// device memory pointer
    type DevMemPtrT = {
        /// base memory
        Base: MemManikinT;
        /// offset in elements
        Offset: int
    }

    /// pre-allocated host memory 
    type HostExternalMemT = {Name: string}
    /// host memory pointer
    type HostMemPtrT = {
        Base: HostExternalMemT;
        Offset: int
    }

    /// additional environment informations for CUDA
    type CudaCompileEnvT = {VarStorLoc: Map<UVarSpecT, ArrayLocT>}

    /// function domain (kernel only or host code that may call kernels)
    type FuncDomainT =
        | KernelFunc
        | CPPFunc

    /// template instantiation specification
    type TmplInstT = {
        FuncName:       string; 
        Domain:         FuncDomainT; 
        TmplArgs:       string list; 
        RetType:        string; 
        ArgTypes:       string list;
    }

    /// a CUDA compute stream
    type StreamT = int

    /// CUDA event object
    type EventObjectT = int

    /// a CUDA event that can be used for synchronization
    type EventT = {
        EventObjectId: int; 
        CorrelationId: int;     
        EmittingExecUnitId: int;
    }

    /// Actual CUDA internal memory allocations and external device and host references
    type CudaExecEnvT = {
        Stream:                 Dictionary<StreamT, CudaStream>;
        Event:                  Dictionary<EventObjectT, CudaEvent>;
        InternalMem:            Dictionary<MemAllocManikinT, CudaDeviceVariable<byte>>;
        mutable ExternalVar:    Map<UVarSpecT, IArrayNDCudaT>;
        mutable HostVar:        Map<UVarSpecT, IArrayNDHostT>;
    }
    
    /// CUDA device memory range
    type DevMemRngT = {
        DeviceMem:              CudaDeviceVariable<byte>;
        OffsetInBytes:          int;
        LengthInBytes:          int;
    }

    /// CUDA host memory range
    type HostMemRngT = {
        HostMem:                CudaRegisteredHostMemory<byte>;
        OffsetInBytes:          int;
        LengthInBytes:          int;
    }


    /// BLAS transpose operation
    type BlasTransposeOpT =
        | BlasId
        | BlasTranspose

        member this.CudaBlasOperation =
            match this with
            | BlasId -> CudaBlas.Operation.NonTranspose
            | BlasTranspose -> CudaBlas.Operation.Transpose

    /// specifies the name of the C++ function
    type CPPFuncNameAttribute (cppFuncName: string) =
        inherit System.Attribute()     
        member this.CPPFuncName = cppFuncName




module CudaExecEnv = 

    /// gets device memory for an internal allocation or external reference
    let getDevMemForManikin (env: CudaExecEnvT) (manikin: ArrayNDManikinT) =
        match manikin.Storage with
        | MemAlloc im -> env.InternalMem.[im]
        | MemExternal vs ->
            let ev = env.ExternalVar.[vs]
            if ArrayND.offset ev = 0 && ArrayND.isContiguous ev then
                ev.Storage.ByteData
            else
                failwithf "external variable %A was expected to be contiguous \
                           with zero offset" vs 

    /// gets host memory for an external reference
    let getHostRegMemForManikin (env: CudaExecEnvT) (manikin: ArrayNDManikinT) =
        match manikin.Storage with
        | MemExternal vs ->
            let hv = env.HostVar.[vs]
            if ArrayND.offset hv = 0 && ArrayND.isContiguous hv then
                ArrayNDHostReg.getCudaRegisteredMemory hv
            else
                failwithf "host variable %A was expected to be contiguous \
                           with zero offset" vs
        | _ -> failwithf "host variable must be of type ExternalMem" 


[<AutoOpen>]
module ArgTemplates =

    /// CUDA C++ argument template
    type ICudaArgTmpl =
        abstract member CPPTypeName : string
        abstract member GetArg : CudaExecEnvT -> StreamT -> obj 

    /// CUDA C++ argument template for values that are passed by value in an array
    type ICudaArrayMemberArgTmpl<'T when 'T :> ValueType> =
        abstract member CPPTypeName : string
        abstract member GetArg : CudaExecEnvT -> 'T

    /// CUDA C++ operation functor description
    type ICudaOp =
        abstract member IsIndexed : bool  

    /// CUDA device memory range template
    type IDevMemRngTmpl =
        abstract member GetRng : CudaExecEnvT -> DevMemRngT

    /// CUDA host memory range template
    type IHostMemRngTmpl =
        abstract member GetRng : CudaExecEnvT -> HostMemRngT

    [<Struct>]
    [<type: StructLayout(LayoutKind.Sequential, Pack=4)>]
    /// C++ ArrayND with static shape and static offset/stride
    type ArrayNDSSArg =
        val Dummy : IntPtr 
        val Data : IntPtr
        new (data: IntPtr) = {Dummy = nativeint 0xdeaddead; Data = data}

    /// Literal C++ typename 
    type CPPTemplateValue (cppTypeName) =
        interface ICudaArgTmpl with
            member this.CPPTypeName = cppTypeName
            member this.GetArg env strm = failwith "TemplateValue has no argument value"

    /// ArrayND argument template
    type ArrayNDArgTmpl (manikin: ArrayNDManikinT) = 
        // TShape is ShapeStaicXD and TStride is StrideStaticXD.
        interface ICudaArgTmpl with
            member this.CPPTypeName = manikin.CPPType
            member this.GetArg env strm =
                // C++ struct just contains the pointer to data memory
                let ptr = (CudaExecEnv.getDevMemForManikin env manikin).DevicePointer |> CudaSup.getIntPtr
                //printfn "Passing pointer 0x%x" ptr
                ArrayNDSSArg(ptr) :> obj

    type ArrayNDSDArgTmpl (manikin: ArrayNDManikinT) =
        // TShape is ShapeStaicXD and TStride is StrideDynamicXD.
        interface ICudaArgTmpl with
            member this.CPPTypeName = manikin.DynamicCPPType
            member this.GetArg env strm =
                // currently this cannot be passed as an argument
                failwith "passing ArrayNDSDArg is not implemented"

    type SizeTPtrFromArrayNDIdxTmpl (manikinOpt: ArrayNDManikinT option) = 
        do 
            match manikinOpt with
            | Some manikin ->
                if manikin.DataType <> typeof<int> then 
                    failwith "SizeTPtrFromArrayNDIdxTmpl manikin must be of type int"
                if ArrayND.nDims manikin <> 0 then 
                    failwith "SizeTPtrFromArrayNDIdxTmpl manikin must be a scalar"
                if ArrayND.offset manikin <> 0 then 
                    failwith "SizeTPtrFromArrayNDIdxTmpl manikin must have zero offset"
            | None -> ()

        interface ICudaArrayMemberArgTmpl<IntPtr> with
            member this.CPPTypeName = "size_t *"
            member this.GetArg env =
                match manikinOpt with
                | Some manikin ->
                    // pass pointer to (only) element
                    (CudaExecEnv.getDevMemForManikin env manikin).DevicePointer |> CudaSup.getIntPtr
                | None -> IntPtr.Zero

    type CPPArrayTmpl<'T when 'T :> ValueType> (valueTmpls: ICudaArrayMemberArgTmpl<'T> list) =       
        interface ICudaArgTmpl with
            member this.CPPTypeName = 
                sprintf "Array<%s, %d>" (valueTmpls.Head.CPPTypeName) (List.length valueTmpls)
            member this.GetArg env strm =
                let argVals =
                    valueTmpls
                    |> List.map (fun vt -> vt.GetArg env)
                    |> List.toArray
                PassArrayByVal.passArrayByValue argVals

    type NullPtrArgTmpl () =
        interface ICudaArgTmpl with
            member this.CPPTypeName = "void *"
            member this.GetArg env strm = box (System.IntPtr 0) 

    type BytePtrArgTmpl (memManikin) =
        interface ICudaArgTmpl with
            member this.CPPTypeName = "char *"
            member this.GetArg env strm = 
                let storage = 
                    match memManikin with
                    | MemAlloc im -> env.InternalMem.[im]
                    | MemExternal vs -> env.ExternalVar.[vs].Storage.ByteData
                storage.DevicePointer |> CudaSup.getIntPtr |> box

    type SizeTArgTmpl (value: int) =
        interface ICudaArgTmpl with
            member this.CPPTypeName = "size_t"
            member this.GetArg env strm = box (nativeint value) 

    type ExecStreamArgTmpl () =
        interface ICudaArgTmpl with
            member this.CPPTypeName = "CUstream"
            member this.GetArg env strm = 
                box env.Stream.[strm].Stream 

    /// device memory range over the elements of a contiguous ArrayND
    type ArrayNDDevMemRngTmpl (manikin: ArrayNDManikinT) =
        do if not (ArrayND.isContiguous manikin) then failwith "manikin for MemRng is not contiguous"
        interface IDevMemRngTmpl with
            member this.GetRng env =
                {DeviceMem = CudaExecEnv.getDevMemForManikin env manikin;
                 OffsetInBytes = ArrayNDManikin.offsetInBytes manikin;
                 LengthInBytes = ArrayNDManikin.sizeInBytes manikin;}
    
    /// registered host memory range over the elements of a contiguous ArrayND    
    type ArrayNDHostRegMemRngTmpl (manikin: ArrayNDManikinT) =
        do if not (ArrayND.isContiguous manikin) then failwith "manikin for MemRng is not contiguous"
        interface IHostMemRngTmpl with
            member this.GetRng env =
                {HostMem = CudaExecEnv.getHostRegMemForManikin env manikin;
                 OffsetInBytes = ArrayNDManikin.offsetInBytes manikin;
                 LengthInBytes = ArrayNDManikin.sizeInBytes manikin;}      

    /// BLAS view of ArrayND. The ArrayND is implicitly transposed and exposed as a "float *"
    type BlasTransposedMatrixTmpl (manikin: ArrayNDManikinT) =
        // All CUBLAS calls use Fortran matrices. This means:
        // - one-based indexing
        // - column major
        // For ArrayND this translates to:
        // CUBLAS #columns    = Shape.[0]
        // CUBLAS #rows       = Shape.[1]
        // CUBLAS leading dim = Stride.[0] >= 1 (no broadcasting)
        // Stride.[1] must be 1.

        do
            if not ((manikin |> ArrayNDManikin.typeName |> TypeName.getType).Equals(typeof<single>)) then
                failwith "CUBLAS currently requires single values"
            match ArrayND.stride manikin with
            | [0; _] -> failwithf "ArrayND for use with BLAS cannot be broadcasted in first dimension"
            | [_; n] when n <> 1 -> failwithf "ArrayND for use with BLAS must be continguous in last dimension but has stride %d" n
            | [_; _] -> ()
            | _ -> failwith "ArrayND for use with BLAS must be 2-dimensional"         

        member this.GetLeadingDimension env =
            (ArrayND.stride manikin).[0] 

        member this.GetColumns env =
            (ArrayND.shape manikin).[0]

        member this.GetRows env =
            (ArrayND.shape manikin).[1]

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
            member this.GetArg env strm = 
                let devVar = CudaExecEnv.getDevMemForManikin env manikin
                // need to adjust by offset
                let offsetBytes = ArrayNDManikin.offsetInBytes manikin
                new CudaDeviceVariable<single>(devVar.DevicePointer + BasicTypes.SizeT(offsetBytes), 
                                               devVar.SizeInBytes - offsetBytes) :> obj


    [<Struct>]
    [<type: StructLayout(LayoutKind.Sequential, Pack=4)>]
    /// const value elementwise operation C++ structure
    type ConstEOpArg<'T when 'T: struct> =
        val Value: 'T
        new (value: 'T) = {Value = value}

    type ConstEOpArgTmpl<'T> (value: 'T) =
        interface ICudaArgTmpl with
            member this.CPPTypeName = "ConstEOp_t"
            member this.GetArg env strm = 
                match box value with
                | :? single as n -> ConstEOpArg(n) :> obj
                | :? double as n -> ConstEOpArg(n) :> obj
                | :? int as n -> ConstEOpArg(n) :> obj
                | :? byte as n -> ConstEOpArg(n) :> obj
                | _ -> failwithf "unsupported type %A" (value.GetType())
        interface ICudaOp with
            member this.IsIndexed = false

    [<Struct>]
    [<type: StructLayout(LayoutKind.Sequential, Pack=4)>]
    type NoArgEOpArg = struct end
    
    type NoArgEOpArgTmpl (cppTypeName: string, indexed: bool) =
        interface ICudaArgTmpl with
            member this.CPPTypeName = cppTypeName
            member this.GetArg env strm = NoArgEOpArg() :> obj
        interface ICudaOp with
            member this.IsIndexed = indexed


[<AutoOpen>]
module NativeFunctionDelegates =

    [<CPPFuncName("sum")>]
    type CPPSum = delegate of ArrayNDSSArg * ArrayNDSSArg * CUstream * IntPtr * nativeint -> unit

    [<CPPFuncName("sumLastAxis")>]
    type CPPSumLastAxis = delegate of ArrayNDSSArg * ArrayNDSSArg -> unit
