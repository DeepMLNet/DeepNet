﻿module CudaExecUnits

open System.Runtime.InteropServices
open ManagedCuda
open Util
open Op
open ExecUnitsGen
open CudaRegMem
open System

/// device memory pointer
type DevMemPtrT = {Base: MemoryT;
                   Offset: int}

/// pre-allocated host memory 
type HostExternalMemT = {Name: string}
/// host memory pointer
type HostMemPtrT = {Base: HostExternalMemT;
                    Offset: int}

/// variable storage location
type VarStorLocT =
    | DevVar
    | HostVar

/// additional environment informations for CUDA
type CudaEnvT = {VarStorLoc: Map<VarSpecT, VarStorLocT>}

/// template instantiation specification
type TmplInstT = {FuncName: string; TmplArgs: string list; 
                  RetType: string; ArgTypes: string list;}

/// dimensionality of parallel work to perform
type WorkDimT = int * int * int

/// Actual CUDA internal memory allocations and external device and host references
type CudaExecEnvT = 
    {InternalMem: Dictionary<MemAllocT, CudaDeviceVariable<byte>>;
     ExternalVar: Map<VarSpecT, NDArrayDev.NDArrayDev>;
     HostVar:     Map<VarSpecT, NDArray.NDArray>}

module CudaExecEnv = 
    /// gets device memory for an internal allocation or external reference
    let getDevVar (env: CudaExecEnvT) view =
        match view.Memory with
        | MemAlloc im -> env.InternalMem.[im]
        | ExternalMem vs ->
            let ev = env.ExternalVar.[vs]
            if ev.Shape = view.Shape && ev.Stride = view.Stride && ev.Offset = view.Offset then
                // TODO: remove conversion from single to byte
                new CudaDeviceVariable<byte>(ev.Data.DevicePointer, ev.Data.SizeInBytes)
            else
                failwithf "external variable is of form %A but form %A was expected" ev view

    /// gets host memory for an external reference
    let getHostVar (env: CudaExecEnvT) view =
        match view.Memory with
        | ExternalMem vs ->
            let hv = env.HostVar.[vs]
            if hv.Shape = view.Shape && hv.Stride = view.Stride && hv.Offset = view.Offset then
                // TODO: remove conversion from single to byte
                let cr = NDArrayLock.getCudaRegisteredMemory hv
                new CudaRegisteredHostMemory<byte>(cr.PinnedHostPointer, cr.SizeInBytes)
            else
                failwithf "host variable is of form %A but form %A was expected" hv view
        | _ -> failwithf "host variable must be of type ExternalMem"


/// CUDA C++ argument template
type ICudaArgTmpl =
    abstract member CPPTypeName : string
    abstract member CPPTypeNameWithoutPtr : string
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
    
/// a CUDA operation 
type CudaOpT =
    // memory operations
    | MemcpyDtoD of IDevMemRngTmpl * IDevMemRngTmpl
    | MemcpyHtoD of IHostMemRngTmpl * IDevMemRngTmpl
    | MemcpyDtoH of IDevMemRngTmpl * IHostMemRngTmpl
    | Memset of single * IDevMemRngTmpl
    // kernel execution
    | LaunchKernel of TmplInstT * WorkDimT * (ICudaArgTmpl list)

/// CUDA C++ operation functor description
type ICudaOp =
    abstract member IsIndexed : bool  

/// NDArray pointer marshalling template
type NDArrayPtrArgTmpl (view: NDArrayViewT) = 
    interface ICudaArgTmpl with
        member this.CPPTypeNameWithoutPtr = 
            let dims = NDArrayView.nDim view
            let shapeStr = if dims = 0 then "" else "<" + (view.Shape |> intToStrSeq |> String.combineWith ",") + ">"
            let strideStr = "<" + ((view.Offset :: view.Stride) |> intToStrSeq |> String.combineWith ",") + ">"
            sprintf "NDArray%dD<Shape%dD%s, Stride%dD%s> *" dims dims shapeStr dims strideStr            

        member this.CPPTypeName = (this :> ICudaArgTmpl).CPPTypeNameWithoutPtr + " *"

        member this.GetArg env =
            // C++ structure is empty and we pass the pointer to data memory
            CudaExecEnv.getDevVar env view :> obj

type NDArrayDevMemRngTmpl (view: NDArrayViewT) =
    interface IDevMemRngTmpl with
        member this.GetRng env =
            {DeviceVar = CudaExecEnv.getDevVar env view;
             OffsetInBytes = view.Offset * sizeof<single>;
             LengthInBytes = (NDArrayView.nElems view) * sizeof<single>;}
        
type NDArrayHostMemRngTmpl (view: NDArrayViewT) =
    interface IHostMemRngTmpl with
        member this.GetRng env =
            {HostVar = CudaExecEnv.getHostVar env view;
             OffsetInBytes = view.Offset * sizeof<single>;
             LengthInBytes = (NDArrayView.nElems view) * sizeof<single>;}        


#nowarn "9"
[<Struct>]
[<type: StructLayout(LayoutKind.Sequential, Pack=4)>]
type ConstEOp =
    val Value: single
    new (value: single) = {Value = value}

type ConstEOpTmpl (value: single) =
    interface ICudaArgTmpl with
        member this.CPPTypeName = "ConstEOp_t"
        member this.CPPTypeNameWithoutPtr = (this :> ICudaArgTmpl).CPPTypeName
        member this.GetArg env = ConstEOp(value) :> obj
    interface ICudaOp with
        member this.IsIndexed = false

[<Struct>]
[<type: StructLayout(LayoutKind.Sequential, Pack=4)>]
type NoArgEOp = struct end
    
type NoArgEOpTmpl (cppTypeName: string, indexed: bool) =
    interface ICudaArgTmpl with
        member this.CPPTypeName = cppTypeName
        member this.CPPTypeNameWithoutPtr = (this :> ICudaArgTmpl).CPPTypeName
        member this.GetArg env = NoArgEOp() :> obj
    interface ICudaOp with
        member this.IsIndexed = indexed

/// computes the definitive target view of an op given its source views
let trgtViewGivenSrc cudaEnv memAllocator trgtShape reqView op srcViews srcShared  =
    // target that shares no elements with any srcView
    let outplaceTrgt =
        match reqView with
        | Some rv when not (List.exists (NDArrayView.overlapping rv) srcViews) -> rv, false
        | _ -> NDArrayView.newContinguous memAllocator trgtShape, false        

    // target that reuses a srcView, if it may be overwritten
    let inplaceOverwriteTrgt =
        match List.tryFindIndex not srcShared with
        | Some i -> srcViews.[i], false
        | None -> outplaceTrgt    

    match op with
    // variable access
    | LeafOp (Var vs) ->       
        match cudaEnv.VarStorLoc |> Map.find vs with
        | DevVar ->
            // we assume that all device input vars are continguous
            {Memory=ExternalMem vs; 
             Shape=trgtShape; Offset=0; 
             Stride=NDArray.contiguousStride trgtShape}, true
        | HostVar ->
            // will transfer variable from host to device during execution
            // need continguous memory for that
            match reqView with
            | Some rv when NDArrayView.isContiguous rv -> rv, false
            | _ -> NDArrayView.newContinguous memAllocator trgtShape, false        
    // tensor creation
    | LeafOp _ -> outplaceTrgt        

    // unary elementwise
    | UnaryOp Negate -> inplaceOverwriteTrgt
    | UnaryOp Log -> inplaceOverwriteTrgt
    | UnaryOp Exp -> inplaceOverwriteTrgt
    // reductions
    | UnaryOp Sum -> outplaceTrgt
    | UnaryOp (SumAxis _) -> outplaceTrgt
    // shape operations
    | UnaryOp (Reshape _) ->        
        // TODO: optimize: check if copy is really necessary
        if NDArrayView.isContiguous srcViews.[0] then
            {srcViews.[0] with Shape=trgtShape; Stride=NDArrayView.contiguousStride trgtShape}, srcShared.[0]
        else outplaceTrgt  // will copy
    | UnaryOp (Broadcast _) ->
        let aView, aShared = srcViews.[0], srcShared.[0]
        {aView with Shape=trgtShape; 
                    Stride=List.map3 
                        (fun aStr aShp tShp -> if aShp = tShp then aStr else 0) 
                        aView.Stride aView.Shape trgtShape}, aShared
    | UnaryOp (SwapDim (ax1, ax2)) ->
        let aView, aShared = srcViews.[0], srcShared.[0]
        let str = aView.Stride
        {aView with Shape=trgtShape; 
                    Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1]}, aShared
    // misc
    | UnaryOp (Annotated _) -> srcViews.[0], srcShared.[0]

    // binary elementwise
    | BinaryOp Add -> inplaceOverwriteTrgt
    | BinaryOp Substract -> inplaceOverwriteTrgt
    | BinaryOp Multiply -> inplaceOverwriteTrgt
    | BinaryOp Divide -> inplaceOverwriteTrgt
    | BinaryOp Power -> inplaceOverwriteTrgt
    // matrix/tensor operations
    | BinaryOp Dot -> outplaceTrgt
    | BinaryOp TensorProduct -> outplaceTrgt
      

/// computes desired source views given desired target view
let srcViewReqsGivenTrgt cudaEnv trgtShape reqView op srcShapes =
    let nSrcs = List.length srcShapes

    // requests all sources to use separate storage
    let outplaceTrgt =
        List.replicate nSrcs None

    // requests one source to be evaluated into our target view
    let inplaceOverwriteTrgt =
        match nSrcs with
        | 0 -> []
        | 1 -> [reqView]
        | _ -> reqView :: List.replicate (nSrcs-1) None

    match op with
    | LeafOp _ -> []

    // unary elementwise
    | UnaryOp Negate -> inplaceOverwriteTrgt
    | UnaryOp Log -> inplaceOverwriteTrgt
    | UnaryOp Exp -> inplaceOverwriteTrgt
    // reductions
    | UnaryOp Sum -> outplaceTrgt
    | UnaryOp (SumAxis _) -> outplaceTrgt
    // shape operations
    | UnaryOp (Reshape _) ->        
        match reqView with
        | Some rv when NDArrayView.isContiguous rv ->
            [Some {rv with Shape=srcShapes.[0]; Stride=NDArrayView.contiguousStride srcShapes.[0]}]
        | _ -> outplaceTrgt
    | UnaryOp (Broadcast _) -> outplaceTrgt
    | UnaryOp (SwapDim (ax1, ax2)) ->
        match reqView with
        | Some rv ->
            let str = rv.Stride
            [Some {rv with Shape=srcShapes.[0]; 
                           Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1]}]
        | _ -> outplaceTrgt
    // misc
    | UnaryOp (Annotated _) -> inplaceOverwriteTrgt

    // binary elementwise
    | BinaryOp Add -> inplaceOverwriteTrgt
    | BinaryOp Substract -> inplaceOverwriteTrgt
    | BinaryOp Multiply -> inplaceOverwriteTrgt
    | BinaryOp Divide -> inplaceOverwriteTrgt
    | BinaryOp Power -> inplaceOverwriteTrgt
    // matrix/tensor operations
    | BinaryOp Dot -> outplaceTrgt
    | BinaryOp TensorProduct -> outplaceTrgt     


/// execution items for an elementwise operation
let execItemsForElemwise trgtView cOp srcViews =
    if srcViews |> List.exists (fun sv -> NDArrayView.nElems trgtView <> NDArrayView.nElems sv) then
        failwithf "sources have different number of elements than target"

    let args = 
        (cOp :> ICudaArgTmpl) ::
        ((NDArrayPtrArgTmpl trgtView) :> ICudaArgTmpl) ::
        (List.map (fun v -> (NDArrayPtrArgTmpl v) :> ICudaArgTmpl) srcViews)

    let nSrc = List.length srcViews
    let hetero = srcViews |> List.exists (fun sv -> trgtView.Shape <> sv.Shape)
    let indexedStr = if (cOp :> ICudaOp).IsIndexed then "Indexed" else ""
    let heteroStr = if hetero then "Heterogenous" else ""
    let kernel = 
        {FuncName=sprintf "elemwise%dAry%dD%s%s" nSrc (NDArrayView.nDim trgtView) indexedStr heteroStr;
         TmplArgs=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeNameWithoutPtr) args;
         RetType="void";
         ArgTypes=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) args;}

    let workDim = 
        match NDArrayView.nDim trgtView with
        | _ when hetero -> (NDArrayView.nElems trgtView, 1, 1)
        | 0 -> (1, 1, 1)
        | 1 -> (trgtView.Shape.[0], 1, 1)
        | 2 -> (trgtView.Shape.[0], trgtView.Shape.[1], 1)
        | 3 -> (trgtView.Shape.[0], trgtView.Shape.[1], trgtView.Shape.[2])
        | d ->
            let rest = {2 .. d-1} |> Seq.map (fun i -> trgtView.Shape.[i]) |> Seq.fold (*) 1 
            (trgtView.Shape.[0], trgtView.Shape.[1], rest)

    [LaunchKernel(kernel, workDim, args)]


/// returns the execution units for the specified op
let execItemsForOp cudaEnv trgtView op srcViews =
    match op with 
    // tensor creation
    | LeafOp (DiagonalOne _) -> execItemsForElemwise trgtView (NoArgEOpTmpl("DiagonalOneIEOp_t", true)) []
    | LeafOp (Zeros _) -> execItemsForElemwise trgtView (NoArgEOpTmpl("ZerosEOp_t", false)) []
    | LeafOp (ScalarConst f) -> execItemsForElemwise trgtView (ConstEOpTmpl(f)) []
    | LeafOp (TensorConst(f, _)) -> execItemsForElemwise trgtView (ConstEOpTmpl(f)) []
    // variable access
    | LeafOp (Var vs) -> 
        match cudaEnv.VarStorLoc |> Map.find vs with
        | DevVar -> []
        | HostVar -> 
            // we assume that host variable has continguous stride and zero offset
            let hv = {Memory=ExternalMem vs; Shape=trgtView.Shape; 
                      Offset=0; Stride=NDArray.contiguousStride trgtView.Shape}            
            [MemcpyHtoD(NDArrayHostMemRngTmpl(hv), NDArrayDevMemRngTmpl(trgtView))]       
    // unary elementwise
    | UnaryOp Negate -> execItemsForElemwise trgtView (NoArgEOpTmpl("NegateEOp_t", false)) srcViews
    | UnaryOp Log -> execItemsForElemwise trgtView (NoArgEOpTmpl("LogEOp_t", false)) srcViews
    | UnaryOp Exp -> execItemsForElemwise trgtView (NoArgEOpTmpl("ExpEOp_t", false)) srcViews
    // reductions
    | UnaryOp Sum -> execItemsForElemwise trgtView (NoArgEOpTmpl("Sum", false)) srcViews // TODO
    | UnaryOp (SumAxis _) -> execItemsForElemwise trgtView (NoArgEOpTmpl("SumAxis", false)) srcViews // TODO
    // shape operations
    | UnaryOp (Reshape _) ->
        if trgtView <> srcViews.[0] then execItemsForElemwise trgtView (NoArgEOpTmpl("IdEOp_t", false)) srcViews
        else []
    | UnaryOp (Broadcast _) -> []
    | UnaryOp (SwapDim _) -> []
    // misc
    | UnaryOp (Annotated _) -> []

    // binary elementwise
    | BinaryOp Add -> execItemsForElemwise trgtView (NoArgEOpTmpl("AddEOp_t", false)) srcViews
    | BinaryOp Substract -> execItemsForElemwise trgtView (NoArgEOpTmpl("SubstractEOp_t", false)) srcViews
    | BinaryOp Multiply -> execItemsForElemwise trgtView (NoArgEOpTmpl("MultiplyEOp_t", false)) srcViews
    | BinaryOp Divide -> execItemsForElemwise trgtView (NoArgEOpTmpl("DivideEOp_t", false)) srcViews
    | BinaryOp Power -> execItemsForElemwise trgtView (NoArgEOpTmpl("PowerEOp_t", false)) srcViews
    // matrix/tensor operations
    | BinaryOp Dot -> [] // TODO
    | BinaryOp TensorProduct -> [] // TODO


/// generates CUDA execution units that will evaluate the given unified expression
let exprToCudaExecUnits cudaEnv =
    exprToExecUnits {ExecItemsForOp=execItemsForOp cudaEnv; 
                     TrgtViewGivenSrc=trgtViewGivenSrc cudaEnv;
                     SrcViewReqsGivenTrgt=srcViewReqsGivenTrgt cudaEnv;}



