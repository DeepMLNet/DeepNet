namespace SymTensor.Compiler.Cuda

open System
open System.Runtime.InteropServices
open ManagedCuda

open Util
open ArrayNDNS
open SymTensor
open SymTensor.Compiler



    



/// a CUDA operation 
type CudaOpT =
    // memory operations
    | MemcpyDtoD of IDevMemRngTmpl * IDevMemRngTmpl
    | MemcpyHtoD of IHostMemRngTmpl * IDevMemRngTmpl
    | MemcpyDtoH of IDevMemRngTmpl * IHostMemRngTmpl
    | Memset of single * IDevMemRngTmpl
    // execution control
    | LaunchKernel of TmplInstT * WorkDimT * (ICudaArgTmpl list)
    | CallCFunc of TmplInstT * System.Type * (ICudaArgTmpl list)
    // CUBLAS calls 
    | BlasGemm of BlasOpT * BlasOpT *  
                  single * BlasTransposedMatrixTmpl * BlasTransposedMatrixTmpl * 
                  single * BlasTransposedMatrixTmpl


/// CUDA C++ operation functor description
type ICudaOp =
    abstract member IsIndexed : bool  

/// NDArray pointer marshalling template
type ArrayNDArgTmpl (manikin: ArrayNDManikinT) = 
    interface ICudaArgTmpl with
        member this.CPPTypeName = manikin.CPPType
        member this.GetArg env =
            // C++ struct just contains the pointer to data memory
            (CudaExecEnv.getDevVar env manikin).DevicePointer :> obj

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

    let outplaceBlasTrgt =
        match reqView with
        | Some rv when not (List.exists (NDArrayView.overlapping rv) srcViews) &&
                       NDArrayView.isBlasTargetable rv -> rv, false
        | _ -> NDArrayView.newBlasTarget memAllocator trgtShape, false

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
             Stride=ArrayND.contiguousStride trgtShape}, true
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
    // variable access
    | UnaryOp (StoreToVar vs) ->
        match cudaEnv.VarStorLoc |> Map.find vs with
        | DevVar -> 
            // we assume that all device input vars are continguous
            {Memory=ExternalMem vs; 
             Shape=trgtShape; Offset=0; 
             Stride=ArrayND.contiguousStride trgtShape}, true
        | HostVar ->
            // need continguous memory to transfer to host
            if NDArrayView.isContiguous srcViews.[0] then srcViews.[0], srcShared.[0]
            else outplaceTrgt 
    // misc
    | UnaryOp (Annotated _) -> srcViews.[0], srcShared.[0]

    // binary elementwise
    | BinaryOp Add -> inplaceOverwriteTrgt
    | BinaryOp Substract -> inplaceOverwriteTrgt
    | BinaryOp Multiply -> inplaceOverwriteTrgt
    | BinaryOp Divide -> inplaceOverwriteTrgt
    | BinaryOp Power -> inplaceOverwriteTrgt
    // matrix/tensor operations
    | BinaryOp Dot -> outplaceBlasTrgt
    | BinaryOp TensorProduct -> outplaceTrgt

    // nary
    | NaryOp Discard -> outplaceTrgt
      

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
    // variable access
    | UnaryOp (StoreToVar vs) ->
        match cudaEnv.VarStorLoc |> Map.find vs with
        | DevVar -> 
            // request to store directly into external var
            // we assume that all device input vars are continguous
            [Some {Memory=ExternalMem vs; 
                   Shape=trgtShape; Offset=0; 
                   Stride=ArrayND.contiguousStride trgtShape}]
        | HostVar ->
            // need continguous storage to transfer to host
            match reqView with
            | Some rv when NDArrayView.isContiguous rv -> [reqView]
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

    // nary
    | NaryOp Discard -> outplaceTrgt


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
         Domain=KernelFunc;
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


/// specifies the name of the C++ function
type CPPFuncNameAttribute (cppFuncName: string) =
    inherit System.Attribute()     
    member this.CPPFuncName = cppFuncName

[<CPPFuncName("sum")>]
type CPPSum = delegate of BasicTypes.CUdeviceptr * BasicTypes.CUdeviceptr -> unit
//type CPPSum = delegate of NDArrayDev * NDArrayDev -> unit

[<CPPFuncName("sumLastAxis")>]
type CPPSumLastAxis = delegate of BasicTypes.CUdeviceptr * BasicTypes.CUdeviceptr -> unit
//type CPPSumLastAxis = delegate of NDArrayDev * NDArrayDev -> unit

/// generate ExecItems to call a C++ template function
let execItemsForCFunc<'FuncDelegate when 'FuncDelegate :> System.Delegate> argTmpls =
    let cDelegateType = typeof<'FuncDelegate>
    let cAttributes = cDelegateType.GetCustomAttributes(typeof<CPPFuncNameAttribute>, false)
    if Array.isEmpty cAttributes then
        failwithf "CPPFuncName attribute is missing on delegate %A" cDelegateType
    let cppFuncNameAttribute = cAttributes.[0] :?> CPPFuncNameAttribute
    let cppFuncName = cppFuncNameAttribute.CPPFuncName

    let cFuncTmpl =
        {FuncName=cppFuncName;
         Domain=CPPFunc;
         TmplArgs=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) argTmpls;
         RetType="void";
         ArgTypes=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) argTmpls;}    
    [CallCFunc(cFuncTmpl, cDelegateType, argTmpls)]


/// generates ExecItems to copy srcView to trgtView 
let copyExecItems trgtView srcView =
    if NDArrayView.nElems trgtView <> NDArrayView.nElems srcView then
        failwithf "cannot copy array with %d elements to array with %d elements"
            (NDArrayView.nElems trgtView) (NDArrayView.nElems srcView)
    execItemsForElemwise trgtView (NoArgEOpTmpl("IdEOp_t", false)) [srcView]

/// BLAS arg passing, so that orientation is preserved
let blasArg memAllocator (view: NDArrayViewT) =
    match view.Stride with
    | [_; 1] -> view, BlasTranspose, []
    | [1; _] -> NDArrayView.transpose view, BlasId, []
    | [_; _] -> 
        // need to copy
        let tmpView = NDArrayView.newContinguous memAllocator view.Shape
        let copyOps = copyExecItems tmpView view
        tmpView, BlasTranspose, copyOps
    | _ -> failwith "need 2-dimensional array for BLAS argument"

/// BLAS result processing, so that orientation is preserved
let blasTarget (view: NDArrayViewT) =
    match view.Stride with
    | [1; _] -> NDArrayView.transpose view
    | _ -> failwith "cannot use specified view as BLAS target"

/// returns the execution units for the specified op
let execItemsForOp cudaEnv memAllocator trgtView op srcViews =
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
                      Offset=0; Stride=ArrayND.contiguousStride trgtView.Shape}            
            [MemcpyHtoD(NDArrayHostMemRngTmpl(hv), NDArrayDevMemRngTmpl(trgtView))]       
    // unary elementwise
    | UnaryOp Negate -> execItemsForElemwise trgtView (NoArgEOpTmpl("NegateEOp_t", false)) srcViews
    | UnaryOp Log -> execItemsForElemwise trgtView (NoArgEOpTmpl("LogEOp_t", false)) srcViews
    | UnaryOp Exp -> execItemsForElemwise trgtView (NoArgEOpTmpl("ExpEOp_t", false)) srcViews
    // reductions
    | UnaryOp Sum -> 
        execItemsForCFunc<CPPSum> [NDArrayPtrArgTmpl trgtView; NDArrayPtrArgTmpl srcViews.[0]]
    | UnaryOp (SumAxis ax) -> 
        // we need to swap axes so that the axes the summation is performed over comes last
        let src = srcViews.[0]
        let nd = NDArrayView.nDim src
        let axOrder = Seq.concat [ {0 .. ax-1}; {ax + 1 .. nd - 1}; Seq.singleton ax] |> Seq.toList
        let srcAdj = NDArrayView.reorderAxes axOrder src
        execItemsForCFunc<CPPSumLastAxis> [NDArrayPtrArgTmpl trgtView; NDArrayPtrArgTmpl srcAdj]
    // shape operations
    | UnaryOp (Reshape _) ->
        if trgtView <> srcViews.[0] then copyExecItems trgtView srcViews.[0]
        else []
    | UnaryOp (Broadcast _) -> []
    | UnaryOp (SwapDim _) -> []
    // variable access
    | UnaryOp (StoreToVar vs) ->
        let copyItems = 
            if trgtView <> srcViews.[0] then copyExecItems trgtView srcViews.[0]
            else []
        match cudaEnv.VarStorLoc |> Map.find vs with
        | DevVar -> 
            // trgtView is the variable we need to store into
            copyItems
        | HostVar ->            
            // we assume that host variable has continguous stride and zero offset
            // trgtView has contingous stride
            let hv = {Memory=ExternalMem vs; Shape=trgtView.Shape; 
                      Offset=0; Stride=ArrayND.contiguousStride trgtView.Shape}            
            copyItems @ [MemcpyDtoH(NDArrayDevMemRngTmpl(trgtView), NDArrayHostMemRngTmpl(hv))]                 
    // misc
    | UnaryOp (Annotated _) -> []

    // binary elementwise
    | BinaryOp Add -> execItemsForElemwise trgtView (NoArgEOpTmpl("AddEOp_t", false)) srcViews
    | BinaryOp Substract -> execItemsForElemwise trgtView (NoArgEOpTmpl("SubstractEOp_t", false)) srcViews
    | BinaryOp Multiply -> execItemsForElemwise trgtView (NoArgEOpTmpl("MultiplyEOp_t", false)) srcViews
    | BinaryOp Divide -> execItemsForElemwise trgtView (NoArgEOpTmpl("DivideEOp_t", false)) srcViews
    | BinaryOp Power -> execItemsForElemwise trgtView (NoArgEOpTmpl("PowerEOp_t", false)) srcViews
    // matrix/tensor operations
    | BinaryOp Dot -> 
        let aView, aOp, aCopyItems = blasArg memAllocator srcViews.[0]
        let bView, bOp, bCopyItems = blasArg memAllocator srcViews.[1]
        let tView = blasTarget trgtView
        let blasItems = [BlasGemm(aOp, bOp, 1.0f, 
                                  BlasTransposedMatrixTmpl(aView), 
                                  BlasTransposedMatrixTmpl(bView),
                                  0.0f, BlasTransposedMatrixTmpl(tView))]
        aCopyItems @ bCopyItems @ blasItems
    | BinaryOp TensorProduct -> [] // TODO

    // nary
    | NaryOp Discard -> []


/// generates CUDA execution units that will evaluate the given unified expression
let exprToCudaExecUnits cudaEnv =
    exprToExecUnits {ExecItemsForOp=execItemsForOp cudaEnv; 
                     TrgtViewGivenSrc=trgtViewGivenSrc cudaEnv;
                     SrcViewReqsGivenTrgt=srcViewReqsGivenTrgt cudaEnv;}



