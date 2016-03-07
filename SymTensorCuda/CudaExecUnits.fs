namespace SymTensor.Compiler.Cuda

open System

open Basics
open Basics.Cuda
open ArrayNDNS
open SymTensor
open SymTensor.Compiler


[<AutoOpen>]
module CudaExecUnitTypes =

    /// a CUDA operation that will be assigned to and executed in a CUDA stream
    type CudaExecItemT =
        // memory operations
        | MemcpyDtoD of IDevMemRngTmpl * IDevMemRngTmpl
        | MemcpyHtoD of IHostMemRngTmpl * IDevMemRngTmpl
        | MemcpyDtoH of IDevMemRngTmpl * IHostMemRngTmpl
        | Memset of single * IDevMemRngTmpl
        // execution control
        | LaunchKernel of TmplInstT * WorkDimT * (ICudaArgTmpl list)
        | CallCFunc of TmplInstT * System.Type * (ICudaArgTmpl list)
        // CUBLAS calls 
        | BlasGemm of BlasTransposeOpT * BlasTransposeOpT *  
                      single * BlasTransposedMatrixTmpl * BlasTransposedMatrixTmpl * 
                      single * BlasTransposedMatrixTmpl


module CudaExecUnit =

    /// Computes desired source views given desired target view.
    /// There is no guarantee that the desired source views will be used.
    let srcReqsGivenTrgt cudaEnv trgtShape reqView op srcShapes =
        let nSrcs = List.length srcShapes

        // requests all sources to use separate storage
        let noSrcReqs =
            List.replicate nSrcs None

        // requests the first source to be evaluated into our target view
        let inplaceFirstSrcReq =
            match nSrcs with
            | 0 -> []
            | 1 -> [reqView]
            | _ -> reqView :: List.replicate (nSrcs-1) None

        match op with
        | ULeafOp _ -> []

        // unary elementwise
        | UUnaryOp Negate -> inplaceFirstSrcReq                        
        | UUnaryOp Abs -> inplaceFirstSrcReq
        | UUnaryOp SignT -> inplaceFirstSrcReq
        | UUnaryOp Log -> inplaceFirstSrcReq
        | UUnaryOp Log10 -> inplaceFirstSrcReq                           
        | UUnaryOp Exp -> inplaceFirstSrcReq                           
        | UUnaryOp Sin -> inplaceFirstSrcReq
        | UUnaryOp Cos -> inplaceFirstSrcReq
        | UUnaryOp Tan -> inplaceFirstSrcReq
        | UUnaryOp Asin -> inplaceFirstSrcReq
        | UUnaryOp Acos -> inplaceFirstSrcReq
        | UUnaryOp Atan -> inplaceFirstSrcReq
        | UUnaryOp Sinh -> inplaceFirstSrcReq
        | UUnaryOp Cosh -> inplaceFirstSrcReq
        | UUnaryOp Tanh -> inplaceFirstSrcReq
        | UUnaryOp Sqrt -> inplaceFirstSrcReq
        | UUnaryOp Ceil -> inplaceFirstSrcReq
        | UUnaryOp Floor -> inplaceFirstSrcReq
        | UUnaryOp Round -> inplaceFirstSrcReq
        | UUnaryOp Truncate -> inplaceFirstSrcReq        
        // reductions
        | UUnaryOp Sum -> noSrcReqs
        | UUnaryOp (SumAxis _) -> noSrcReqs
        // shape operations
        | UUnaryOp (Reshape _) ->        
            match reqView with
            | Some rv when ArrayND.isContiguous rv ->
                [Some (ArrayND.reshapeView srcShapes.[0] rv)]
            | _ -> noSrcReqs
        | UUnaryOp (DoBroadcast _) -> noSrcReqs
        | UUnaryOp (SwapDim (ax1, ax2)) ->
            match reqView with
            | Some rv -> [Some (ArrayND.swapDim ax1 ax2 rv)]
            | _ -> noSrcReqs
        // variable access
        | UUnaryOp (StoreToVar vs) ->
            match cudaEnv.VarStorLoc |> Map.find vs with
            | LocDev -> 
                // request to store directly into external var
                // we assume that all device input vars are continguous
                [Some (ArrayNDManikin.externalContiguous (MemExternal vs) trgtShape)]
            | LocHost -> noSrcReqs
            | loc -> unsupLoc loc
        // misc
        | UUnaryOp (Annotated _) -> inplaceFirstSrcReq

        // binary elementwise
        | UBinaryOp Add -> inplaceFirstSrcReq
        | UBinaryOp Substract -> inplaceFirstSrcReq
        | UBinaryOp Multiply -> inplaceFirstSrcReq
        | UBinaryOp Divide -> inplaceFirstSrcReq
        | UBinaryOp Modulo -> inplaceFirstSrcReq
        | UBinaryOp Power -> inplaceFirstSrcReq
        // matrix/tensor operations
        | UBinaryOp Dot -> noSrcReqs
        | UBinaryOp TensorProduct -> noSrcReqs     

        // nary
        | UNaryOp Discard -> noSrcReqs
        | UNaryOp (Subtensor _) -> noSrcReqs
        | UNaryOp (SetSubtensor _) -> 
            // "a" can be evaluated into requested manikin, but "b" (the replacement value) must be placed
            // in a temporary manikin and copied over to avoid race conditions.
            inplaceFirstSrcReq
        | UNaryOp (ExtensionOp eop) -> failwith "not implemented yet"


    /// computes the definitive target view of an op given its source views
    let trgtGivenSrc compileEnv memAllocator (typ: TypeNameT) (trgtShape: NShapeSpecT) (req: ArrayNDManikinT option) 
                     (op: UOpT) (srcs: ArrayNDManikinT list) (srcShared: bool list)  =

        // new allocated target
        let newTrgt =
            ArrayNDManikin.newContiguous memAllocator typ trgtShape, false        

        // target that shares no elements with any srcView
        let outplaceTrgt =
            match req with
            | Some rv when not (List.exists (ArrayND.overlapping rv) srcs) -> rv, false
            | _ -> newTrgt 

        let outplaceBlasTrgt =
            match req with
            | Some rv when not (List.exists (ArrayND.overlapping rv) srcs) &&
                           ArrayND.isBlasTargetable rv -> rv, false
            | _ -> ArrayNDManikin.newColumnMajor memAllocator typ trgtShape, false

        // target that reuses a srcView, if it may be overwritten
        let inplaceOvrwrtTrgt =
            match List.tryFindIndex not srcShared with
            | Some i -> srcs.[i], false
            | None -> outplaceTrgt    

        match op with
        // variable access
        | ULeafOp (Var vs) ->       
            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev ->
                // we assume that all device input vars are continguous
                ArrayNDManikin.externalContiguous (MemExternal vs) trgtShape, true
            | LocHost ->
                // will transfer variable from host to device during execution
                // need continguous memory for that
                match req with
                | Some rv when ArrayND.isContiguous rv -> rv, false
                | _ -> ArrayNDManikin.newContiguous memAllocator typ trgtShape, false    
            | loc -> unsupLoc loc                    
        // tensor creation
        | ULeafOp _ -> outplaceTrgt        

        // unary elementwise
        | UUnaryOp Negate -> inplaceOvrwrtTrgt                        
        | UUnaryOp Abs -> inplaceOvrwrtTrgt
        | UUnaryOp SignT -> inplaceOvrwrtTrgt
        | UUnaryOp Log -> inplaceOvrwrtTrgt
        | UUnaryOp Log10 -> inplaceOvrwrtTrgt                           
        | UUnaryOp Exp -> inplaceOvrwrtTrgt                           
        | UUnaryOp Sin -> inplaceOvrwrtTrgt
        | UUnaryOp Cos -> inplaceOvrwrtTrgt
        | UUnaryOp Tan -> inplaceOvrwrtTrgt
        | UUnaryOp Asin -> inplaceOvrwrtTrgt
        | UUnaryOp Acos -> inplaceOvrwrtTrgt
        | UUnaryOp Atan -> inplaceOvrwrtTrgt
        | UUnaryOp Sinh -> inplaceOvrwrtTrgt
        | UUnaryOp Cosh -> inplaceOvrwrtTrgt
        | UUnaryOp Tanh -> inplaceOvrwrtTrgt
        | UUnaryOp Sqrt -> inplaceOvrwrtTrgt
        | UUnaryOp Ceil -> inplaceOvrwrtTrgt
        | UUnaryOp Floor -> inplaceOvrwrtTrgt
        | UUnaryOp Round -> inplaceOvrwrtTrgt
        | UUnaryOp Truncate -> inplaceOvrwrtTrgt    
        // reductions
        | UUnaryOp Sum -> outplaceTrgt
        | UUnaryOp (SumAxis _) -> outplaceTrgt
        // shape operations
        | UUnaryOp (Reshape _) ->        
            // TODO: optimize: check if copy is really necessary
            if ArrayND.isContiguous srcs.[0] then
                ArrayND.reshapeView trgtShape srcs.[0], srcShared.[0]
            else outplaceTrgt  // will copy
        | UUnaryOp (DoBroadcast _) ->
            ArrayND.broadcastToShape trgtShape srcs.[0], srcShared.[0]
        | UUnaryOp (SwapDim (ax1, ax2)) ->
            ArrayND.swapDim ax1 ax2 srcs.[0], srcShared.[0]
        // variable access
        | UUnaryOp (StoreToVar _) -> newTrgt
        // misc
        | UUnaryOp (Annotated _) -> srcs.[0], srcShared.[0]

        // binary elementwise
        | UBinaryOp Add -> inplaceOvrwrtTrgt
        | UBinaryOp Substract -> inplaceOvrwrtTrgt
        | UBinaryOp Multiply -> inplaceOvrwrtTrgt
        | UBinaryOp Divide -> inplaceOvrwrtTrgt
        | UBinaryOp Modulo -> inplaceOvrwrtTrgt
        | UBinaryOp Power -> inplaceOvrwrtTrgt
        // matrix/tensor operations
        | UBinaryOp Dot -> outplaceBlasTrgt
        | UBinaryOp TensorProduct -> outplaceTrgt

        // nary
        | UNaryOp Discard -> outplaceTrgt
        | UNaryOp (Subtensor srs) -> 
            if SimpleRangesSpec.isDynamic srs then 
                // dynamic subtensors will be copied out of the src
                outplaceTrgt
            else
                // symbolic subtensors use a view of the src 
                let rng = SimpleRangesSpec.eval (fun _ -> failwith "is static") srs
                srcs.[0].[rng] :?> ArrayNDManikinT, srcShared.[0]
        | UNaryOp (SetSubtensor _) ->
            if not (srcShared.[0]) then srcs.[0], false
            else outplaceTrgt
        | UNaryOp (ExtensionOp eop) -> failwith "not implemented yet"
   
    /// execution item to lunch the given kernel template function
    let execItemsForKernel cppFuncName tmplTmpls argTmpls workDim = 
        let cFuncTmpl =
            {FuncName=cppFuncName;
             Domain=KernelFunc;
             TmplArgs=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) tmplTmpls;
             RetType="void";
             ArgTypes=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) argTmpls;}    
        [LaunchKernel(cFuncTmpl, workDim, argTmpls)]

    /// returns the CUDA work dimensions for an elementwise operation
    let workDimForElemwise trgt hetero =
        match ArrayND.nDims trgt with
        | _ when hetero -> (ArrayND.nElems trgt, 1, 1)
        | 0 -> (1, 1, 1)
        | 1 -> ((ArrayND.shape trgt).[0], 1, 1)
        | 2 -> ((ArrayND.shape trgt).[0], (ArrayND.shape trgt).[1], 1)
        | 3 -> ((ArrayND.shape trgt).[0], (ArrayND.shape trgt).[1], (ArrayND.shape trgt).[2])
        | d ->
            let rest = {2 .. d-1} |> Seq.map (fun i -> (ArrayND.shape trgt).[i]) |> Seq.fold (*) 1 
            ((ArrayND.shape trgt).[0], (ArrayND.shape trgt).[1], rest)

    /// execution items for an elementwise operation
    let execItemsForElemwise trgt cOp srcViews =
        if srcViews |> List.exists (fun sv -> ArrayND.nElems trgt <> ArrayND.nElems sv) then
            failwithf "sources have different number of elements than target"

        let args = 
            (cOp :> ICudaArgTmpl) ::
            ((ArrayNDArgTmpl trgt) :> ICudaArgTmpl) ::
            (List.map (fun v -> (ArrayNDArgTmpl v) :> ICudaArgTmpl) srcViews)

        let nSrc = List.length srcViews
        let hetero = srcViews |> List.exists (fun sv -> (ArrayND.shape trgt) <> (ArrayND.shape sv))
        let indexedStr = if (cOp :> ICudaOp).IsIndexed then "Indexed" else ""
        let heteroStr = if hetero then "Heterogenous" else ""
        let funcName = sprintf "elemwise%dAry%dD%s%s" nSrc (ArrayND.nDims trgt) indexedStr heteroStr

        execItemsForKernel funcName args args (workDimForElemwise trgt hetero)


    let dynamicSubtensorTmplAndIdx (bas: ArrayNDManikinT) (rngs: UExprRngsSpecT) (rngManikins: ArrayNDManikinT list) =
        // Apply symbolic ranges to src, and leave dynamic axes unharmed.
        // (0 is added to offset and their size is changed appropriately)
        let basStatic = bas.[SimpleRangesSpec.eval (fun _ -> 0) rngs] :?> ArrayNDManikinT

        // convert simplified range specification to array of pointers to expressions calculating
        // the indices
        let rec rngToIdxPntrs rngs rngManikins =
            match rngs, rngManikins with
            | SRSDynStartSymSize _ :: rrngs, rngManikin :: rrngManikins ->
                // for dynamic range pass pointer to result of expression calculating the index
                (SizeTPtrFromArrayNDIdxTmpl (Some rngManikin) :> ICudaArrayMemberArgTmpl<IntPtr>) :: 
                    rngToIdxPntrs rrngs rrngManikins 
            | SRSSymStartSymEnd _ :: rrngs, _ ->
                // symbolic range has already been applied, pass null (meaning no offset to add)
                (SizeTPtrFromArrayNDIdxTmpl None :> ICudaArrayMemberArgTmpl<IntPtr>) :: 
                    rngToIdxPntrs rrngs rngManikins 
            | [], [] -> []
            | _ -> failwith "invalid dynamic range specification"
        let basIdxPntrs = rngToIdxPntrs rngs rngManikins

        // C++ parameters
        ArrayNDArgTmpl basStatic, ArrayNDSDArgTmpl basStatic, CPPArrayTmpl basIdxPntrs

    let execItemsForCopyFromDynamicSubtensor trgt src rngs rngManikins =
        // C++ signature is:
        //template <typename TTarget, typename TBaseSrc, typename TDynSrc, size_t nDims,
        //          TElemwise1Ary<IdEOp_t, TTarget, TDynSrc>::type copyFun>
        //_dev void copyFromDynamicSubtensor(TTarget &trgt,  
        //                                   const TBaseSrc &baseSrc, const Array<size_t, nDims> &srcIdx)

        let srcTmpl, srcDynTmpl, srcIdxPntrsTmpl = dynamicSubtensorTmplAndIdx src rngs rngManikins
        let nDimsStr = sprintf "%d" (ArrayND.nDims trgt)
        let elemwiseFunc = sprintf "elemwise1Ary%dD" (ArrayND.nDims trgt)

        execItemsForKernel 
            "copyFromDynamicSubtensor" 
            [ArrayNDArgTmpl trgt; srcTmpl; srcDynTmpl; CPPTemplateValue nDimsStr; CPPTemplateValue elemwiseFunc]
            [ArrayNDArgTmpl trgt; srcTmpl; srcIdxPntrsTmpl]
            (workDimForElemwise trgt false)

    let execItemsForCopyToDynamicSubtensor trgt rngs rngManikins src =
        // C++ signature is:
        //template <typename TBaseTrgt, typename TDynTrgt, size_t nDims, typename TSrc,
        //          TElemwise1Ary<IdEOp_t, TDynTrgt, TSrc>::type copyFun>
        //_dev void copyToDynamicSubtensor(TBaseTrgt &baseTrgt, const Array<size_t, nDims> &trgtIdx,
        //                                 const TSrc &src)

        let trgtTmpl, trgtDynTmpl, trgtIdxPntrsTmpl = dynamicSubtensorTmplAndIdx trgt rngs rngManikins
        let nDimsStr = sprintf "%d" (ArrayND.nDims src)
        let elemwiseFunc = sprintf "elemwise1Ary%dD" (ArrayND.nDims src)

        execItemsForKernel 
            "copyToDynamicSubtensor" 
            [trgtTmpl; trgtDynTmpl; CPPTemplateValue nDimsStr; ArrayNDArgTmpl src; CPPTemplateValue elemwiseFunc]
            [trgtTmpl; trgtIdxPntrsTmpl; ArrayNDArgTmpl src]
            (workDimForElemwise src false)


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
    let copyExecItems trgt src =
        if ArrayND.nElems trgt <> ArrayND.nElems src then
            failwithf "cannot copy array with %d elements to array with %d elements"
                (ArrayND.nElems trgt) (ArrayND.nElems src)
        execItemsForElemwise trgt (NoArgEOpArgTmpl("IdEOp_t", false)) [src]

    /// BLAS input argument passing, so that orientation is preserved
    let blasArg memAllocator (manikin: ArrayNDManikinT) =
        match (ArrayND.stride manikin) with
        | [_; 1] -> manikin, BlasTranspose, []
        | [1; _] -> ArrayND.transpose manikin, BlasId, []
        | [_; _] -> 
            // need to copy
            let tmpView = ArrayNDManikin.newContiguous memAllocator 
                                                       (ArrayNDManikin.typeName manikin) (ArrayND.shape manikin)
            let copyOps = copyExecItems tmpView manikin
            tmpView, BlasTranspose, copyOps
        | _ -> failwith "need 2-dimensional array for BLAS argument"

    /// BLAS target argument passing, so that orientation is preserved
    let blasTarget (manikin: ArrayNDManikinT) =
        match (ArrayND.stride manikin) with
        | [1; _] -> ArrayND.transpose manikin
        | _ -> failwith "cannot use specified view as BLAS target"

    /// returns the execution units for the specified op
    let execItemsForOp compileEnv memAllocator trgt op srcs =
        match op with 
        // tensor creation
        | ULeafOp (Identity _) -> execItemsForElemwise trgt (NoArgEOpArgTmpl("DiagonalOneIEOp_t", true)) []
        | ULeafOp (Zeros _) -> execItemsForElemwise trgt (NoArgEOpArgTmpl("ZerosEOp_t", false)) []
        | ULeafOp (ScalarConst f) -> execItemsForElemwise trgt (ConstEOpArgTmpl f) []
        // variable access
        | ULeafOp (Var vs) -> 
            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev -> []
            | LocHost -> 
                // we assume that host variable has continguous stride and zero offset
                let hv = ArrayNDManikin.externalContiguous (MemExternal vs) (ArrayND.shape trgt)
                [MemcpyHtoD(ArrayNDHostRegMemRngTmpl(hv), ArrayNDDevMemRngTmpl(trgt))]       
            | loc -> unsupLoc loc
        // unary elementwise
        | UUnaryOp Negate -> execItemsForElemwise trgt (NoArgEOpArgTmpl("NegateEOp_t", false)) srcs
        | UUnaryOp Abs -> execItemsForElemwise trgt (NoArgEOpArgTmpl("AbsEOp_t", false)) srcs
        | UUnaryOp SignT -> execItemsForElemwise trgt (NoArgEOpArgTmpl("SignTEOp_t", false)) srcs
        | UUnaryOp Log -> execItemsForElemwise trgt (NoArgEOpArgTmpl("LogEOp_t", false)) srcs
        | UUnaryOp Log10 -> execItemsForElemwise trgt (NoArgEOpArgTmpl("Log10EOp_t", false)) srcs
        | UUnaryOp Exp -> execItemsForElemwise trgt (NoArgEOpArgTmpl("ExpEOp_t", false)) srcs
        | UUnaryOp Sin -> execItemsForElemwise trgt (NoArgEOpArgTmpl("SinEOp_t", false)) srcs
        | UUnaryOp Cos -> execItemsForElemwise trgt (NoArgEOpArgTmpl("CosEOp_t", false)) srcs
        | UUnaryOp Tan -> execItemsForElemwise trgt (NoArgEOpArgTmpl("TanEOp_t", false)) srcs
        | UUnaryOp Asin -> execItemsForElemwise trgt (NoArgEOpArgTmpl("AsinEOp_t", false)) srcs
        | UUnaryOp Acos -> execItemsForElemwise trgt (NoArgEOpArgTmpl("AcosEOp_t", false)) srcs
        | UUnaryOp Atan -> execItemsForElemwise trgt (NoArgEOpArgTmpl("AtanEOp_t", false)) srcs
        | UUnaryOp Sinh -> execItemsForElemwise trgt (NoArgEOpArgTmpl("SinhEOp_t", false)) srcs
        | UUnaryOp Cosh -> execItemsForElemwise trgt (NoArgEOpArgTmpl("CoshEOp_t", false)) srcs
        | UUnaryOp Tanh -> execItemsForElemwise trgt (NoArgEOpArgTmpl("TanhEOp_t", false)) srcs
        | UUnaryOp Sqrt -> execItemsForElemwise trgt (NoArgEOpArgTmpl("SqrtEOp_t", false)) srcs
        | UUnaryOp Ceil -> execItemsForElemwise trgt (NoArgEOpArgTmpl("CeilEOp_t", false)) srcs
        | UUnaryOp Floor -> execItemsForElemwise trgt (NoArgEOpArgTmpl("FloorEOp_t", false)) srcs
        | UUnaryOp Round -> execItemsForElemwise trgt (NoArgEOpArgTmpl("RoundEOp_t", false)) srcs
        | UUnaryOp Truncate -> execItemsForElemwise trgt (NoArgEOpArgTmpl("TruncateEOp_t", false)) srcs
        // reductions
        | UUnaryOp Sum -> 
            execItemsForCFunc<CPPSum> [ArrayNDArgTmpl trgt; ArrayNDArgTmpl srcs.[0]]
        | UUnaryOp (SumAxis ax) -> 
            // we need to swap axes so that the axes the summation is performed over comes last
            let src = srcs.[0]
            let nd = ArrayND.nDims src
            let axOrder = Seq.concat [ {0 .. ax-1}; {ax + 1 .. nd - 1}; Seq.singleton ax] |> Seq.toList
            let srcAdj = ArrayND.reorderAxes axOrder src
            execItemsForCFunc<CPPSumLastAxis> [ArrayNDArgTmpl trgt; ArrayNDArgTmpl srcAdj]
        // shape operations
        | UUnaryOp (Reshape _) ->
            if trgt <> srcs.[0] then copyExecItems trgt srcs.[0]
            else []
        | UUnaryOp (DoBroadcast _) -> []
        | UUnaryOp (SwapDim _) -> []
        // variable access
        | UUnaryOp (StoreToVar vs) ->
            let varShp, varType = ArrayND.shape srcs.[0], srcs.[0].TypeName

            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev when trgt.Storage <> (MemExternal vs) -> 
                // Our source has not been evaluated directly into the variable storage.
                // Therefore we need to copy into the variable.
                // We assume that all device vars are continguous.
                let dv = ArrayNDManikin.externalContiguous (MemExternal vs) varShp
                copyExecItems dv srcs.[0]
            | LocDev ->
                // Source was evaluated directly into the variable storage.
                // No copy necessary.
                []
            | LocHost ->            
                let copyItems, memcpySrc = 
                    if ArrayND.isContiguous srcs.[0] then 
                        // Source is contiguous. Can directly copy to host.
                        [], srcs.[0]
                    else
                        // Need to copy to temporary contiguous storage first.
                        let tmp = ArrayNDManikin.newContiguous memAllocator varType varShp
                        copyExecItems tmp srcs.[0], tmp

                // We assume that all host vars are continguous.
                // trgtView has contingous stride
                let hv = ArrayNDManikin.externalContiguous (MemExternal vs) varShp
                copyItems @ [MemcpyDtoH(ArrayNDDevMemRngTmpl(memcpySrc), ArrayNDHostRegMemRngTmpl(hv))]   
            | loc -> unsupLoc loc                              
        // misc
        | UUnaryOp (Annotated _) -> []

        // binary elementwise
        | UBinaryOp Add ->       execItemsForElemwise trgt (NoArgEOpArgTmpl("AddEOp_t",       false)) srcs
        | UBinaryOp Substract -> execItemsForElemwise trgt (NoArgEOpArgTmpl("SubstractEOp_t", false)) srcs
        | UBinaryOp Multiply ->  execItemsForElemwise trgt (NoArgEOpArgTmpl("MultiplyEOp_t",  false)) srcs
        | UBinaryOp Divide ->    execItemsForElemwise trgt (NoArgEOpArgTmpl("DivideEOp_t",    false)) srcs
        | UBinaryOp Modulo ->    execItemsForElemwise trgt (NoArgEOpArgTmpl("ModuloEOp_t",    false)) srcs
        | UBinaryOp Power ->     execItemsForElemwise trgt (NoArgEOpArgTmpl("PowerEOp_t",     false)) srcs
        // matrix/tensor operations
        | UBinaryOp Dot -> 
            let aView, aOp, aCopyItems = blasArg memAllocator srcs.[0]
            let bView, bOp, bCopyItems = blasArg memAllocator srcs.[1]
            let tView = blasTarget trgt
            let blasItems = [BlasGemm(aOp, bOp, 1.0f, 
                                      BlasTransposedMatrixTmpl(aView), 
                                      BlasTransposedMatrixTmpl(bView),
                                      0.0f, BlasTransposedMatrixTmpl(tView))]
            aCopyItems @ bCopyItems @ blasItems
        | UBinaryOp TensorProduct -> [] // TODO

        // nary
        | UNaryOp Discard -> []
        | UNaryOp (Subtensor srs) ->
            if SimpleRangesSpec.isDynamic srs then 
                // copy dynamic subtensor out of the src
                execItemsForCopyFromDynamicSubtensor trgt srcs.[0] srs (List.tail srcs)
            else [] // symbolic subtensor uses a slice of the src view
        | UNaryOp (SetSubtensor srs) ->
            // copy "a" if necessary
            let copyItems = 
                if trgt <> srcs.[0] then copyExecItems trgt srcs.[0] else []
            // copy "b" into a
            let setItems =
                execItemsForCopyToDynamicSubtensor trgt srs (List.skip 2 srcs) srcs.[1]
            copyItems @ setItems
        | UNaryOp (ExtensionOp eop) -> failwith "not implemented yet"



    /// generates CUDA execution units that will evaluate the given unified expression
    let exprToCudaExecUnits (compileEnv: CudaCompileEnvT) =
        ExecUnit.exprToExecUnits {ExecItemsForOp=execItemsForOp compileEnv; 
                                  TrgtGivenSrc=trgtGivenSrc compileEnv;
                                  SrcReqsGivenTrgt=srcReqsGivenTrgt compileEnv;}




