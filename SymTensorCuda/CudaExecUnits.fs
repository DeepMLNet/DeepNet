namespace SymTensor.Compiler.Cuda

open System

open Basics
open Basics.Cuda
open ArrayNDNS
open SymTensor
open SymTensor.Compiler
open UExprTypes


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
        | BlasGemmBatched of BlasTransposeOpT * BlasTransposeOpT *  
                             single * BlasTransposedMatrixBatchTmpl * BlasTransposedMatrixBatchTmpl * 
                             single * BlasTransposedMatrixBatchTmpl
        // LAPACK calls
        | BlasGetrfBatched of BlasTransposedMatrixBatchTmpl * 
                              BlasIntArrayTmpl * BlasIntArrayTmpl
        | BlasGetriBatched of BlasTransposedMatrixBatchTmpl * BlasIntArrayTmpl *
                              BlasTransposedMatrixBatchTmpl * BlasIntArrayTmpl
                              
        // pointer array creation for CUBLAS batch calls
        | BlasInitPointerArray of BlasTransposedMatrixBatchTmpl
        // misc
        | Trace of UExprT * ArrayNDManikinT


module CudaExecUnit =
    open ManagedCuda.BasicTypes

    /// The operation the blasArg will perform.
    type BlasArgOperation =
        /// no operation
        | BlasArgId
        /// in-place transposition
        | BlasArgTranspose
        /// copy into temporary array of row-major layot
        /// (no transposition occurs)
        | BlasArgCopy

    /// Returns the operation that blasArg will perform.
    let blasArgOperation (manikin: ArrayNDManikinT) shared willOverwrite =
        let st = ArrayND.stride manikin
        match st.[st.Length-2 ..] with
        | [_; 1] when not (shared && willOverwrite) -> BlasArgId
        | [1; _] when not (shared && willOverwrite) -> BlasArgTranspose
        | _ -> BlasArgCopy

    /// Computes desired source views given desired target view.
    /// There is no guarantee that the desired source views will be used.
    let srcReqs cudaEnv {TargetShape=trgtShape
                         TargetRequest=reqChViews
                         Op=op
                         SrcShapes=srcShapes} : ChannelReqsT list =
        let nSrcs = List.length srcShapes

        /// Creates a channel request for the default channel.
        let defaultChReq view : ChannelReqsT =
            Map [defaultChannelId, view] 

        /// The view request for the default channel of the target.
        let targetDefChReq = reqChViews.[defaultChannelId]

        /// Requests the default channel of all sources without
        /// a storage requests.
        let defaultSrcWithNoViewReq =
            List.replicate nSrcs (defaultChReq None)

        /// Requests the default channel of the first source to be evaluated 
        /// into our requested target view of the the default channel.
        let inplaceFirstSrcReq =
            match nSrcs with
            | 0 -> []
            | 1 -> [defaultChReq targetDefChReq]
            | _ -> defaultChReq targetDefChReq :: List.replicate (nSrcs-1) (defaultChReq None)

        match op with
        | ULeafOp _ -> []

        // unary element-wise
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
        // tensor ops
        | UUnaryOp (Diag _) -> defaultSrcWithNoViewReq
        | UUnaryOp (DiagMat _) -> defaultSrcWithNoViewReq
        | UUnaryOp Invert -> defaultSrcWithNoViewReq          
        // reductions
        | UUnaryOp Sum -> defaultSrcWithNoViewReq
        | UUnaryOp (SumAxis _) -> defaultSrcWithNoViewReq
        // shape operations
        | UUnaryOp (Reshape _) ->        
            match targetDefChReq with
            | Some rv when ArrayND.isC rv ->
                [defaultChReq (Some (ArrayND.reshapeView srcShapes.[0] rv))]
            | _ -> defaultSrcWithNoViewReq
        | UUnaryOp (DoBroadcast _) -> defaultSrcWithNoViewReq
        | UUnaryOp (SwapDim (ax1, ax2)) ->
            match targetDefChReq with
            | Some rv -> [defaultChReq (Some (ArrayND.swapDim ax1 ax2 rv))]
            | _ -> defaultSrcWithNoViewReq

        // variable access
        | UUnaryOp (StoreToVar vs) ->
            match cudaEnv.VarStorLoc |> Map.find vs with
            | LocDev -> 
                // request to store directly into external var
                // we assume that all device input vars are continguous
                [defaultChReq (Some (ArrayNDManikin.externalC (MemExternal vs) srcShapes.[0]))]
            | LocHost -> defaultSrcWithNoViewReq
            | loc -> unsupLoc loc
        // misc
        | UUnaryOp (Annotated _) -> inplaceFirstSrcReq

        // binary element-wise
        | UBinaryOp Add -> inplaceFirstSrcReq
        | UBinaryOp Substract -> inplaceFirstSrcReq
        | UBinaryOp Multiply -> inplaceFirstSrcReq
        | UBinaryOp Divide -> inplaceFirstSrcReq
        | UBinaryOp Modulo -> inplaceFirstSrcReq
        | UBinaryOp Power -> inplaceFirstSrcReq
        // matrix/tensor operations
        | UBinaryOp Dot -> defaultSrcWithNoViewReq
        | UBinaryOp TensorProduct -> defaultSrcWithNoViewReq     

        // nary
        | UNaryOp Discard -> defaultSrcWithNoViewReq
        | UNaryOp (Subtensor _) -> defaultSrcWithNoViewReq
        | UNaryOp (SetSubtensor _) -> 
            // "a" can be evaluated into requested manikin, but "b" (the replacement value) must be placed
            // in a temporary manikin and copied over to avoid race conditions.
            inplaceFirstSrcReq
        | UNaryOp (ExtensionOp eop) -> failwith "not implemented yet"


    /// computes the definitive target view of an op given its source views
    let trgtGivenSrcs compileEnv {MemAllocator=memAllocator
                                  TargetRequest=reqChViews
                                  Op=op
                                  Metadata={TargetType=typ
                                            TargetNShape=trgtShape}
                                  Srcs=srcs} =

        /// Default channels of all sources.
        let srcsDefaultCh, srcsDefaultChShared =
            srcs
            |> List.map (fun srcChs -> srcChs.[defaultChannelId])
            |> List.unzip

        /// The view request for the default channel of the target.
        let targetDefChReq = reqChViews.[defaultChannelId]

        /// Target for default channel.
        let defaultChTrgt view shared : ChannelManikinsAndSharedT =
            Map [defaultChannelId, (view, shared)] 

        // New allocated target for default channel.
        let newDefaultChTrgt () = 
            defaultChTrgt (ArrayNDManikin.newC memAllocator typ trgtShape) false        

        /// True if specified manikin overlaps with any channel of any source.
        let overlappingWithAnySrc (rv: ArrayNDManikinT) =
            srcs
            |> List.exists (Map.exists (fun ch (view, shared) -> ArrayND.overlapping rv view))

        /// default channel target that shares no elements with any srcView 
        let defaultChOutplaceTrgt () =
            match targetDefChReq with
            | Some rv when not (overlappingWithAnySrc rv) -> defaultChTrgt rv false
            | _ -> newDefaultChTrgt () 
             
        /// default channel target that shares no elements with any srcView and can be used for BLAS
        let defaultChOutplaceBlasTrgt () = 
            match targetDefChReq with
            | Some rv when ArrayNDManikin.canBeBlasTarget rv && 
                           not (overlappingWithAnySrc rv) -> defaultChTrgt rv false
            | _ -> 
                defaultChTrgt (ArrayNDManikin.newBlasTarget memAllocator typ trgtShape) false

        /// default channel target that shares no elements with any srcView and the transpose of which can be used for BLAS
        let defaultChOutplaceTransposedBlasTrgt () = 
            match targetDefChReq with
            | Some rv when ArrayNDManikin.canBeBlasTarget rv.T && 
                           not (overlappingWithAnySrc rv) -> defaultChTrgt rv false
            | _ -> 
                defaultChTrgt (ArrayNDManikin.newC memAllocator typ trgtShape) false  

        /// Default channel target that reuses the default channel of a srcView, 
        /// if it may be overwritten. Otherwise uses defaultChOutplaceTrgt.
        let defaultChInplaceOvrwrtTrgt () : ChannelManikinsAndSharedT =
            match srcs 
                  |> List.tryFind (fun srcChs ->
                                    let view, shared = srcChs.[defaultChannelId] 
                                    not (ArrayND.isBroadcasted view) && not shared) with
            | Some srcChs -> Map [defaultChannelId, srcChs.[defaultChannelId]]
            | None -> defaultChOutplaceTrgt ()     

        match op with
        // variable access
        | ULeafOp (Var vs) ->       
            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev ->
                // we assume that all device input vars are contiguous
                defaultChTrgt (ArrayNDManikin.externalC (MemExternal vs) trgtShape) true
            | LocHost ->
                // will transfer variable from host to device during execution
                // need contiguous memory for that
                match targetDefChReq with
                | Some rv when ArrayND.isC rv -> defaultChTrgt rv false
                | _ -> defaultChTrgt (ArrayNDManikin.newC memAllocator typ trgtShape) false    
            | loc -> unsupLoc loc                    
        // tensor creation
        | ULeafOp _ -> defaultChOutplaceTrgt ()      

        // unary element-wise
        | UUnaryOp Negate -> defaultChInplaceOvrwrtTrgt ()                       
        | UUnaryOp Abs -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp SignT -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Log -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Log10 -> defaultChInplaceOvrwrtTrgt ()                          
        | UUnaryOp Exp -> defaultChInplaceOvrwrtTrgt ()                           
        | UUnaryOp Sin -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Cos -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Tan -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Asin -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Acos -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Atan -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Sinh -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Cosh -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Tanh -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Sqrt -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Ceil -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Floor -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Round -> defaultChInplaceOvrwrtTrgt ()
        | UUnaryOp Truncate -> defaultChInplaceOvrwrtTrgt ()    
        // tensor ops
        | UUnaryOp (Diag (ax1, ax2)) ->
            defaultChTrgt (ArrayND.diagAxis ax1 ax2 srcsDefaultCh.[0]) srcsDefaultChShared.[0]
        | UUnaryOp (DiagMat (ax1, ax2)) -> defaultChOutplaceTrgt ()
        | UUnaryOp Invert -> 
            // If source will be transposed, then target will also be transposed.
            // Thus, in this case, we must request an array the transpose of which 
            // can be used as a BLAS target.
            match blasArgOperation srcsDefaultCh.[0] srcsDefaultChShared.[0] true with
            | BlasArgTranspose -> defaultChOutplaceBlasTrgt ()
            | _ -> defaultChOutplaceTransposedBlasTrgt ()
        // reductions
        | UUnaryOp Sum -> defaultChOutplaceTrgt ()
        | UUnaryOp (SumAxis _) -> defaultChOutplaceTrgt ()
        // shape operations
        | UUnaryOp (Reshape _) ->        
            // TODO: optimize: check if copy is really necessary
            if ArrayND.isC srcsDefaultCh.[0] then
                defaultChTrgt (ArrayND.reshapeView trgtShape srcsDefaultCh.[0]) srcsDefaultChShared.[0] 
            else defaultChOutplaceTrgt () // will copy
        | UUnaryOp (DoBroadcast _) ->
            defaultChTrgt (ArrayND.broadcastToShape trgtShape srcsDefaultCh.[0]) srcsDefaultChShared.[0]
        | UUnaryOp (SwapDim (ax1, ax2)) ->
            defaultChTrgt (ArrayND.swapDim ax1 ax2 srcsDefaultCh.[0]) srcsDefaultChShared.[0]
        // variable access
        | UUnaryOp (StoreToVar _) -> 
            // output of StoreToVar is empty 
            newDefaultChTrgt ()
        // misc
        | UUnaryOp (Annotated _) -> defaultChTrgt srcsDefaultCh.[0] srcsDefaultChShared.[0]

        // binary element-wise
        | UBinaryOp Add -> defaultChInplaceOvrwrtTrgt ()
        | UBinaryOp Substract -> defaultChInplaceOvrwrtTrgt ()
        | UBinaryOp Multiply -> defaultChInplaceOvrwrtTrgt ()
        | UBinaryOp Divide -> defaultChInplaceOvrwrtTrgt ()
        | UBinaryOp Modulo -> defaultChInplaceOvrwrtTrgt ()
        | UBinaryOp Power -> defaultChInplaceOvrwrtTrgt ()
        // matrix/tensor operations
        | UBinaryOp Dot -> defaultChOutplaceBlasTrgt ()
        | UBinaryOp TensorProduct -> defaultChOutplaceTrgt ()

        // nary
        | UNaryOp Discard -> defaultChOutplaceTrgt ()
        | UNaryOp (Subtensor srs) -> 
            if SimpleRangesSpec.isDynamic srs then 
                // dynamic sub-tensors will be copied out of the src
                defaultChOutplaceTrgt ()
            else
                // symbolic sub-tensors use a view of the src 
                let rng = SimpleRangesSpec.eval (fun _ -> failwith "must be static") srs
                defaultChTrgt (srcsDefaultCh.[0].[rng] :?> ArrayNDManikinT) srcsDefaultChShared.[0]
        | UNaryOp (SetSubtensor _) ->
            if not (srcsDefaultChShared.[0]) then 
                defaultChTrgt srcsDefaultCh.[0] false
            else defaultChOutplaceTrgt ()
        | UNaryOp (ExtensionOp eop) -> failwith "not implemented yet"
   
    /// execution item to launch the given kernel template function
    let execItemsForKernel cppFuncName tmplTmpls argTmpls workDim = 
        let cFuncTmpl =
            {FuncName=cppFuncName;
             Domain=KernelFunc;
             TmplArgs=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) tmplTmpls;
             RetType="void";
             ArgTypes=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) argTmpls;}    
        [LaunchKernel(cFuncTmpl, workDim, argTmpls)]

    /// returns the CUDA work dimensions for an element-wise operation
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


    /// returns the C++ template instantiation code for the given template and argument list
    let cppTemplateInstantiation tmpl args =
        if List.isEmpty args then tmpl
        else sprintf "%s<%s>" tmpl (args |> String.concat ", ")

    /// function name of element-wise wrapper and its arguments for the given target, operation and sources
    let elemwiseFuncnameAndArgs trgt cOp srcViews =
        let args = 
            (cOp :> ICudaArgTmpl) ::
            ((ArrayNDArgTmpl trgt) :> ICudaArgTmpl) ::
            (List.map (fun v -> (ArrayNDArgTmpl v) :> ICudaArgTmpl) srcViews)

        let nSrc = List.length srcViews
        let hetero = srcViews |> List.exists (fun sv -> (ArrayND.shape trgt) <> (ArrayND.shape sv))
        let indexedStr = if (cOp :> ICudaOp).IsIndexed then "Indexed" else ""
        let dimsStr = if hetero then "Heterogenous" else sprintf "%dD" (ArrayND.nDims trgt)
        let funcName = sprintf "elemwise%dAry%s%s" nSrc dimsStr indexedStr 
        funcName, args

    /// execution items for an element-wise operation
    let execItemsForElemwise trgt cOp srcViews =
        if srcViews |> List.exists (fun sv -> ArrayND.nElems trgt <> ArrayND.nElems sv) then
            failwithf "a source of an elemwise op has different number of elements than target"

        let funcName, args = elemwiseFuncnameAndArgs trgt cOp srcViews
        let hetero = srcViews |> List.exists (fun sv -> (ArrayND.shape trgt) <> (ArrayND.shape sv))
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

        execItemsForKernel 
            "copyFromDynamicSubtensor" 
            [ArrayNDArgTmpl trgt; srcTmpl; srcDynTmpl; CPPTemplateValue nDimsStr]
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

        execItemsForKernel 
            "copyToDynamicSubtensor" 
            [trgtTmpl; trgtDynTmpl; CPPTemplateValue nDimsStr; ArrayNDArgTmpl src]
            [trgtTmpl; trgtIdxPntrsTmpl; ArrayNDArgTmpl src]
            (workDimForElemwise src false)


    /// generate ExecItems to call a C++ template function
    let execItemsForCFunc<'FuncDelegate when 'FuncDelegate :> System.Delegate> tmplTmpls argTmpls =
        let cDelegateType = typeof<'FuncDelegate>
        let cAttributes = cDelegateType.GetCustomAttributes(typeof<CPPFuncNameAttribute>, false)
        if Array.isEmpty cAttributes then
            failwithf "CPPFuncName attribute is missing on delegate %A" cDelegateType
        let cppFuncNameAttribute = cAttributes.[0] :?> CPPFuncNameAttribute
        let cppFuncName = cppFuncNameAttribute.CPPFuncName

        let cFuncTmpl =
            {FuncName=cppFuncName;
             Domain=CPPFunc;
             TmplArgs=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) tmplTmpls;
             RetType="void";
             ArgTypes=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) argTmpls;}    
        [CallCFunc(cFuncTmpl, cDelegateType, argTmpls)]


    /// generates ExecItems to copy srcView to trgtView 
    let copyExecItems trgt src =
        if ArrayND.nElems trgt <> ArrayND.nElems src then
            failwithf "cannot copy array with %d elements to array with %d elements"
                (ArrayND.nElems trgt) (ArrayND.nElems src)
        execItemsForElemwise trgt (NoArgEOpArgTmpl("IdEOp_t", false)) [src]

    /// If all batch dimensions (all dimensions but the last two) of the array are of
    /// size one, a view of the last two dimensions is returned.
    /// Otherwise the original array is returned.
    let trimUnitaryBatchedBlasDims (manikin: ArrayNDManikinT) =
        let nd = manikin.NDims
        if nd > 2 then
            let isUnitary = manikin.Shape.[0..nd-3] |> List.forall ((=) 1)
            if isUnitary then
                manikin |> ArrayND.reshapeView manikin.Shape.[nd-2..]
            else manikin
        else manikin           

    /// BLAS input argument passing, so that orientation is preserved.
    /// Can return copy items if deemed necessary.
    let blasArg memAllocator (manikin: ArrayNDManikinT) shared willOverwrite =
        let manikin = trimUnitaryBatchedBlasDims manikin
        if ArrayND.nDims manikin < 2 then
            failwith "need at least 2-dimensional array for BLAS argument"
        match blasArgOperation manikin shared willOverwrite with
        | BlasArgId        -> manikin, BlasTranspose, [], shared
        | BlasArgTranspose -> ArrayND.transpose manikin, BlasId, [], shared
        | BlasArgCopy -> 
            let tmpView = ArrayNDManikin.newC memAllocator (ArrayNDManikin.typeName manikin) (ArrayND.shape manikin)
            let copyOps = copyExecItems tmpView manikin
            tmpView, BlasTranspose, copyOps, false

    /// BLAS target argument passing, so that orientation is preserved
    let blasTarget (manikin: ArrayNDManikinT) =
        let manikin = trimUnitaryBatchedBlasDims manikin
        if not (ArrayNDManikin.canBeBlasTarget manikin) then
            failwithf "cannot use specified view with shape %A and stride %A as BLAS target" 
                manikin.Shape (ArrayND.stride manikin)
        ArrayND.transpose manikin

    let execItemsForSum memAllocator trgt src =
        // C++ signature:
        // void sum(TTarget &trgt, TSrc &src, 
        //          CUstream &stream, char *tmp_buffer, size_t tmp_buffer_size);
        let tmpSize = ArrayNDManikin.sizeInBytes src
        let tmp = memAllocator TypeName.ofType<byte> tmpSize MemAllocDev       
        execItemsForCFunc<CPPSum> [] [ArrayNDArgTmpl trgt; ArrayNDArgTmpl src;
                                      ExecStreamArgTmpl(); BytePtrArgTmpl tmp; SizeTArgTmpl tmpSize]

    let execItemsForSumAxis memAllocator ax trgt src =
        // we need to swap axes so that the axes the summation is performed over comes last
        let nd = ArrayND.nDims src
        let axOrder = Seq.concat [ {0 .. ax-1}; {ax + 1 .. nd - 1}; Seq.singleton ax] |> Seq.toList
        let srcAdj = ArrayND.reorderAxes axOrder src

        // C++ signature:
        // void sumLastAxis(TTarget &trgt, TSrc &src, 
        //                  CUstream &stream, char *tmp_buffer, size_t tmp_buffer_size);
        let tmpSize = ArrayNDManikin.sizeInBytes srcAdj
        let tmp = memAllocator TypeName.ofType<byte> tmpSize MemAllocDev
        execItemsForCFunc<CPPSumLastAxis> [] [ArrayNDArgTmpl trgt; ArrayNDArgTmpl srcAdj;
                                                ExecStreamArgTmpl(); BytePtrArgTmpl tmp; SizeTArgTmpl tmpSize]

    /// returns the execution units for the specified op
    let execItemsForOp compileEnv {MemAllocator=memAllocator
                                   Target=trgtChs
                                   Op=op
                                   Metadata=metadata
                                   Srcs=srcsAndShared
                                   SubmitInitItems=submitInit} =

        /// Default channel of target.
        let defaultChTrgt = trgtChs.[defaultChannelId]

        /// Default channels of all sources.
        let srcsDefaultCh, srcsDefaultChShared =
            srcsAndShared
            |> List.map (fun srcChs -> srcChs.[defaultChannelId])
            |> List.unzip
    
        // set pointer array values either during initialization (for allocated arrays)
        // or runtime (for variable arrays)
        let appendPointerArrayItems (tmpl: BlasTransposedMatrixBatchTmpl) execItems =
            match tmpl.Manikin.Storage with
            | MemAlloc _ -> submitInit [BlasInitPointerArray tmpl]; execItems
            | MemExternal _ -> execItems @ [BlasInitPointerArray tmpl]

        match op with 
        // tensor creation
        | ULeafOp (Identity _) -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("DiagonalOneIEOp_t", true)) []
        | ULeafOp (Zeros _) -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("ZerosEOp_t", false)) []
        | ULeafOp (ScalarConst f) -> execItemsForElemwise defaultChTrgt (ConstEOpArgTmpl f) [] 
        | ULeafOp (SizeValue sv) -> 
            let value = Convert.ChangeType(SizeSpec.eval sv, defaultChTrgt.DataType)
            let opType = typedefof<ConstEOpArgTmpl<_>>.MakeGenericType(defaultChTrgt.DataType)
            let op = Activator.CreateInstance(opType, value) :?> ICudaOpAndArgTmpl 
            execItemsForElemwise defaultChTrgt op [] 
        // variable access
        | ULeafOp (Var vs) -> 
            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev -> []
            | LocHost -> 
                // we assume that host variable has continguous stride and zero offset
                let hv = ArrayNDManikin.externalC (MemExternal vs) (ArrayND.shape defaultChTrgt)
                [MemcpyHtoD(ArrayNDHostRegMemRngTmpl(hv), ArrayNDDevMemRngTmpl(defaultChTrgt))]       
            | loc -> unsupLoc loc
        // unary element-wise
        | UUnaryOp Negate -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("NegateEOp_t", false)) srcsDefaultCh
        | UUnaryOp Abs -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("AbsEOp_t", false)) srcsDefaultCh
        | UUnaryOp SignT -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("SignTEOp_t", false)) srcsDefaultCh
        | UUnaryOp Log -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("LogEOp_t", false)) srcsDefaultCh
        | UUnaryOp Log10 -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("Log10EOp_t", false)) srcsDefaultCh
        | UUnaryOp Exp -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("ExpEOp_t", false)) srcsDefaultCh
        | UUnaryOp Sin -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("SinEOp_t", false)) srcsDefaultCh
        | UUnaryOp Cos -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("CosEOp_t", false)) srcsDefaultCh
        | UUnaryOp Tan -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("TanEOp_t", false)) srcsDefaultCh
        | UUnaryOp Asin -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("AsinEOp_t", false)) srcsDefaultCh
        | UUnaryOp Acos -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("AcosEOp_t", false)) srcsDefaultCh
        | UUnaryOp Atan -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("AtanEOp_t", false)) srcsDefaultCh
        | UUnaryOp Sinh -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("SinhEOp_t", false)) srcsDefaultCh
        | UUnaryOp Cosh -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("CoshEOp_t", false)) srcsDefaultCh
        | UUnaryOp Tanh -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("TanhEOp_t", false)) srcsDefaultCh
        | UUnaryOp Sqrt -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("SqrtEOp_t", false)) srcsDefaultCh
        | UUnaryOp Ceil -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("CeilEOp_t", false)) srcsDefaultCh
        | UUnaryOp Floor -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("FloorEOp_t", false)) srcsDefaultCh
        | UUnaryOp Round -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("RoundEOp_t", false)) srcsDefaultCh
        | UUnaryOp Truncate -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("TruncateEOp_t", false)) srcsDefaultCh
        // reductions
        | UUnaryOp Sum -> execItemsForSum memAllocator defaultChTrgt srcsDefaultCh.[0]
        | UUnaryOp (SumAxis ax) -> execItemsForSumAxis memAllocator ax defaultChTrgt srcsDefaultCh.[0]

        // tensor ops
        | UUnaryOp (Diag _) -> []
        | UUnaryOp (DiagMat (ax1, ax2)) ->
            let trgtDiag = ArrayND.diagAxis ax1 ax2 defaultChTrgt
            let zeroItems = execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("ZerosEOp_t", false)) []
            let copyItems = copyExecItems trgtDiag srcsDefaultCh.[0]
            zeroItems @ copyItems
        | UUnaryOp Invert ->
            let aView, _, aCopyItems, _ = blasArg memAllocator srcsDefaultCh.[0] srcsDefaultChShared.[0] true

            let tView =
                // If the source is transposed by us then the target must be transposed by us 
                // as well to preserve orientation. The blasTarget function always transposes.
                match blasArgOperation srcsDefaultCh.[0] srcsDefaultChShared.[0] true with
                | BlasArgTranspose -> blasTarget defaultChTrgt
                | _ -> blasTarget (ArrayND.transpose defaultChTrgt)

            // allocate variables and initialize pointer arrays
            let aArg = BlasTransposedMatrixBatchTmpl (aView, memAllocator)
            let tArg = BlasTransposedMatrixBatchTmpl (tView, memAllocator)
            let pivot = BlasIntArrayTmpl (aArg.Rows * aArg.NSamples, memAllocator)
            let info = BlasIntArrayTmpl (aArg.NSamples, memAllocator)
            let ptrAryItems =
                []
                |> appendPointerArrayItems aArg
                |> appendPointerArrayItems tArg

            // Perform LU decomposition in-place in b.
            let luItems = [BlasGetrfBatched (aArg, pivot, info)]

            // Perform matrix inversion from b into t.
            let invItems = [BlasGetriBatched (aArg, pivot, tArg, info)]

            aCopyItems @ ptrAryItems @ luItems @ invItems

        // shape operations
        | UUnaryOp (Reshape _) ->
            if defaultChTrgt <> srcsDefaultCh.[0] then 
                copyExecItems defaultChTrgt srcsDefaultCh.[0]
            else []
        | UUnaryOp (DoBroadcast _) -> []
        | UUnaryOp (SwapDim _) -> []
        // variable access
        | UUnaryOp (StoreToVar vs) ->
            let varShp, varType = 
                ArrayND.shape srcsDefaultCh.[0], srcsDefaultCh.[0].TypeName

            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev when srcsDefaultCh.[0].Storage = (MemExternal vs) ->
                // Source was evaluated directly into the variable storage.
                // No copy necessary.
                []
            | LocDev  -> 
                // Our source has not been evaluated directly into the variable storage.
                // Therefore we need to copy into the variable.
                // We assume that all device vars are continguous.
                let dv = ArrayNDManikin.externalC (MemExternal vs) varShp
                copyExecItems dv srcsDefaultCh.[0]
            | LocHost ->            
                let copyItems, memcpySrc = 
                    if ArrayND.isC srcsDefaultCh.[0] then 
                        // Source is contiguous. Can directly copy to host.
                        [], srcsDefaultCh.[0]
                    else
                        // Need to copy to temporary contiguous storage first.
                        let tmp = ArrayNDManikin.newC memAllocator varType varShp
                        copyExecItems tmp srcsDefaultCh.[0], tmp

                // We assume that all host vars are continguous.
                // trgtView has contingous stride
                let hv = ArrayNDManikin.externalC (MemExternal vs) varShp
                copyItems @ [MemcpyDtoH(ArrayNDDevMemRngTmpl(memcpySrc), ArrayNDHostRegMemRngTmpl(hv))]   
            | loc -> unsupLoc loc                              
        // misc
        | UUnaryOp (Annotated _) -> []

        // binary element-wise
        | UBinaryOp Add ->       execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("AddEOp_t",       false)) srcsDefaultCh
        | UBinaryOp Substract -> execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("SubstractEOp_t", false)) srcsDefaultCh
        | UBinaryOp Multiply ->  execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("MultiplyEOp_t",  false)) srcsDefaultCh
        | UBinaryOp Divide ->    execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("DivideEOp_t",    false)) srcsDefaultCh
        | UBinaryOp Modulo ->    execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("ModuloEOp_t",    false)) srcsDefaultCh
        | UBinaryOp Power ->     execItemsForElemwise defaultChTrgt (NoArgEOpArgTmpl("PowerEOp_t",     false)) srcsDefaultCh
        // matrix/tensor operations
        | UBinaryOp Dot -> 
            let aView, aOp, aCopyItems, aShared = blasArg memAllocator srcsDefaultCh.[0] srcsDefaultChShared.[0] false
            let bView, bOp, bCopyItems, bShared = blasArg memAllocator srcsDefaultCh.[1] srcsDefaultChShared.[1] false
            let tView = blasTarget defaultChTrgt
        
            let blasItems =    
                match aView.NDims with
                | 0 | 1 -> failwith "BLAS matrix must be at least two dimensional" 
                | 2 -> // single matrix multiplication
                    [BlasGemm(aOp, bOp, 1.0f, 
                              BlasTransposedMatrixTmpl(aView), 
                              BlasTransposedMatrixTmpl(bView),
                              0.0f, BlasTransposedMatrixTmpl(tView))]                
                | _ -> // batched matrix multiplication

                    // allocate memory for pointer arrays and create argument templates
                    let aTmpl = BlasTransposedMatrixBatchTmpl(aView, memAllocator)   
                    let bTmpl = BlasTransposedMatrixBatchTmpl(bView, memAllocator)   
                    let tTmpl = BlasTransposedMatrixBatchTmpl(tView, memAllocator)   

                    let execItems =
                        []
                        |> appendPointerArrayItems aTmpl
                        |> appendPointerArrayItems bTmpl
                        |> appendPointerArrayItems tTmpl

                    execItems @ [BlasGemmBatched(aOp, bOp, 1.0f, aTmpl, bTmpl, 0.0f, tTmpl)]

            aCopyItems @ bCopyItems @ blasItems

        | UBinaryOp TensorProduct -> [] // TODO

        // nary
        | UNaryOp Discard -> []
        | UNaryOp (Subtensor srs) ->
            if SimpleRangesSpec.isDynamic srs then 
                // copy dynamic subtensor out of the src
                execItemsForCopyFromDynamicSubtensor defaultChTrgt 
                    srcsDefaultCh.[0] srs (List.tail srcsDefaultCh)
            else [] // symbolic subtensor uses a slice of the src view
        | UNaryOp (SetSubtensor srs) ->
            // copy "a" if necessary
            let copyItems = 
                if defaultChTrgt <> srcsDefaultCh.[0] then 
                    copyExecItems defaultChTrgt srcsDefaultCh.[0] else []
            // copy "b" into a
            let setItems =
                execItemsForCopyToDynamicSubtensor defaultChTrgt srs 
                    (List.skip 2 srcsDefaultCh) srcsDefaultCh.[1]
            copyItems @ setItems
        | UNaryOp (ExtensionOp eop) -> failwith "not implemented yet"

                
    /// returns the execution units for tracing the result
    let traceItemsForExpr compileEnv {MemAllocator=memAllocator
                                      Target=trgtChs
                                      Expr=uexpr} =
        /// Default channel of target.
        let defaultChTrgt = trgtChs.[defaultChannelId]

        [Trace (uexpr, defaultChTrgt)]


    /// generates CUDA execution units that will evaluate the given unified expression
    let exprToCudaExecUnits (compileEnv: CudaCompileEnvT) =
        ExecUnit.exprToExecUnits {
            ExecItemsForOp=execItemsForOp compileEnv
            TraceItemsForExpr=traceItemsForExpr compileEnv
            TrgtGivenSrcs=trgtGivenSrcs compileEnv
            SrcReqs=srcReqs compileEnv
        } 




