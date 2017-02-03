namespace SymTensor.Compiler.Cuda

open System

open Basics
open Basics.Cuda
open ArrayNDNS
open SymTensor
open SymTensor.Compiler
open Expr
open UExprTypes


[<AutoOpen>]
module CudaExecUnitTypes =

    /// information for executing a loop
    type ExecLoopInfoT = {
        Length:                int64
        Channels:              Map<ChannelT, LoopEval.LoopChannelInfoT>
        Vars:                  Map<VarSpecT, LoopInputT>
        Workspace:             SubWorkspaceT
        Args:                  ArrayNDManikinT list
        IterManikin:           ArrayNDManikinT
        ItersRemainingManikin: ArrayNDManikinT
    }

    /// A custom CUDA execution item.
    type ICudaExecItem =
        /// Asynchronously execute the item on the specified CUDA stream.
        abstract Execute: CudaExecEnvT -> StreamT -> unit

    /// a CUDA operation that will be assigned to and executed in a CUDA stream
    type CudaExecItemT =
        // memory operations
        | MemcpyDtoD   of IDevMemRngTmpl * IDevMemRngTmpl
        | MemcpyHtoD   of IHostMemRngTmpl * IDevMemRngTmpl
        | MemcpyDtoH   of IDevMemRngTmpl * IHostMemRngTmpl
        | MemsetSingle of single * IDevMemRngTmpl
        | MemsetUInt32 of uint32 * IDevMemRngTmpl
        // execution control
        | LaunchKernel of TmplInstT * WorkDimT * (ICudaArgTmpl list)
        | CallCFunc    of TmplInstT * System.Type * (ICudaArgTmpl list)
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
        | BlasInitPointerArray  of BlasTransposedMatrixBatchTmpl
        // loop
        | ExecLoop              of ExecLoopInfoT
        // misc
        | Trace                 of UExprT * ArrayNDManikinT
        | TraceEnteringLoop     of UExprT
        | TraceLeavingLoop      of UExprT
        | PrintWithMsg          of string * ArrayNDManikinT
        | DumpValue             of string * ArrayNDManikinT
        | CheckNonFiniteCounter of string * ArrayNDManikinT
        // extension item
        | ExtensionExecItem     of ICudaExecItem


    type SrcReqsHelpersT = {
        /// Creates a channel request for the default channel.
        DfltChReq:               ArrayNDManikinT option -> ChannelReqsT
        /// The view request for the default channel of the target.
        TrgtDfltChReq:           unit -> ArrayNDManikinT option
        /// Requests the default channel of all sources without
        /// a storage requests.
        DfltSrcWithNoViewReq:    unit -> ChannelReqsT list
        /// Requests the default channel of the first source to be evaluated 
        /// into our requested target view of the the default channel.
        InplaceFirstSrcReq:      unit -> ChannelReqsT list
    }

    type TrgtGivenSrcsHelpersT = {
        /// Default channels of all sources.
        SrcsDfltCh:                         unit -> ArrayNDManikinT list
        /// Default channel is shared for all sources?
        SrcsDfltChShared:                   unit -> bool list   
        /// The view request for the default channel of the target.
        TrgtDefChReq:                       unit -> ArrayNDManikinT option
        /// Target for default channel.
        DfltChTrgt:                         ArrayNDManikinT -> bool -> ChannelManikinsAndSharedT
        // New allocated target for default channel.
        NewDfltChTrgt:                      unit -> ChannelManikinsAndSharedT
        /// default channel target that shares no elements with any srcView 
        DfltChOutplaceTrgt:                 unit -> ChannelManikinsAndSharedT     
        /// default channel target that shares no elements with any srcView 
        /// and can be used for BLAS   
        DfltChOutplaceBlasTrgt:             unit -> ChannelManikinsAndSharedT
        /// default channel target that shares no elements with any srcView 
        /// and the transpose of which can be used for BLAS
        DfltChOutplaceTransposedBlasTrgt:   unit -> ChannelManikinsAndSharedT
        /// Default channel target that reuses the default channel of a srcView, 
        /// if it may be overwritten. Otherwise uses DfltChOutplaceTrgt.
        DfltChInplaceOvrwrtTrgt:            unit -> ChannelManikinsAndSharedT
    }

    type ExecItemsHelpersT = {
        /// Default channels of all sources.
        SrcsDfltCh:                         unit -> ArrayNDManikinT list
        /// Default channel is shared for all sources?
        SrcsDfltChShared:                   unit -> bool list 
        /// Target for default channel.
        DfltChTrgt:                         unit -> ArrayNDManikinT           
        // Set pointer array values either during initialization (for allocated arrays)
        // or runtime (for variable arrays).
        AppendPointerArrayItems:            BlasTransposedMatrixBatchTmpl -> 
                                            CudaExecItemT list -> CudaExecItemT list
    }

    /// A unified op that can be compiled to CUDA execution items.
    type ICudaUOp =
        inherit IUOp

        /// Computes desired source manikins given desired target manikin.
        /// There is no guarantee that the desired source manikins will be used.
        /// Also, it is not obligatory to use the requested target manikin.
        abstract SrcReqs: CudaCompileEnvT -> SrcReqsArgs -> SrcReqsHelpersT -> 
                          ChannelReqsT list

        /// Computes the definitive target manikin of an op given its source manikins.
        /// It is not obligatory to use the requested target manikin.
        abstract TrgtGivenSrcs: CudaCompileEnvT -> TrgtGivenSrcsArgs -> 
                                TrgtGivenSrcsHelpersT -> ChannelManikinsAndSharedT
    
        /// Returns the execution items for the op.
        /// It must read from the given source manikin and write to the target manikin.
        /// Additional memory may be allocated for temporary results.
        abstract ExecItems: CudaCompileEnvT -> ExecItemsForOpArgs<CudaExecItemT> -> 
                            ExecItemsHelpersT -> CudaExecItemT list


module CudaExecUnit =
    open ManagedCuda.BasicTypes

    /// converts a IUOp or a IOp to a ICudaUOp
    let toCudaUOp (uop: obj)  =
        match uop with
        | :? ICudaUOp as cudaUOp -> cudaUOp
        | _ -> failwith "For CUDA compilation the op %A needs to implement the ICudaUOp interface."

    /// failure for extra ops
    let needExtra op =
        failwith "the op %A requires extra handling and should have been converted to an UExtraOp"

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
        let st, shp = ArrayND.stride manikin, ArrayND.shape manikin
        match st.[st.Length-2 ..], shp.[st.Length-2 ..] with
        | [m;  1L], [ms; ns] when m >= max 1L ns && not (shared && willOverwrite) -> BlasArgId
        | [1L; n],  [ms; ns] when n >= max 1L ms && not (shared && willOverwrite) -> BlasArgTranspose
        | _ -> BlasArgCopy


    /// Computes desired source views given desired target view.
    /// There is no guarantee that the desired source views will be used.
    let srcReqs cudaEnv ({TargetRequest=reqChViews
                          Op=op
                          Metadata=metadata
                          SrcShapes=srcShapes} as args) : ChannelReqsT list =

        /// number of arguments
        let nSrcs = List.length srcShapes

        /// Creates a channel request for the default channel.
        let dfltChReq view : ChannelReqsT = Map [dfltChId, view] 

        /// The view request for the default channel of the target.
        let trgtDfltChReq () = reqChViews.[dfltChId]

        /// Requests the default channel of all sources without
        /// a storage requests.
        let dfltSrcWithNoViewReq () = List.replicate nSrcs (dfltChReq None)

        /// Requests the default channel of the first source to be evaluated 
        /// into our requested target view of the the default channel.
        let inplaceFirstSrcReq () =
            match nSrcs with
            | 0 -> []
            | 1 -> [dfltChReq (trgtDfltChReq ())]
            | _ -> dfltChReq (trgtDfltChReq ()) :: List.replicate (nSrcs-1) (dfltChReq None)

        let helpers = {
            DfltChReq               = dfltChReq
            TrgtDfltChReq           = trgtDfltChReq
            DfltSrcWithNoViewReq    = dfltSrcWithNoViewReq
            InplaceFirstSrcReq      = inplaceFirstSrcReq
        }

        match op with
        | ULeafOp _ -> []

        // unary element-wise
        | UUnaryOp Negate -> inplaceFirstSrcReq ()                       
        | UUnaryOp Abs -> inplaceFirstSrcReq ()
        | UUnaryOp SignT -> inplaceFirstSrcReq ()
        | UUnaryOp Log -> inplaceFirstSrcReq ()
        | UUnaryOp Log10 -> inplaceFirstSrcReq ()                          
        | UUnaryOp Exp -> inplaceFirstSrcReq ()                           
        | UUnaryOp Sin -> inplaceFirstSrcReq ()
        | UUnaryOp Cos -> inplaceFirstSrcReq ()
        | UUnaryOp Tan -> inplaceFirstSrcReq ()
        | UUnaryOp Asin -> inplaceFirstSrcReq ()
        | UUnaryOp Acos -> inplaceFirstSrcReq ()
        | UUnaryOp Atan -> inplaceFirstSrcReq ()
        | UUnaryOp Sinh -> inplaceFirstSrcReq ()
        | UUnaryOp Cosh -> inplaceFirstSrcReq ()
        | UUnaryOp Tanh -> inplaceFirstSrcReq ()
        | UUnaryOp Sqrt -> inplaceFirstSrcReq ()
        | UUnaryOp Ceil -> inplaceFirstSrcReq ()
        | UUnaryOp Floor -> inplaceFirstSrcReq ()
        | UUnaryOp Round -> inplaceFirstSrcReq ()
        | UUnaryOp Truncate -> inplaceFirstSrcReq ()

        // unary element-wise logic      
        | UUnaryOp Not -> inplaceFirstSrcReq ()

        // tensor ops
        | UUnaryOp (Diag _) -> dfltSrcWithNoViewReq ()
        | UUnaryOp (DiagMat _) -> dfltSrcWithNoViewReq ()
        | UUnaryOp Invert -> dfltSrcWithNoViewReq () 
                
        // reductions
        | UUnaryOp Sum -> dfltSrcWithNoViewReq ()
        | UUnaryOp (SumAxis _) -> dfltSrcWithNoViewReq ()
        | UUnaryOp (MaxAxis _) -> dfltSrcWithNoViewReq ()
        | UUnaryOp (MinAxis _) -> dfltSrcWithNoViewReq ()

        // index reductions
        | UUnaryOp (ArgMaxAxis _) -> dfltSrcWithNoViewReq ()
        | UUnaryOp (ArgMinAxis _) -> dfltSrcWithNoViewReq ()

        // shape operations
        | UUnaryOp (Reshape _) ->        
            match trgtDfltChReq () with
            | Some rv when ArrayND.isC rv ->
                [dfltChReq (Some (ArrayND.reshapeView srcShapes.[0].[dfltChId] rv))]
            | _ -> dfltSrcWithNoViewReq ()
        | UUnaryOp (DoBroadcast _) -> dfltSrcWithNoViewReq ()
        | UUnaryOp (PermuteAxes perm) ->
            match trgtDfltChReq () with
            | Some rv -> [dfltChReq (Some (ArrayND.permuteAxes (Permutation.invert perm) rv))]
            | _ -> dfltSrcWithNoViewReq ()
        | UUnaryOp (ReverseAxis ax) ->
            match trgtDfltChReq () with
            | Some rv -> [dfltChReq (Some (ArrayND.reverseAxis ax rv))]
            | _ -> dfltSrcWithNoViewReq ()

        // variable access
        | UUnaryOp (StoreToVar vs) ->
            match cudaEnv.VarStorLoc |> Map.find vs with
            | LocDev -> 
                // request to store directly into external var
                let shp = vs.Shape |> ShapeSpec.eval
                let stride = cudaEnv |> CudaCompileEnv.strideForVar vs
                [dfltChReq (Some (ArrayNDManikin.external (MemExternal vs) shp stride))]
            | LocHost -> dfltSrcWithNoViewReq ()
            | loc -> unsupLoc loc

        // misc
        | UUnaryOp (Print _) -> inplaceFirstSrcReq ()
        | UUnaryOp (Dump _) -> inplaceFirstSrcReq ()
        | UUnaryOp (Annotated _) -> inplaceFirstSrcReq ()
        | UUnaryOp (CheckFinite _) -> inplaceFirstSrcReq ()

        // binary element-wise
        | UBinaryOp Add -> inplaceFirstSrcReq ()
        | UBinaryOp Substract -> inplaceFirstSrcReq ()
        | UBinaryOp Multiply -> inplaceFirstSrcReq ()
        | UBinaryOp Divide -> inplaceFirstSrcReq ()
        | UBinaryOp Modulo -> inplaceFirstSrcReq ()
        | UBinaryOp Power -> inplaceFirstSrcReq ()
        | UBinaryOp MaxElemwise -> inplaceFirstSrcReq ()
        | UBinaryOp MinElemwise -> inplaceFirstSrcReq ()

        // binary element-wise comparison
        | UBinaryOp Equal -> dfltSrcWithNoViewReq ()
        | UBinaryOp Less -> dfltSrcWithNoViewReq ()
        | UBinaryOp LessEqual -> dfltSrcWithNoViewReq ()
        | UBinaryOp Greater -> dfltSrcWithNoViewReq ()
        | UBinaryOp GreaterEqual -> dfltSrcWithNoViewReq ()   
        | UBinaryOp NotEqual -> dfltSrcWithNoViewReq ()  

        // binary elment-wise logic
        | UBinaryOp And -> inplaceFirstSrcReq ()
        | UBinaryOp Or -> inplaceFirstSrcReq ()

        // matrix/tensor operations
        | UBinaryOp Dot -> dfltSrcWithNoViewReq ()
        | UBinaryOp TensorProduct -> dfltSrcWithNoViewReq ()  

        // nary
        | UNaryOp Discard -> dfltSrcWithNoViewReq ()
        | UNaryOp (Interpolate _) -> inplaceFirstSrcReq ()

        // extra
        | UUnaryOp (Expr.Held _) -> needExtra op

        | UNaryOp (Expr.Channel _) -> needExtra op
        | UExtraOp (Channel ch) -> [Map [ch, trgtDfltChReq()]]

        | UExtraOp (Loop _) -> dfltSrcWithNoViewReq ()

        | UUnaryOp (Expr.Subtensor _) -> needExtra op
        | UExtraOp (Subtensor _) -> dfltSrcWithNoViewReq ()

        | UBinaryOp (Expr.SetSubtensor _) -> needExtra op
        | UExtraOp (SetSubtensor _) -> 
            // "a" can be evaluated into requested manikin if it is not broadcasted, 
            // but "b" (the replacement value) must be placed
            // in a temporary manikin and copied over to avoid race conditions.
            match trgtDfltChReq () with
            | Some req when not (ArrayND.isBroadcasted req) -> inplaceFirstSrcReq ()
            | _ -> dfltSrcWithNoViewReq ()            

        | UNaryOp (Expr.Elements _) -> needExtra op
        | UExtraOp (Elements _) -> dfltSrcWithNoViewReq ()           

        | UBinaryOp (Expr.IfThenElse _) -> needExtra op
        | UExtraOp IfThenElse -> inplaceFirstSrcReq ()

        | UUnaryOp (Expr.Gather _) -> needExtra op
        | UExtraOp (Gather idxArgs) -> dfltSrcWithNoViewReq ()

        | UUnaryOp (Expr.Scatter _) -> needExtra op
        | UExtraOp (Scatter idxArgs) -> dfltSrcWithNoViewReq ()
            
        | UUnaryOp (Expr.NullifyJacobian) -> needExtra op
        | UUnaryOp (Expr.AssumeJacobian _) -> needExtra op

        // extension ops
        | UNaryOp (ExtensionOp eop) -> (toCudaUOp eop).SrcReqs cudaEnv args helpers
        | UExtraOp (ExtensionExtraOp eop) -> (toCudaUOp eop).SrcReqs cudaEnv args helpers


    /// computes the definitive target view of an op given its source views
    let trgtGivenSrcs compileEnv ({MemAllocator=memAllocator
                                   TargetRequest=reqChViews
                                   Op=op
                                   Metadata={ChannelType=trgtTypenames
                                             ChannelShape=trgtShapes
                                             Expr=expr}
                                   Srcs=srcs} as args) =

        /// Default channel shape of target.
        let trgtDfltChShape () = trgtShapes.[dfltChId]
        /// Default channel type of target.
        let trgtDfltChType () = trgtTypenames.[dfltChId]

        /// Default channels of all sources.
        let srcsDfltCh () = srcs |> List.map (fun srcChs -> fst srcChs.[dfltChId])
        /// Default channel shared of all sources.
        let srcsDfltChShared () = srcs |> List.map (fun srcChs -> snd srcChs.[dfltChId])

        /// Default channel of first source.
        let firstSrcDfltCh () = (srcsDfltCh()).[0]
        /// Default channel shared of first source.
        let firstSrcDfltChShared () = (srcsDfltChShared()).[0]

        /// The view request for the default channel of the target.
        let trgtDefChReq () = reqChViews.[dfltChId]

        /// Target for default channel.
        let dfltChTrgt view shared : ChannelManikinsAndSharedT =
            Map [dfltChId, (view, shared)] 

        // New allocated target for default channel.
        let newDfltChTrgt () = 
            dfltChTrgt (ArrayNDManikin.newC memAllocator 
                            (trgtDfltChType()) (trgtDfltChShape())) false        

        /// True if specified manikin overlaps with any channel of any source.
        let overlappingWithAnySrc (rv: ArrayNDManikinT) =
            srcs
            |> List.exists (Map.exists (fun ch (view, shared) -> 
                                            ArrayNDManikin.maybeOverlapping rv view))

        /// default channel target that shares no elements with any srcView 
        let dfltChOutplaceTrgt () =
            match trgtDefChReq () with
            | Some rv when not (overlappingWithAnySrc rv) && 
                           not (ArrayND.isBroadcasted rv) &&
                           rv.TypeName = trgtTypenames.[dfltChId] &&
                           rv.Shape = trgtShapes.[dfltChId] 
                -> dfltChTrgt rv false
            | _ -> newDfltChTrgt () 
             
        /// default channel target that shares no elements with any srcView and can be used for BLAS
        let dfltChOutplaceBlasTrgt () = 
            match trgtDefChReq () with
            | Some rv when ArrayNDManikin.canBeBlasTarget rv && 
                           not (overlappingWithAnySrc rv) &&
                           rv.TypeName = trgtTypenames.[dfltChId] &&
                           rv.Shape = trgtShapes.[dfltChId] 
                -> dfltChTrgt rv false
            | _ -> 
                dfltChTrgt (ArrayNDManikin.newBlasTarget memAllocator 
                                (trgtDfltChType()) (trgtDfltChShape())) false

        /// default channel target that shares no elements with any srcView and the transpose of which can be used for BLAS
        let dfltChOutplaceTransposedBlasTrgt () = 
            match trgtDefChReq () with
            | Some rv when ArrayNDManikin.canBeBlasTarget rv.T && 
                           not (overlappingWithAnySrc rv) &&
                           rv.TypeName = trgtTypenames.[dfltChId] &&
                           rv.Shape = trgtShapes.[dfltChId]
                -> dfltChTrgt rv false
            | _ -> 
                dfltChTrgt (ArrayNDManikin.newC memAllocator 
                                (trgtDfltChType()) (trgtDfltChShape())) false  

        /// Default channel target that reuses the default channel of a srcView, 
        /// if it may be overwritten. Otherwise uses defaultChOutplaceTrgt.
        let dfltChInplaceOvrwrtTrgt () : ChannelManikinsAndSharedT =
            match srcs 
                  |> List.tryFind (fun srcChs ->
                                    let view, shared = srcChs.[dfltChId] 
                                    view.TypeName = trgtTypenames.[dfltChId] &&
                                    view.Shape = trgtShapes.[dfltChId] &&
                                    not (ArrayND.isBroadcasted view) && 
                                    not shared) with
            | Some srcChs -> Map [dfltChId, srcChs.[dfltChId]]
            | None -> dfltChOutplaceTrgt ()     

        let helpers = {
            SrcsDfltCh                          = srcsDfltCh
            SrcsDfltChShared                    = srcsDfltChShared
            TrgtDefChReq                        = trgtDefChReq
            DfltChTrgt                          = dfltChTrgt
            NewDfltChTrgt                       = newDfltChTrgt
            DfltChOutplaceTrgt                  = dfltChOutplaceTrgt
            DfltChOutplaceBlasTrgt              = dfltChOutplaceBlasTrgt
            DfltChOutplaceTransposedBlasTrgt    = dfltChOutplaceTransposedBlasTrgt
            DfltChInplaceOvrwrtTrgt             = dfltChInplaceOvrwrtTrgt
        }

        match op with
        // variable access
        | ULeafOp (Var vs) ->       
           match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev ->
                // create manikin for external variable
                let stride = compileEnv |> CudaCompileEnv.strideForVar vs
                dfltChTrgt (ArrayNDManikin.external (MemExternal vs) vs.NShape stride) true
            | LocHost ->
                // check that host variable has C-stride
                let hvStride = compileEnv |> CudaCompileEnv.strideForVar vs
                let hvLayout = {Shape=vs.NShape; Stride=hvStride; Offset=0L}
                if not (ArrayNDLayout.isC hvLayout) then
                    failwithf "host variable %A must be in C-order" vs

                // We will transfer variable from host to device during execution.
                // We allocate contiguous device memory for that.
                match trgtDefChReq () with
                | Some rv when ArrayND.isC rv -> dfltChTrgt rv false
                | _ -> 
                    dfltChTrgt (ArrayNDManikin.newC memAllocator vs.TypeName vs.NShape) false    
            | loc -> unsupLoc loc     
                           
        // tensor creation
        | ULeafOp _ -> dfltChOutplaceTrgt ()      

        // unary element-wise
        | UUnaryOp Negate -> dfltChInplaceOvrwrtTrgt ()                       
        | UUnaryOp Abs -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp SignT -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Log -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Log10 -> dfltChInplaceOvrwrtTrgt ()                          
        | UUnaryOp Exp -> dfltChInplaceOvrwrtTrgt ()                           
        | UUnaryOp Sin -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Cos -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Tan -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Asin -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Acos -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Atan -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Sinh -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Cosh -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Tanh -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Sqrt -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Ceil -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Floor -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Round -> dfltChInplaceOvrwrtTrgt ()
        | UUnaryOp Truncate -> dfltChInplaceOvrwrtTrgt ()   

        // unary element-wise logic      
        | UUnaryOp Not -> dfltChInplaceOvrwrtTrgt ()   

        // tensor ops
        | UUnaryOp (Diag (ax1, ax2)) ->
            dfltChTrgt (ArrayND.diagAxis ax1 ax2 (firstSrcDfltCh())) (firstSrcDfltChShared())
        | UUnaryOp (DiagMat (ax1, ax2)) -> dfltChOutplaceTrgt ()
        | UUnaryOp Invert -> 
            // If source will be transposed, then target will also be transposed.
            // Thus, in this case, we must request an array the transpose of which 
            // can be used as a BLAS target.
            match blasArgOperation (firstSrcDfltCh()) (firstSrcDfltChShared()) true with
            | BlasArgTranspose -> dfltChOutplaceBlasTrgt ()
            | _ -> dfltChOutplaceTransposedBlasTrgt ()

        // reductions
        | UUnaryOp Sum -> dfltChOutplaceTrgt ()
        | UUnaryOp (SumAxis _) -> dfltChOutplaceTrgt ()
        | UUnaryOp (MaxAxis _) -> dfltChOutplaceTrgt ()
        | UUnaryOp (MinAxis _) -> dfltChOutplaceTrgt ()

        // index reductions
        | UUnaryOp (ArgMaxAxis _) -> dfltChOutplaceTrgt ()
        | UUnaryOp (ArgMinAxis _) -> dfltChOutplaceTrgt ()

        // shape operations
        | UUnaryOp (Reshape _) ->        
            // TODO: optimize: check if copy is really necessary
            if ArrayND.isC (firstSrcDfltCh()) then
                dfltChTrgt (ArrayND.reshapeView (trgtDfltChShape()) (firstSrcDfltCh())) (firstSrcDfltChShared()) 
            else dfltChOutplaceTrgt () // will copy
        | UUnaryOp (DoBroadcast _) ->
            dfltChTrgt (ArrayND.broadcastToShape (trgtDfltChShape()) (firstSrcDfltCh())) (firstSrcDfltChShared())
        | UUnaryOp (PermuteAxes perm) ->
            dfltChTrgt (ArrayND.permuteAxes perm (firstSrcDfltCh())) (firstSrcDfltChShared())
        | UUnaryOp (ReverseAxis ax) ->
            dfltChTrgt (ArrayND.reverseAxis ax (firstSrcDfltCh())) (firstSrcDfltChShared())

        // variable access
        | UUnaryOp (StoreToVar _) -> 
            // output of StoreToVar is empty 
            newDfltChTrgt ()

        // misc
        | UUnaryOp (Print _) -> dfltChTrgt (firstSrcDfltCh()) (firstSrcDfltChShared())
        | UUnaryOp (Dump _) -> dfltChTrgt (firstSrcDfltCh()) (firstSrcDfltChShared())
        | UUnaryOp (Annotated _) -> dfltChTrgt (firstSrcDfltCh()) (firstSrcDfltChShared())
        | UUnaryOp (CheckFinite _) -> dfltChTrgt (firstSrcDfltCh()) (firstSrcDfltChShared())

        // binary element-wise
        | UBinaryOp Add -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp Substract -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp Multiply -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp Divide -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp Modulo -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp Power -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp MaxElemwise -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp MinElemwise -> dfltChInplaceOvrwrtTrgt ()

        // binary element-wise comparison
        | UBinaryOp Equal -> dfltChOutplaceTrgt ()
        | UBinaryOp Less -> dfltChOutplaceTrgt ()
        | UBinaryOp LessEqual -> dfltChOutplaceTrgt ()
        | UBinaryOp Greater -> dfltChOutplaceTrgt ()
        | UBinaryOp GreaterEqual -> dfltChOutplaceTrgt ()   
        | UBinaryOp NotEqual -> dfltChOutplaceTrgt ()   

        // binary elment-wise logic
        | UBinaryOp And -> dfltChInplaceOvrwrtTrgt ()
        | UBinaryOp Or -> dfltChInplaceOvrwrtTrgt ()

        // matrix/tensor operations
        | UBinaryOp Dot -> dfltChOutplaceBlasTrgt ()
        | UBinaryOp TensorProduct -> dfltChOutplaceTrgt ()

        // nary
        | UNaryOp Discard -> dfltChOutplaceTrgt ()
        | UNaryOp (Interpolate _) -> dfltChInplaceOvrwrtTrgt ()  
        
        // extra
        | UUnaryOp (Expr.Held _) -> needExtra op

        | UNaryOp (Expr.Channel _) -> needExtra op
        | UExtraOp (Channel channel) ->
            if srcs.Length <> 1 then 
                failwith "channel op requires exactly one source"     
            if not (srcs.[0].ContainsKey channel) then
                failwithf "channel %s does not exist in %A" channel expr
            let srcManikin, srcShared = srcs.[0].[channel]
            dfltChTrgt srcManikin srcShared

        | UExtraOp (Loop loopspec) ->      
            // Create targets with strides so that the slice dimension is the slowest, thus
            // the loop length does not affect the stride.
            trgtShapes
            |> Map.map (fun ch shp ->
                let sliceDim = loopspec.Channels.[ch].SliceDim
                let strideOrder = [0 .. shp.Length-1] |> List.swap 0 sliceDim |> List.rev
                ArrayNDManikin.newOrdered memAllocator trgtTypenames.[ch] shp strideOrder, false)

        | UUnaryOp (Expr.Subtensor _) -> needExtra op
        | UExtraOp (Subtensor srs) -> 
            if SimpleRangesSpec.isDynamic srs then 
                // dynamic sub-tensors will be copied out of the src
                dfltChOutplaceTrgt ()
            else
                // symbolic sub-tensors use a view of the src 
                let rng = SimpleRangesSpec.eval (fun _ -> failwith "must be static") srs
                dfltChTrgt ((firstSrcDfltCh()).[rng] :?> ArrayNDManikinT) (firstSrcDfltChShared())

        | UBinaryOp (Expr.SetSubtensor _) -> needExtra op
        | UExtraOp (SetSubtensor _) ->
            if not (firstSrcDfltChShared()) && not (ArrayND.isBroadcasted (firstSrcDfltCh())) then 
                dfltChTrgt (firstSrcDfltCh()) false
            else dfltChOutplaceTrgt ()

        | UNaryOp (Expr.Elements _) -> needExtra op
        | UExtraOp (Elements _) -> dfltChOutplaceTrgt ()

        | UBinaryOp (Expr.IfThenElse _) -> needExtra op
        | UExtraOp IfThenElse ->  dfltChInplaceOvrwrtTrgt ()  

        | UUnaryOp (Expr.NullifyJacobian) -> needExtra op
        | UUnaryOp (Expr.AssumeJacobian _) -> needExtra op

        | UUnaryOp (Expr.Gather _) -> needExtra op
        | UExtraOp (Gather idxArgs) -> dfltChOutplaceTrgt ()

        | UUnaryOp (Expr.Scatter _) -> needExtra op
        | UExtraOp (Scatter idxArgs) -> dfltChOutplaceTrgt ()

        // extension        
        | UNaryOp (ExtensionOp eop) -> 
            (toCudaUOp eop).TrgtGivenSrcs compileEnv args helpers
        | UExtraOp (ExtensionExtraOp eop) -> 
            (toCudaUOp eop).TrgtGivenSrcs compileEnv args helpers

   
    /// execution item to launch the given kernel template function
    let execItemsForKernel cppFuncName tmplTmpls argTmpls workDim = 
        let cFuncTmpl = {
            FuncName=cppFuncName
            Domain=KernelFunc
            TmplArgs=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) tmplTmpls
            RetType="void"
            ArgTypes=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) argTmpls
        }    
        [LaunchKernel (cFuncTmpl, workDim, argTmpls)]

    /// returns the CUDA work dimensions (x, y, z) for an element-wise or elements operation
    let workDimForElemwise trgt hetero =
        match ArrayND.nDims trgt with
        | _ when hetero -> (ArrayND.nElems trgt, 1L, 1L)
        | 0 -> (1L, 1L, 1L)
        | 1 -> ((ArrayND.shape trgt).[0], 1L, 1L)
        | 2 -> ((ArrayND.shape trgt).[1], (ArrayND.shape trgt).[0], 1L)
        | 3 -> ((ArrayND.shape trgt).[2], (ArrayND.shape trgt).[1], (ArrayND.shape trgt).[0])
        | d ->
            let rest = {0 .. d-3} |> Seq.map (fun i -> (ArrayND.shape trgt).[i]) |> Seq.fold (*) 1L 
            ((ArrayND.shape trgt).[d-1], (ArrayND.shape trgt).[d-2], rest)

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

    /// execution items for a gather operation
    let execItemsForGather trgt src idxViews =
        let funcName = sprintf "gather%dDTo%dD" (ArrayND.nDims src) (ArrayND.nDims trgt)
        let args = 
            ((ArrayNDArgTmpl trgt) :> ICudaArgTmpl) ::
            ((ArrayNDArgTmpl src) :> ICudaArgTmpl) ::
            (List.map (function | Some v -> ArrayNDArgTmpl v :> ICudaArgTmpl
                                | None   -> ArrayNDNullArgTmpl (TypeName.ofType<int>, trgt.Shape) 
                                            :> ICudaArgTmpl) idxViews)
        execItemsForKernel funcName args args (workDimForElemwise trgt false)

    /// execution items for a scatter operation
    let execItemsForScatter trgt src idxViews =
        let funcName = sprintf "scatter%dDTo%dD" (ArrayND.nDims src) (ArrayND.nDims trgt)
        let args = 
            ((ArrayNDArgTmpl trgt) :> ICudaArgTmpl) ::
            ((ArrayNDArgTmpl src) :> ICudaArgTmpl) ::
            (List.map (function | Some v -> ArrayNDArgTmpl v :> ICudaArgTmpl
                                | None   -> ArrayNDNullArgTmpl (TypeName.ofType<int>, src.Shape) 
                                            :> ICudaArgTmpl) idxViews)
        execItemsForKernel funcName args args (workDimForElemwise src false)

    /// function name of reduction wrapper and its arguments for the given target, operation, initial value and source
    let reductionFuncnameAndArgs trgt indexed cOp cInitialOp src =
        let args = [cOp :> ICudaArgTmpl
                    cInitialOp :> ICudaArgTmpl
                    ArrayNDArgTmpl trgt :> ICudaArgTmpl
                    ArrayNDArgTmpl src :> ICudaArgTmpl]
        let idxStr = if indexed then "Idx" else ""
        let funcName = sprintf "reduce%sTo%dD" idxStr (ArrayND.nDims trgt)
        funcName, args

    /// execution items for a reduction operation
    let execItemsForReduction trgt indexed cOp cInitialOp src =
        match ArrayND.shape trgt, ArrayND.shape src with
        | _, [] -> failwith "cannot reduce a scalar array"
        | trgtShp, srcShp when trgtShp.Length <> srcShp.Length - 1  ->
            failwithf "cannot reduce from %d dimensions to %d dimensions" srcShp.Length trgtShp.Length
        | trgtShp, srcShp when trgtShp <> srcShp.[0 .. srcShp.Length-2] ->
            failwithf "cannot reduce from shape %A to shape %A" srcShp trgtShp 
        | _ -> ()

        let funcName, args = reductionFuncnameAndArgs trgt indexed cOp cInitialOp src
        execItemsForKernel funcName args args (workDimForElemwise trgt false)

    /// function name of elements wrapper and its arguments for the given target, operation and sources
    let elementsFuncnameAndArgs trgt cOp srcViews =
        let args = 
            (cOp :> ICudaArgTmpl) ::
            ((ArrayNDArgTmpl trgt) :> ICudaArgTmpl) ::
            (List.map (fun v -> (ArrayNDArgTmpl v) :> ICudaArgTmpl) srcViews)

        let nSrc = List.length srcViews
        let dimsStr = sprintf "%dD" (ArrayND.nDims trgt)
        let funcName = sprintf "elements%dAry%s" nSrc dimsStr 
        funcName, args

    /// execution items for an element-wise operation
    let execItemsForElements compileEnv trgt elemFunc srcViews =
        let opName = 
            match compileEnv.ElemFuncsOpNames |> Map.tryFind elemFunc with
            | Some opName -> opName
            | None ->
                let id = compileEnv.ElemFuncsOpNames |> Map.toSeq |> Seq.length
                let opName = sprintf "ElemFunc%dOp" id
                compileEnv.ElemFuncsOpNames <- compileEnv.ElemFuncsOpNames |> Map.add elemFunc opName
                opName
        let opTmplArgs = 
            srcViews
            |> List.map (fun (manikin: ArrayNDManikinT) -> manikin.CPPType)
            |> String.concat ", "       
        let opTypeName = 
            if opTmplArgs = "" then opName
            else sprintf "%s<%s>" opName opTmplArgs

        let funcName, args = elementsFuncnameAndArgs trgt (ElementsOpArgTmpl opTypeName) srcViews
        let workDims = workDimForElemwise trgt false
        execItemsForKernel funcName args args workDims

    let dynamicSubtensorTmplAndIdx (bas: ArrayNDManikinT) (rngs: UExprRngsSpecT) (rngManikins: ArrayNDManikinT list) =
        // Apply symbolic ranges to src, and leave dynamic axes unharmed.
        // (0 is added to offset and their size is changed appropriately)
        let basStatic = bas.[SimpleRangesSpec.eval (fun _ -> 0L) rngs] :?> ArrayNDManikinT

        // convert simplified range specification to array of pointers to expressions calculating
        // the indices
        let rec rngToIdxPntrs rngs rngManikins =
            match rngs, rngManikins with
            | SRSDynStartSymSize _ :: rrngs, rngManikin :: rrngManikins ->
                // for dynamic range pass pointer to result of expression calculating the index
                (IdxTPtrFromArrayNDIdxTmpl (Some rngManikin) :> ICudaArrayMemberArgTmpl<IntPtr>) :: 
                    rngToIdxPntrs rrngs rrngManikins 
            | SRSSymStartSymEnd _ :: rrngs, _ ->
                // symbolic range has already been applied, pass null (meaning no offset to add)
                (IdxTPtrFromArrayNDIdxTmpl None :> ICudaArrayMemberArgTmpl<IntPtr>) :: 
                    rngToIdxPntrs rrngs rngManikins 
            | [], [] -> []
            | _ -> failwith "invalid dynamic range specification"
        let basIdxPntrs = rngToIdxPntrs rngs rngManikins

        // C++ parameters
        ArrayNDArgTmpl basStatic, ArrayNDSDArgTmpl basStatic, CPPArrayTmpl basIdxPntrs

    let execItemsForCopyFromDynamicSubtensor trgt src rngs rngManikins =
        // C++ signature is:
        //template <typename TTarget, typename TBaseSrc, typename TDynSrc, idx_t nDims,
        //          TElemwise1Ary<IdEOp_t, TTarget, TDynSrc>::type copyFun>
        //_dev void copyFromDynamicSubtensor(TTarget &trgt,  
        //                                   const TBaseSrc &baseSrc, const Array<idx_t, nDims> &srcIdx)

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

        let cFuncTmpl = {
            FuncName=cppFuncName
            Domain=CPPFunc
            TmplArgs=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) tmplTmpls
            RetType="void"
            ArgTypes=List.map (fun (a: ICudaArgTmpl) -> a.CPPTypeName) argTmpls
        }    
        [CallCFunc(cFuncTmpl, cDelegateType, argTmpls)]


    /// generates ExecItems to copy srcView to trgtView 
    let copyExecItems trgt src =
        if ArrayND.nElems trgt <> ArrayND.nElems src then
            failwithf "cannot copy array with %d elements to array with %d elements"
                (ArrayND.nElems trgt) (ArrayND.nElems src)
        execItemsForElemwise trgt (NoArgEOpArgTmpl("IdEOp_t", false)) [src]

    /// Generates ExecItems to copy srcView into newly allocated memory in C-order.
    /// Broadcasted dimensions of srcView for which broadcastAllowed is true are kept broadcasted.
    let copyKeepingBroadcasted memAllocator (broadcastAllowed: bool list) (src: ArrayNDManikinT) =
        assert (broadcastAllowed.Length = src.NDims)
        let isBroadcasted = 
            List.zip src.Layout.Shape src.Layout.Stride
            |> List.map (fun (size, str) -> size > 1L && str = 0L)
        let tmpShp, srcRngs = 
            List.zip3 src.Shape isBroadcasted broadcastAllowed
            |> List.map (fun (size, isB, allowed) -> 
                            match size, isB, allowed with
                            | _, false, _ -> size, RngAll                    // not broadcasted
                            | _, true, true -> 1L, Rng (Some 0L, Some 0L)    // keep broadcasted
                            | _, true, false -> size, RngAll)                // unbroadcast by copying
            |> List.unzip
        let srcView = src.[srcRngs] :?> ArrayNDManikinT
        let tmpView = ArrayNDManikin.newC memAllocator src.TypeName tmpShp
        let copyOps = copyExecItems tmpView srcView
        let dstView = tmpView |> ArrayND.broadcastToShape src.Shape
        dstView, copyOps                             

    /// If all batch dimensions (all dimensions but the last two) of the array are of
    /// size one, a view of the last two dimensions is returned.
    /// Otherwise the original array is returned.
    let trimUnitaryBatchedBlasDims (manikin: ArrayNDManikinT) =
        let nd = manikin.NDims
        if nd > 2 then
            let isUnitary = manikin.Shape.[0..nd-3] |> List.forall ((=) 1L)
            if isUnitary then
                let mutable m = manikin
                for i=0 to nd-3 do m <- ArrayND.cutLeft m
                m
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
            let bcAllowed = (List.replicate (manikin.NDims - 2) true) @ [false; false]
            let tmpView, copyOps = copyKeepingBroadcasted memAllocator bcAllowed manikin
            tmpView, BlasTranspose, copyOps, false

    /// BLAS target argument passing, so that orientation is preserved
    let blasTarget (manikin: ArrayNDManikinT) =
        let manikin = trimUnitaryBatchedBlasDims manikin
        if not (ArrayNDManikin.canBeBlasTarget manikin) then
            failwithf "cannot use specified view with shape %A and stride %A as BLAS target" 
                manikin.Shape (ArrayND.stride manikin)
        ArrayND.transpose manikin

    /// exection items to reduce src over the last axis into trgt
    let rec batchReduceLastAxis (memAllocator: MemAllocatorT) reduceFn (trgt: ArrayNDManikinT) (src: ArrayNDManikinT) 
            : CudaExecItemT list =
        let reduceBatchSize = 16L
        let nReduceElems = src.Shape.[src.NDims - 1]

        if nReduceElems <= reduceBatchSize then
            // no split necessary
            reduceFn trgt src
        else
            // split last dimension
            let reduceBatches = nReduceElems / reduceBatchSize
            let reduceRem = nReduceElems - reduceBatches * reduceBatchSize

            // create array manikin for source with split last dimension
            let batchSrcShp = 
                (ArrayND.shape src).[0 .. src.NDims-2] @ [reduceBatches; reduceBatchSize]
            let reduceStride = (ArrayND.stride src).[src.NDims-1]
            let batchSrcStride = 
                (ArrayND.stride src).[0 .. src.NDims-2] @ [reduceStride * reduceBatchSize; reduceStride]
            let batchSrc = 
                src |> ArrayND.relayout {src.Layout with Shape=batchSrcShp; Stride=batchSrcStride}

            // create temporary target
            let tmpShp = 
                if reduceRem = 0L then trgt.Shape @ [reduceBatches]
                else trgt.Shape @ [reduceBatches + 1L]
            let tmpTrgt = ArrayNDManikin.newC memAllocator trgt.TypeName tmpShp

            // perform reduction of batch
            let batchTrgt = tmpTrgt.[Fill, 0L .. reduceBatches-1L] :?> ArrayNDManikinT
            let batchExecItems = reduceFn batchTrgt batchSrc

            // perform reduction of remaining elements, if necessary
            let remExecItems =
                if reduceRem = 0L then []
                else
                    let remSrc = src.[Fill, reduceBatches*reduceBatchSize ..] :?> ArrayNDManikinT
                    let remTrgt = tmpTrgt.[Fill, reduceBatches] :?> ArrayNDManikinT
                    reduceFn remTrgt remSrc

            // recursively reduce temporary target
            let recExecItems = batchReduceLastAxis memAllocator reduceFn trgt tmpTrgt

            batchExecItems @ remExecItems @ recExecItems

    /// reduce one axis by appling an operation such as sum, max, min, ...
    let execItemsForReduceAxis memAllocator ax eOpName initial (trgt: ArrayNDManikinT) (src: ArrayNDManikinT) =
        // we need to swap axes so that the axes the reduction is performed over comes last
        let nd = ArrayND.nDims src
        let axOrder = Seq.concat [{0 .. ax-1}; {nd-1 .. nd-1}; {ax .. nd-2}] |> Seq.toList
        let srcAdj = ArrayND.permuteAxes axOrder src
        (trgt, srcAdj) ||> batchReduceLastAxis memAllocator (fun tmpTrgt tmpSrc ->
            execItemsForReduction tmpTrgt false (NoArgEOpArgTmpl(eOpName, false)) (ConstEOpArgTmpl initial) tmpSrc)

    /// reduce one axis by appling an operation on indices such as argMax, argMin, ...
    let execItemsForIdxReduceAxis memAllocator ax eOpName initial (trgt: ArrayNDManikinT) (src: ArrayNDManikinT) =
        // we need to swap axes so that the axes the reduction is performed over comes last
        let nd = ArrayND.nDims src
        let axOrder = Seq.concat [{0 .. ax-1}; {nd-1 .. nd-1}; {ax .. nd-2}] |> Seq.toList
        let srcAdj = ArrayND.permuteAxes axOrder src
        execItemsForReduction trgt true (NoArgEOpArgTmpl(eOpName, false)) (ConstEOpArgTmpl initial) srcAdj

    /// exection items to reduce all elements of src into the scalar trgt
    let rec execItemsForReduce memAllocator eOpName initial (trgt: ArrayNDManikinT) (src: ArrayNDManikinT) =
        if ArrayND.nDims trgt <> 0 then failwith "reduce target must be scalar"
        match src.Shape with
        | [_] -> execItemsForReduceAxis memAllocator 0 eOpName initial trgt src
        | [] -> copyExecItems trgt src
        | srcShp ->
            // create temporary target
            let nDims = ArrayND.nDims src
            let tmpShp = srcShp.[0 .. nDims-2]
            let tmp = ArrayNDManikin.newC memAllocator src.TypeName tmpShp            

            // reduce over last axis, and then reduce recursively over remaining axes
            let reduceLastExecItems = execItemsForReduceAxis memAllocator (nDims-1) eOpName initial tmp src
            let reduceOtherExecItems = execItemsForReduce memAllocator eOpName initial trgt tmp
            reduceLastExecItems @ reduceOtherExecItems

    /// returns the execution units for the specified op
    let execItemsForOp compileEnv ({MemAllocator=memAllocator
                                    Target=trgtChs
                                    Op=op
                                    Metadata=metadata
                                    Srcs=srcsAndShared
                                    SubmitInitItems=submitInit} as args) =

        /// Default channel of target.
        let dfltChTrgt () = trgtChs.[dfltChId]
        /// Default channel shape of target.
        let trgtDfltChShape () = metadata.ChannelShape.[dfltChId]
        /// Default channel type of target.
        let trgtDfltChType () = metadata.ChannelType.[dfltChId]

        /// Default channels of all sources.
        let srcsDfltCh () = srcsAndShared |> List.map (fun srcChs -> fst srcChs.[dfltChId])
        /// Default channel shared of all sources.
        let srcsDfltChShared () = srcsAndShared |> List.map (fun srcChs -> snd srcChs.[dfltChId])
    
        /// Default channel of first source.
        let firstSrcDfltCh () = (srcsDfltCh()).[0]
        /// Default channel shared of first source.
        let firstSrcDfltChShared () = (srcsDfltChShared()).[0]

        // set pointer array values either during initialization (for allocated arrays)
        // or runtime (for variable arrays)
        let appendPointerArrayItems (tmpl: BlasTransposedMatrixBatchTmpl) execItems =
            match tmpl.Manikin.Storage with
            | MemZero _
            | MemConst _
            | MemAlloc _ -> submitInit [BlasInitPointerArray tmpl]; execItems
            | MemExternal _ -> execItems @ [BlasInitPointerArray tmpl]

        let helpers = {
            SrcsDfltCh              = srcsDfltCh
            SrcsDfltChShared        = srcsDfltChShared
            DfltChTrgt              = dfltChTrgt
            AppendPointerArrayItems = appendPointerArrayItems
        }

        match op with 
        // tensor creation
        | ULeafOp (Identity _) -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("DiagonalOneIEOp_t", true)) []
        | ULeafOp (ScalarConst cs) -> execItemsForElemwise (dfltChTrgt()) (ConstEOpArgTmpl cs) [] 
        | ULeafOp (SizeValue (sv, _)) -> 
            let value = Convert.ChangeType(SizeSpec.eval sv, TypeName.getType (trgtDfltChType()))
            let cs = ConstSpec.ofValue value
            execItemsForElemwise (dfltChTrgt()) (ConstEOpArgTmpl cs) [] 
        | ULeafOp (Arange _) ->
            execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("CountingIEOp_t", true)) [] 

        // variable access
        | ULeafOp (Var vs) -> 
            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev -> []
            | LocHost -> 
                let hvStride = compileEnv |> CudaCompileEnv.strideForVar vs
                let hv = ArrayNDManikin.external (MemExternal vs) vs.NShape hvStride
                [MemcpyHtoD(ArrayNDHostRegMemRngTmpl(hv), ArrayNDDevMemRngTmpl(dfltChTrgt()))]       
            | loc -> unsupLoc loc

        // unary element-wise
        | UUnaryOp Negate -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("NegateEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Abs -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("AbsEOp_t", false)) (srcsDfltCh())
        | UUnaryOp SignT -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("SignTEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Log -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("LogEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Log10 -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("Log10EOp_t", false)) (srcsDfltCh())
        | UUnaryOp Exp -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("ExpEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Sin -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("SinEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Cos -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("CosEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Tan -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("TanEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Asin -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("AsinEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Acos -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("AcosEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Atan -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("AtanEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Sinh -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("SinhEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Cosh -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("CoshEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Tanh -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("TanhEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Sqrt -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("SqrtEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Ceil -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("CeilEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Floor -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("FloorEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Round -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("RoundEOp_t", false)) (srcsDfltCh())
        | UUnaryOp Truncate -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("TruncateEOp_t", false)) (srcsDfltCh())

        // unary element-wise logic      
        | UUnaryOp Not -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("NotEOp_t", false)) (srcsDfltCh())

        // reductions
        | UUnaryOp Sum -> execItemsForReduce memAllocator "AddEOp_t" (ConstSpec.zero (trgtDfltChType().Type)) (dfltChTrgt()) (firstSrcDfltCh())
        | UUnaryOp (SumAxis ax) -> execItemsForReduceAxis memAllocator ax "AddEOp_t" (ConstSpec.zero (trgtDfltChType().Type)) (dfltChTrgt()) (firstSrcDfltCh())
        | UUnaryOp (MaxAxis ax) -> execItemsForReduceAxis memAllocator ax "MaxEOp_t" (ConstSpec.minValue (trgtDfltChType().Type)) (dfltChTrgt()) (firstSrcDfltCh())
        | UUnaryOp (MinAxis ax) -> execItemsForReduceAxis memAllocator ax "MinEOp_t" (ConstSpec.maxValue (trgtDfltChType().Type)) (dfltChTrgt()) (firstSrcDfltCh())

        // index reductions
        | UUnaryOp (ArgMaxAxis ax) -> execItemsForIdxReduceAxis memAllocator ax "ArgMaxIROp_t" (ConstSpec.minValue (firstSrcDfltCh().DataType)) (dfltChTrgt()) (firstSrcDfltCh())
        | UUnaryOp (ArgMinAxis ax) -> execItemsForIdxReduceAxis memAllocator ax "ArgMinIROp_t" (ConstSpec.maxValue (firstSrcDfltCh().DataType)) (dfltChTrgt()) (firstSrcDfltCh())

        // tensor ops
        | UUnaryOp (Diag _) -> []
        | UUnaryOp (DiagMat (ax1, ax2)) ->
            let trgtDiag = ArrayND.diagAxis ax1 ax2 (dfltChTrgt())
            let zeroItems = execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("ZerosEOp_t", false)) []
            let copyItems = copyExecItems trgtDiag (firstSrcDfltCh())
            zeroItems @ copyItems
        | UUnaryOp Invert ->
            let aView, _, aCopyItems, _ = blasArg memAllocator (firstSrcDfltCh()) (firstSrcDfltChShared()) true

            let tView =
                // If the source is transposed by us then the target must be transposed by us 
                // as well to preserve orientation. The blasTarget function always transposes.
                match blasArgOperation (firstSrcDfltCh()) (firstSrcDfltChShared()) true with
                | BlasArgTranspose -> blasTarget (dfltChTrgt())
                | _ -> blasTarget (ArrayND.transpose (dfltChTrgt()))

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
            if dfltChTrgt() <> firstSrcDfltCh() then 
                copyExecItems (dfltChTrgt()) (firstSrcDfltCh())
            else []
        | UUnaryOp (DoBroadcast _) -> []
        | UUnaryOp (PermuteAxes _) -> []
        | UUnaryOp (ReverseAxis _) -> []

        // variable access
        | UUnaryOp (StoreToVar vs) ->
            match compileEnv.VarStorLoc |> Map.find vs with
            | LocDev when (firstSrcDfltCh()).Storage = (MemExternal vs) ->
                // Source was evaluated directly into the variable storage.
                // No copy necessary.
                []
            | LocDev  -> 
                // Our source has not been evaluated directly into the variable storage.
                // Therefore we need to copy into the variable.
                let varStride = compileEnv |> CudaCompileEnv.strideForVar vs
                let dv = ArrayNDManikin.external (MemExternal vs) vs.NShape varStride
                copyExecItems dv (firstSrcDfltCh())
            | LocHost ->            
                let copyItems, memcpySrc = 
                    if ArrayND.isC (firstSrcDfltCh()) then 
                        // Source has C-strides. Can directly copy to host.
                        [], firstSrcDfltCh()
                    else
                        // Need to copy to temporary C-stride storage first.
                        let tmp = ArrayNDManikin.newC memAllocator vs.TypeName vs.NShape
                        copyExecItems tmp (firstSrcDfltCh()), tmp

                // check that host variable has C-stride
                let hvStride = compileEnv |> CudaCompileEnv.strideForVar vs
                let hvLayout = {Shape=vs.NShape; Stride=hvStride; Offset=0L}
                if not (ArrayNDLayout.isC hvLayout) then
                    failwithf "host variable %A must be in C-order" vs

                // copy
                let hv = ArrayNDManikin.external (MemExternal vs) vs.NShape hvStride
                copyItems @ [MemcpyDtoH(ArrayNDDevMemRngTmpl(memcpySrc), ArrayNDHostRegMemRngTmpl(hv))]   
            | loc -> unsupLoc loc         
                                 
        // misc
        | UUnaryOp (Print msg) -> [PrintWithMsg (msg, firstSrcDfltCh())]
        | UUnaryOp (Dump name) -> [DumpValue (name, firstSrcDfltCh())]
        | UUnaryOp (CheckFinite name) ->
            let nonFiniteCount = ArrayNDManikin.newC memAllocator TypeName.ofType<int> [1L]
            let initItems = [MemsetUInt32 (0u, ArrayNDDevMemRngTmpl nonFiniteCount)]
            let countItems = execItemsForElemwise (dfltChTrgt()) (CheckFiniteIEOpArgTmpl (nonFiniteCount, name)) (srcsDfltCh())
            let checkItems = [CheckNonFiniteCounter (name, nonFiniteCount)]
            initItems @ countItems @ checkItems
        | UUnaryOp (Annotated _) -> []

        // binary element-wise
        | UBinaryOp Add ->         execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("AddEOp_t",       false)) (srcsDfltCh())
        | UBinaryOp Substract ->   execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("SubstractEOp_t", false)) (srcsDfltCh())
        | UBinaryOp Multiply ->    execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("MultiplyEOp_t",  false)) (srcsDfltCh())
        | UBinaryOp Divide ->      execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("DivideEOp_t",    false)) (srcsDfltCh())
        | UBinaryOp Modulo ->      execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("ModuloEOp_t",    false)) (srcsDfltCh())
        | UBinaryOp Power ->       execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("PowerEOp_t",     false)) (srcsDfltCh())
        | UBinaryOp MaxElemwise -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("MaxEOp_t",       false)) (srcsDfltCh())
        | UBinaryOp MinElemwise -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("MinEOp_t",       false)) (srcsDfltCh())

        // binary element-wise comparison
        | UBinaryOp Equal ->        execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("EqualEOp_t",        false)) (srcsDfltCh())
        | UBinaryOp Less ->         execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("LessEOp_t",         false)) (srcsDfltCh())
        | UBinaryOp LessEqual ->    execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("LessEqualEOp_t",    false)) (srcsDfltCh())
        | UBinaryOp Greater ->      execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("GreaterEOp_t",      false)) (srcsDfltCh())
        | UBinaryOp GreaterEqual -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("GreaterEqualEOp_t", false)) (srcsDfltCh())   
        | UBinaryOp NotEqual ->     execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("NotEqualEOp_t",     false)) (srcsDfltCh())   

        // binary elment-wise logic
        | UBinaryOp And -> execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("AndEOp_t", false)) (srcsDfltCh())
        | UBinaryOp Or ->  execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("OrEOp_t",  false)) (srcsDfltCh())

        // matrix/tensor operations
        | UBinaryOp Dot -> 
            let aView, aOp, aCopyItems, aShared = blasArg memAllocator (srcsDfltCh()).[0] (srcsDfltChShared()).[0] false
            let bView, bOp, bCopyItems, bShared = blasArg memAllocator (srcsDfltCh()).[1] (srcsDfltChShared()).[1] false
            let tView = blasTarget (dfltChTrgt())
        
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

        | UBinaryOp TensorProduct -> failwith "TensorProduct is not implemented"

        // nary
        | UNaryOp Discard -> []
        | UNaryOp (Interpolate ip) -> 
            execItemsForElemwise (dfltChTrgt()) (InterpolateEOpArgTmpl (ip, compileEnv)) (srcsDfltCh())

        // extra
        | UUnaryOp (Expr.Held _) -> needExtra op

        | UNaryOp (Expr.Channel _) -> needExtra op
        | UExtraOp (Channel _) -> []

        | UExtraOp (Loop loopSpec) ->
            // build channel infos
            let channelInfos = 
                loopSpec.Channels 
                |> Map.map (fun ch lv ->
                    {
                        LoopEval.Shape       = UExpr.dfltChShape lv.UExpr
                        LoopEval.SliceDim    = lv.SliceDim
                        LoopEval.Target      = trgtChs.[ch] :> IArrayNDT
                    })

            // obtain stride information
            let srcs = srcsDfltCh() |> List.map (fun s -> s :> IArrayNDT)
            let argStrides, chStrides, srcReqStrideOrder = 
                LoopEval.buildStrides loopSpec.Vars srcs channelInfos 

            // copy sources to temporary variable if necessary to match strides
            let copiedSrcs, copyItems =   
                ([], List.zip srcs srcReqStrideOrder)
                ||> List.mapFold (fun copyItems (src, reqOrder) ->
                    let src = src :?> ArrayNDManikinT
                    match reqOrder with
                    | Some order ->
                        let tmp = ArrayNDManikin.newOrdered memAllocator src.TypeName src.Shape order
                        let copyItems = copyItems @ copyExecItems tmp src
                        tmp, copyItems
                    | None -> src, copyItems)

            // create recipe description for sub-workspace that will evaluate one
            // loop iteration
            let recipeDesc = {
                CompileEnv  = {SymSizes       = SymSizeEnv.empty  
                               VarLocs        = loopSpec.Vars |> Map.map (fun _ _ -> LocDev)
                               VarStrides     = argStrides      
                               ChannelStrides = chStrides                        
                               ResultLoc      = LocDev
                               CanDelay       = false}
                UExprs      = loopSpec.Channels |> Map.map (fun ch lv -> lv.UExpr) 
            }

            // emit loop executor
            let execLoopInfo = {
                Length                = loopSpec.Length
                Channels              = channelInfos
                Vars                  = loopSpec.Vars
                Workspace             = compileEnv |> CudaCompileEnv.newSubrecipe recipeDesc
                Args                  = copiedSrcs
                IterManikin           = ArrayNDManikin.newC memAllocator TypeName.ofType<int> []
                ItersRemainingManikin = ArrayNDManikin.newC memAllocator TypeName.ofType<int> []
            }
            copyItems @ [ExecLoop execLoopInfo]

        | UUnaryOp (Expr.Subtensor _) -> needExtra op
        | UExtraOp (Subtensor srs) ->
            if SimpleRangesSpec.isDynamic srs then 
                // copy dynamic subtensor out of the src
                execItemsForCopyFromDynamicSubtensor (dfltChTrgt())
                    (firstSrcDfltCh()) srs (List.tail (srcsDfltCh()))
            else [] // symbolic subtensor uses a slice of the src view

        | UBinaryOp (Expr.SetSubtensor _) -> needExtra op
        | UExtraOp (SetSubtensor srs) ->
            // copy "a" if necessary
            let copyItems = 
                if (dfltChTrgt()) <> (srcsDfltCh()).[0] then 
                    copyExecItems (dfltChTrgt()) (srcsDfltCh()).[0] 
                else []
            // copy "b" into a
            let setItems =
                execItemsForCopyToDynamicSubtensor (dfltChTrgt()) srs 
                    (List.skip 2 (srcsDfltCh())) (srcsDfltCh()).[1]
            copyItems @ setItems

        | UNaryOp (Expr.Elements _) -> needExtra op
        | UExtraOp (Elements (_, elemFunc)) ->
            execItemsForElements compileEnv (dfltChTrgt()) elemFunc (srcsDfltCh())

        | UBinaryOp (Expr.IfThenElse _) -> needExtra op
        | UExtraOp IfThenElse ->  
            execItemsForElemwise (dfltChTrgt()) (NoArgEOpArgTmpl("IfThenElseEOp_t", false)) (srcsDfltCh())

        | UUnaryOp (Expr.NullifyJacobian) -> needExtra op
        | UUnaryOp (Expr.AssumeJacobian _) -> needExtra op

        | UUnaryOp (Expr.Gather _) -> needExtra op
        | UExtraOp (Gather idxArgs) -> 
            let srcs = srcsDfltCh ()
            let idxArgs = idxArgs |> List.map (function | Some n -> Some srcs.[n]
                                                        | None   -> None)
            execItemsForGather (dfltChTrgt()) srcs.[0] idxArgs

        | UUnaryOp (Expr.Scatter _) -> needExtra op
        | UExtraOp (Scatter idxArgs) -> 
            let trgt, srcs = dfltChTrgt(), srcsDfltCh()
            // set target to zero
            let zero = ConstSpec.zero trgt.DataType
            let zeroItems = execItemsForElemwise trgt (ConstEOpArgTmpl zero) []
            // scatter from src into target
            let idxArgs = idxArgs |> List.map (function | Some n -> Some srcs.[n]
                                                        | None   -> None)
            let scatterItems = execItemsForScatter trgt srcs.[0] idxArgs
            zeroItems @ scatterItems

        // extension
        | UNaryOp (ExtensionOp eop) -> 
            (toCudaUOp eop).ExecItems compileEnv args helpers
        | UExtraOp (ExtensionExtraOp eop) -> 
            (toCudaUOp eop).ExecItems compileEnv args helpers

    /// returns the execution units for tracing becore execution of the op items
    let tracePreItemsForExpr compileEnv {TraceItemsForExprArgs.Expr=uexpr} =
        match uexpr with
        | UExpr (UExtraOp (Loop loopSpec), _, _) -> [TraceEnteringLoop uexpr]
        | _ -> []
                
    /// returns the execution units for tracing the result after execution of the op items
    let tracePostItemsForExpr compileEnv {Target=trgtChs; Expr=uexpr} =
        match uexpr with
        | UExpr (UExtraOp (Loop _), _, _) -> [TraceLeavingLoop uexpr]
        | _ ->
            // do tracing of default channel, when available
            match trgtChs.TryFind dfltChId with
            | Some dfltChTrgt -> [Trace (uexpr, dfltChTrgt)]
            | None -> []

    /// generates CUDA execution units that will evaluate the given unified expression
    let exprToCudaExecUnits (compileEnv: CudaCompileEnvT) =
        ExecUnit.exprToExecUnits {
            ExecItemsForOp=execItemsForOp compileEnv
            TracePreItemsForExpr=tracePreItemsForExpr compileEnv
            TracePostItemsForExpr=tracePostItemsForExpr compileEnv
            TrgtGivenSrcs=trgtGivenSrcs compileEnv
            SrcReqs=srcReqs compileEnv
        } 





