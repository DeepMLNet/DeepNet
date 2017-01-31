namespace SymTensor.Compiler.Cuda

open System.Collections.Generic
open System.Diagnostics

open ManagedCuda
open Basics
open Basics.Cuda
open ArrayNDNS
open SymTensor
open SymTensor.Compiler
open UExprTypes


[<AutoOpen>]
module CudaRecipeTypes =

    /// CUDA call flags
    type CudaFlagsT = int

    /// CUDA api call
    type CudaCallT =
        // memory mangement
        | MemAlloc          of MemAllocManikinT
        | MemFree           of MemAllocManikinT
        // stream management
        | StreamCreate      of StreamT * BasicTypes.CUStreamFlags
        | StreamDestory     of StreamT
        | StreamWaitEvent   of StreamT * EventObjectT 
        // event mangement
        | EventCreate       of EventObjectT * BasicTypes.CUEventFlags
        | EventDestory      of EventObjectT
        | EventRecord       of EventObjectT * StreamT
        | EventSynchronize  of EventObjectT
        // texture object management
        | TextureCreate     of TextureObjectT
        | TextureDestroy    of TextureObjectT
        // sub-workspace management
        | SubWorkspaceNew     of SubWorkspaceT
        | SubWorkspaceDispose of SubWorkspaceT
        // execution control
        | LaunchCPPKernel   of TmplInstT * WorkDimT * int * StreamT * (ICudaArgTmpl list)
        | LaunchCKernel     of string * WorkDimT * int * StreamT * (ICudaArgTmpl list)
        | CallCFunc         of string * System.Type * StreamT * (ICudaArgTmpl list)
        // execution item
        | ExecItem          of CudaExecItemT * StreamT

    /// function instantiation state
    type TmplInstCacheT = {
        mutable Insts:          (TmplInstT * string) list
        mutable Code:           (FuncDomainT * string) list
    } 

    /// CUDA execution recipe
    type CudaRecipeT = {
        ChannelVars:        Map<ChannelT, VarSpecT option>
        ChannelAllocators:  Map<ChannelT, unit -> IArrayNDT>
        VarStrides:         VarStridesT
        VarStorLoc:         VarLocsT
        KernelCode:         string
        CPPCode:            string
        InitCalls:          CudaCallT list
        DisposeCalls:       CudaCallT list
        ExecCalls:          CudaCallT list
        ConstantValues:     Map<MemConstManikinT, IArrayNDCudaT>
        SubRecipes:         Map<SubWorkspaceT, CudaRecipeT>
    } 


module TmplInstCache =

    /// gets the generated code for the specified domain
    let getCodeForDomain domain cache =
        cache.Code
        |> List.rev
        |> List.fold (fun code (tiDomain, tiCode) ->
            if tiDomain = domain then code + "\n" + tiCode
            else code) ""

    /// instantiates a template C++ function with a unique C linkage function name and returns the C function name
    let instCPPTmplFunc (ti: TmplInstT) cache =  
        match cache.Insts |> List.tryFind (fun (cti, _) -> cti = ti) with
        | Some (_, cName) -> cName
        | None ->
            // generate C function name
            let nPrv = 
                cache.Insts 
                |> List.filter (fun (oti, _) -> oti.FuncName = ti.FuncName) 
                |> List.length
            let cName = sprintf "%s_%d" ti.FuncName nPrv
            cache.Insts <- (ti, cName)::cache.Insts

            // generate template instantiation with C linkage
            let instStr =
                if List.isEmpty ti.TmplArgs then ti.FuncName
                else sprintf "%s<%s>" ti.FuncName (ti.TmplArgs |> String.concat ", ")
            let krnlStr = match ti.Domain with
                          | KernelFunc -> "__global__"
                          | CPPFunc -> "__declspec(dllexport)"
            let traceFunc = match ti.Domain with
                            | KernelFunc -> "KERNEL_TRACE"
                            | CPPFunc -> "HOST_TRACE"
            let argDeclStr = ti.ArgTypes |> List.mapi (fun i t -> sprintf "%s p%d" t i)  |> String.concat ", "
            let argCallStr = ti.ArgTypes |> List.mapi (fun i _ -> sprintf "p%d" i) |> String.concat ", "
            let retCmd = if ti.RetType.Trim() = "void" then "" else "return"
            let declStr =
                sprintf "extern \"C\" %s %s %s (%s) {\n" krnlStr ti.RetType cName argDeclStr
                + sprintf "  %s(\"%s\");\n" traceFunc cName
                + sprintf "  %s %s (%s);\n" retCmd instStr argCallStr
                + sprintf "}\n"
                + sprintf "\n"
            cache.Code <- (ti.Domain, declStr) :: cache.Code

            cName

    /// instantiates an element calculation functor
    let instElemOp (elemFunc: UElemExpr.UElemFuncT) opName cache =
        let functorCode = CudaElemExpr.generateFunctor opName elemFunc
        cache.Code <- (KernelFunc, functorCode) :: cache.Code



module CudaRecipe =

    let commonIncludes = ["Utils.cuh"; "NDSupport.cuh"; "Subtensor.cuh"; "Ops.cuh"; "Interpolate.cuh"]
    let kernelModuleIncludes = commonIncludes
    let cppModuleIncludes = commonIncludes @ ["ThrustInterface.cuh"; "Reduce.cuh"; "stdio.h"]

    let generateIncludes incls =
        incls
        |> List.map (sprintf "#include \"%s\"\n")
        |> String.concat ""

    let traceHeader = ""

    //let traceHeader =
    //    if Debug.TraceCalls then "#define ENABLE_CALL_TRACE  \n" 
    //    else ""

    /// Header of generated CUDA kernel module
    let kernelModuleHeader = 
        traceHeader +
        generateIncludes kernelModuleIncludes 

    /// Header of generated C++ module
    let cppModuleHeader =
        traceHeader +
        generateIncludes cppModuleIncludes 

    /// gets all CUDA C kernel launches performed 
    let getAllCKernelLaunches recipe = 
        let extract = List.filter (fun c -> 
            match c with 
            | LaunchCKernel _ -> true
            | _ -> false)
        (extract recipe.InitCalls) @ (extract recipe.DisposeCalls) @ (extract recipe.ExecCalls)

    /// gets all C++ function calls
    let getAllCFuncCalls recipe =
        let extract = List.filter (fun c -> 
            match c with 
            | CallCFunc _ -> true
            | _ -> false)
        (extract recipe.InitCalls) @ (extract recipe.DisposeCalls) @ (extract recipe.ExecCalls)

    /// Cuda calls for executing an CudaExecItem in the given stream
    let callsForExecItem cmd cache strm =
        match cmd with
        | LaunchKernel(ti, workDim, args) -> 
            [LaunchCKernel(TmplInstCache.instCPPTmplFunc ti cache, workDim, 0, strm, args)]
        | CudaExecItemT.CallCFunc(ti, dlgte, args) ->
            [CallCFunc(TmplInstCache.instCPPTmplFunc ti cache, dlgte, strm, args)]
        | cmd -> [ExecItem (cmd, strm)]

    type private StreamQueue<'c> = Queue<CudaCmdT<'c>>

    /// generates a sequence of CUDA calls from streams
    let generateCalls (streams: StreamCmdsT<CudaExecItemT> list) cache =    
        let calls = ResizeArray<CudaCallT> ()

        /// the number of times WaitOnEvent is called for a particular correlation
        let correlationIdWaiters =
            seq {
                for strm in streams do
                    for exec in strm do
                        match exec with
                        | WaitOnEvent evt -> yield evt.CorrelationId
                        | _ -> ()
            } |> Seq.countBy id |> Map.ofSeq

        // convert stream lists to queues
        let streams = streams |> List.map (fun strm -> Queue strm)       

        // active events
        let activeEventObjects = Dictionary<EventObjectT, int> ()
        let activeEventCorrelations = Dictionary<int, int> ()

        /// the number of total calls since the last call on a particular stream
        let lastCallOnStream = Dictionary<StreamT, int> ()       
        for strmId = 0 to streams.Length - 1 do
            lastCallOnStream.[strmId] <- 0

        while streams |> Seq.exists (fun s -> s.Count > 0) do
            // Sort streams so that the stream that was called the longest time ago
            // comes first. This ensures that CUDA stream calls are interleaved
            // properly.
            let streamsSorted = 
                streams
                |> Seq.indexed
                |> Seq.sortByDescending (fun (strmId, strm) ->                         
                    // Prioritze emitting of events and penalize synchronization.
                    let syncModifier = 
                        match strm.TryPeek with
                        | Some (EmitEvent _)   | Some (EmitRerunEvent _)   ->  1000
                        | Some (WaitOnEvent _) | Some (WaitOnRerunEvent _) -> -1000
                        | _ -> 0
                    lastCallOnStream.[strmId] + syncModifier)    
                |> List.ofSeq     

            // find stream to process
            let strmIdToProcess, strmToProcess = 
                try
                    streamsSorted 
                    |> List.find (fun (_, strm) ->
                        match strm.TryPeek with
                        | Some (WaitOnEvent evt) ->
                            // WaitOnEvent can only be called when EmitEvent 
                            // with same CorrelationId has been called before.
                            activeEventCorrelations.GetOrDefault evt.CorrelationId 0 > 0
                        | Some (EmitEvent evtp) ->
                            // EmitEvent for a given event must be called
                            // after all necessary calls to WaitOnEvent for a previous correlation.
                            activeEventObjects.GetOrDefault (!evtp).Value.EventObjectId 0 = 0 
                        | Some _ -> true
                        | None -> false)
                with :? System.Collections.Generic.KeyNotFoundException ->
                    // cannot find a stream that matches above rules
                    printfn "Error: deadlock during stream sequencing"
                    printfn "Streams to process:\n%A" streamsSorted
                    printfn "Active event objects:\n%A" activeEventObjects
                    printfn "Active event correlations:\n%A" activeEventCorrelations
                    failwith "deadlock during stream sequencing"

            // generate call for stream
            let execOp = strmToProcess.Dequeue ()
            match execOp with
            | WaitOnEvent evt ->
                // remove active event
                activeEventObjects.[evt.EventObjectId] <- activeEventObjects.[evt.EventObjectId] - 1
                activeEventCorrelations.[evt.CorrelationId] <- activeEventCorrelations.[evt.CorrelationId] - 1                

                calls.Add (StreamWaitEvent (strmIdToProcess, evt.EventObjectId))
            | WaitOnRerunEvent evt ->
                calls.Add (StreamWaitEvent (strmIdToProcess, evt.EventObjectId))
            | EmitEvent evtp ->
                let evt = Option.get !evtp
                // add active event as many times as it will be waited upon
                let nWaiters = correlationIdWaiters.[evt.CorrelationId]
                activeEventObjects.[evt.EventObjectId] <-
                    activeEventObjects.GetOrDefault evt.EventObjectId 0 + nWaiters
                activeEventCorrelations.[evt.CorrelationId] <-
                    activeEventCorrelations.GetOrDefault evt.CorrelationId 0 + nWaiters

                calls.Add (EventRecord (evt.EventObjectId, strmIdToProcess))
            | EmitRerunEvent evtp ->
                let evt = Option.get !evtp
                calls.Add (EventRecord (evt.EventObjectId, strmIdToProcess))
            | Perform cmd ->
                // perform a non-synchronization operation, i.e. some useful work
                for strmId=0 to streams.Length - 1 do
                    lastCallOnStream.[strmId] <-
                        if strmId = strmIdToProcess then 0
                        else lastCallOnStream.[strmId] + 1

                // generate CUDA call template
                let cmdCalls = callsForExecItem cmd cache strmIdToProcess
                calls.AddRange cmdCalls
            | RerunSatisfied _ | ExecUnitStart _ | ExecUnitEnd _ -> 
                // ignore informational markers
                ()

        calls |> List.ofSeq

    /// generates init and dispose calls for CUDA resources
    let generateAllocAndDispose compileEnv memAllocs streamCnt eventObjCnt =
        let memAllocCalls = 
            memAllocs 
            |> List.map CudaCallT.MemAlloc
        let memDisposeCalls = 
            memAllocs 
            |> List.map CudaCallT.MemFree

        let streamAllocCalls = 
            {0 .. streamCnt - 1} 
            |> Seq.map (fun strmId -> StreamCreate(strmId, BasicTypes.CUStreamFlags.NonBlocking))
            |> Seq.toList
        let streamDisposeCalls =
            {0 .. streamCnt - 1} 
            |> Seq.map (fun strmId -> StreamDestory(strmId))
            |> Seq.toList

        let eventAllocCalls =
            {0 .. eventObjCnt - 1}
            |> Seq.map (fun evntId -> EventCreate(evntId, 
                                                  BasicTypes.CUEventFlags.DisableTiming ||| 
                                                  BasicTypes.CUEventFlags.BlockingSync))
            |> Seq.toList
        let eventDisposeCalls =
            {0 .. eventObjCnt - 1}
            |> Seq.map (fun evntId -> EventDestory(evntId))
            |> Seq.toList        

        let textureAllocCalls =
            compileEnv.TextureObjects
            |> Seq.map TextureCreate
            |> Seq.toList
        let textureDisposeCalls = 
            compileEnv.TextureObjects
            |> Seq.map TextureDestroy
            |> Seq.toList

        let subWorkspaceAllocCalls =
            [0 .. compileEnv.SubWorkspaces.Count - 1]
            |> List.map SubWorkspaceNew
        let subWorkspaceDisposeCalls =
            [0 .. compileEnv.SubWorkspaces.Count - 1]
            |> List.map SubWorkspaceDispose

        memAllocCalls @ streamAllocCalls @ eventAllocCalls @ textureAllocCalls @ subWorkspaceAllocCalls, 
        eventDisposeCalls @ streamDisposeCalls @ memDisposeCalls @ textureDisposeCalls @ subWorkspaceDisposeCalls

    /// generate initalization CUDA calls
    let generateInitCalls initItems cache =
        [for cmd in initItems do
            yield! callsForExecItem cmd cache 0]

    type private ResultInfoT = {
        TransferUExpr:  UExprT
        Var:            VarSpecT option
        Stride:         int64 list option
        Allocator:      unit -> IArrayNDT
    }

    /// builds a CUDA recipe for the given unified expression
    let rec buildFromUExprs (cudaCompileEnv: CudaCompileEnvT) (expr: UExprT) =
        if Debug.TraceCompile then printfn "Start of compilation..."

        // generate execution units from unified expression
        if Debug.TraceCompile then printfn "Generating execution units..."
        let sw = Stopwatch.StartNew()
        let euData = CudaExecUnit.exprToCudaExecUnits cudaCompileEnv expr
        let timeForExecUnits = sw.Elapsed

        // map execution units to streams
        if Debug.TraceCompile then printfn "Generating streams..."
        let sw = Stopwatch.StartNew()
        let streams, eventObjCnt = CudaStreamSeq.execUnitsToStreams euData.ExecUnits
        let timeForStreams = sw.Elapsed

        /// generate ops
        if Debug.TraceCompile then printfn "Generating ops..."
        let sw = Stopwatch.StartNew()
        let tmplInstCache = {Insts = []; Code = []}
        for KeyValue (elemFunc, opName) in cudaCompileEnv.ElemFuncsOpNames do
            tmplInstCache |> TmplInstCache.instElemOp elemFunc opName
        let timeForOps = sw.Elapsed

        // generate CUDA calls for execution and initialization
        if Debug.TraceCompile then printfn "Generating calls..."
        let sw = Stopwatch.StartNew()
        let execCalls = generateCalls streams tmplInstCache
        let allocCalls, disposeCalls = 
            generateAllocAndDispose cudaCompileEnv euData.MemAllocs (List.length streams) eventObjCnt
        let initCalls =
            allocCalls @ generateInitCalls euData.InitItems tmplInstCache
        let timeForCalls = sw.Elapsed

        // compile recipes for sub-workspaces
        if Debug.TraceCompile then printfn "Generating sub-recipes..."
        let sw = Stopwatch.StartNew()
        let subRecipes = 
            cudaCompileEnv.SubWorkspaces 
            |> Seq.mapi (fun subWs desc -> subWs, buildFromDesc desc)
            |> Map.ofSeq
        let timeForSubrecipes = sw.Elapsed      

        // diagnostic output
        if Debug.TraceCompile then printfn "Compilation done."
        if Debug.Timing then
            printfn "Time for building CUDA recipe:"
            printfn "Execution units:        %A" timeForExecUnits
            printfn "Stream generation:      %A" timeForStreams
            printfn "Op generation:          %A" timeForOps
            printfn "Call generation:        %A" timeForCalls
            printfn "Sub-recipe generation:  %A" timeForSubrecipes
        if Debug.ResourceUsage then
            let memUsage = euData.MemAllocs |> List.sumBy (fun ma -> ma.ByteSize)
            printfn "Used CUDA memory:       %d MB" (memUsage / pown 2L 20)
            printfn "Used CUDA streams:      %d" streams.Length
            printfn "Used CUDA events:       %d" eventObjCnt
            printfn "Total CUDA exec calls:  %d" execCalls.Length
        if Debug.TerminateAfterRecipeGeneration then
            exit 0

        {
            ChannelVars       = cudaCompileEnv.ChannelVars
            ChannelAllocators = cudaCompileEnv.ChannelAllocators
            VarStrides        = cudaCompileEnv.VarStrides
            VarStorLoc        = cudaCompileEnv.VarStorLoc
            KernelCode        = kernelModuleHeader + TmplInstCache.getCodeForDomain KernelFunc tmplInstCache
            CPPCode           = cppModuleHeader + TmplInstCache.getCodeForDomain CPPFunc tmplInstCache
            InitCalls         = initCalls
            DisposeCalls      = disposeCalls
            ExecCalls         = execCalls
            ConstantValues    = cudaCompileEnv.ConstantValues |> Map.ofDictionary
            SubRecipes        = subRecipes
        }


    /// builds a recipe for evaluating the specified expressions and storing
    /// their results into automatically generated variables
    and buildFromDesc {CompileEnv=compileEnv; UExprs=uexprs} =         
        // add storage op for results
        let resInfos = 
            uexprs
            |> Map.map (fun channel (UExpr (_, _, {ChannelType=ct; ChannelShape=cs}) as uexpr) ->
                let tn, nshp = ct.[dfltChId], cs.[dfltChId]
                let layout = ArrayNDLayout.newC nshp

                // create result storage allocator
                let resAllocator = fun () ->
                    match compileEnv.ResultLoc with
                    | LocHost -> ArrayNDHost.newOfType (TypeName.getType tn) layout :> IArrayNDT
                    | LocDev  -> ArrayNDCuda.newOfType (TypeName.getType tn) layout :> IArrayNDT  
                    | l -> failwithf "CUDA cannot work with result location %A" l      

                if List.fold (*) 1L nshp > 0L then
                    // expression has data that needs to be stored       
                    // create variable that will be inserted into expression
                    let resVarName = sprintf "__%s__" channel
                    let resVar = {
                        VarSpecT.Name = resVarName
                        Shape         = nshp |> List.map SizeSpec.fix
                        TypeName      = tn
                    }

                    // insert StoreToVar op in expression
                    let metadata = {
                        ChannelType  = Map [dfltChId, tn]
                        ChannelShape = Map [dfltChId, [0L]]
                        Expr         = None
                    }
                    {
                        TransferUExpr = UExpr (UUnaryOp (Expr.StoreToVar resVar), [uexpr], metadata)
                        Var           = Some resVar
                        Stride        = compileEnv.ChannelStrides |> Map.tryFind channel
                        Allocator     = resAllocator
                    }
                else
                    // no data needs to be transferred back
                    {
                        TransferUExpr = uexpr
                        Var           = None
                        Stride        = None
                        Allocator     = resAllocator
                    }
                )

        /// result variable locations and strides
        let resVarLocs, resVarStrides = 
            ((Map.empty, Map.empty), resInfos)
            ||> Map.fold (fun (locs, strides) ch resInfo ->
                match resInfo.Var with
                | Some vs -> 
                    let stride = 
                        match resInfo.Stride with
                        | Some stride -> stride
                        | None -> ArrayNDLayout.cStride vs.NShape
                    let locs = locs |> Map.add vs compileEnv.ResultLoc                    
                    let strides = strides |> Map.add vs stride
                    locs, strides                                
                | None -> locs, strides)

        /// unified expression containing all expressions to evaluate
        let mergedUexpr =
            match resInfos |> Map.toList with
            // no expressions
            | [] -> UExpr (UNaryOp Expr.Discard, [], 
                           {ChannelType  = Map [dfltChId, TypeName.ofType<int>]
                            ChannelShape = Map [dfltChId, [0L]]
                            Expr         = None})
            // single expression
            | [_, {TransferUExpr=uexpr}] -> uexpr
            // multiple expressions
            | _::_ ->
                let allUExprs = resInfos |> Map.toList |> List.map (fun (ch, ri) -> ri.TransferUExpr)
                UExpr (UNaryOp Expr.Discard, allUExprs,
                       {ChannelType  = Map [dfltChId, TypeName.ofType<int>]
                        ChannelShape = Map [dfltChId, [0L]]
                        Expr         = None})                       

        // compile expression and create workspace
        let cudaCompileEnv = {
            ChannelVars          = resInfos |> Map.map (fun _ ri -> ri.Var)
            ChannelAllocators    = resInfos |> Map.map (fun _ ri -> ri.Allocator)
            VarStorLoc           = Map.join compileEnv.VarLocs resVarLocs
            VarStrides           = Map.join compileEnv.VarStrides resVarStrides
            ElemFuncsOpNames     = Map.empty    
            TextureObjects       = ResizeArray<_>()
            InterpolatorTextures = Dictionary<_, _>()
            ConstantValues       = Dictionary<_, _>()
            SubWorkspaces        = ResizeArray<_>()
        }
        buildFromUExprs cudaCompileEnv mergedUexpr

