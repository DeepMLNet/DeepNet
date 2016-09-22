namespace SymTensor.Compiler.Cuda

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
        KernelCode:         string
        CPPCode:            string
        InitCalls:          CudaCallT list
        DisposeCalls:       CudaCallT list
        ExecCalls:          CudaCallT list
        ConstantValues:     Map<MemConstManikinT, IArrayNDCudaT>
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

    let commonIncludes = ["Utils.cuh"; "NDSupport.cuh"; "Subtensor.cuh"; "Ops.cuh"]
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

    /// generates a sequence of CUDA calls from streams
    let generateCalls streams cache =    
        /// the number of times WaitOnEvent is called for a particular correlation
        let correlationIdWaiters =
            seq {
                for strm in streams do
                    for exec in strm do
                        match exec with
                        | WaitOnEvent evt -> yield evt.CorrelationId
                        | _ -> ()
            } |> Seq.countBy id |> Map.ofSeq
        
        let rec generate streamCallHistory activeEvents streams =
            if List.exists ((<>) []) streams then
                // sort streams by call history
                let streamsSorted = 
                    streams
                    |> List.indexed
                    |> List.sortByDescending (fun (i, strm) ->                         
                        let callsBetween = 
                            match streamCallHistory |> List.tryFindIndex ((=) i) with
                            | Some ord -> ord
                            | None -> 9999
                        let syncPenalty = 
                            match strm with
                            | EmitEvent _::_ -> 1000
                            | WaitOnEvent _::_ -> -1000
                            | _ -> 0
                        callsBetween + syncPenalty)         

                // find stream to process
                let strmIdToProcess, strmToProcess = 
                    try
                        streamsSorted 
                        |> List.find (fun (_, strm) ->
                            match strm with
                            | WaitOnEvent evt ::_ when 
                                activeEvents |> List.exists (fun e -> e.CorrelationId = evt.CorrelationId) -> true
                                // WaitOnEvent can only be called when EmitEvent 
                                // with same CorrelationId has been called before.
                            | WaitOnEvent _ ::_ -> false
                            | EmitEvent evtp ::_ ->
                                match !evtp with
                                | Some evt when
                                    activeEvents |> List.exists (fun e -> e.EventObjectId = evt.EventObjectId) -> false
                                    // EmitEvent for a given event must be called
                                    // after all necessary calls to WaitOnEvent for a previous correlation.
                                | _ -> true
                            | [] -> false
                            | _ -> true)
                    with
                    :? System.Collections.Generic.KeyNotFoundException ->
                        // cannot find a stream that matches above rules
                        printfn "Error: deadlock during stream sequencing"
                        printfn "Streams to process:\n%A" streamsSorted
                        printfn "Active events:\n%A" activeEvents
                        failwith "deadlock during stream sequencing"

                // book keeping
                let execOp = List.head strmToProcess       
                let remainingStreams = 
                    streams 
                    |> List.map (fun strm -> 
                        if strm = strmToProcess then List.tail strm
                        else strm)

                match execOp with
                | WaitOnEvent evt ->
                    // remove active event
                    let activeEvents = activeEvents |> List.removeValueOnce evt

                    let cmd = StreamWaitEvent (strmIdToProcess, evt.EventObjectId)
                    cmd :: generate streamCallHistory activeEvents remainingStreams
                | WaitOnRerunEvent evt ->
                    let cmd = StreamWaitEvent (strmIdToProcess, evt.EventObjectId)
                    cmd :: generate streamCallHistory activeEvents remainingStreams
                | EmitEvent evtp ->
                    // add active event as many times as it will be waited upon
                    let evt = Option.get !evtp
                    let activeEvents = List.replicate correlationIdWaiters.[evt.CorrelationId] evt @ activeEvents

                    let cmd = EventRecord (evt.EventObjectId, strmIdToProcess)
                    cmd :: generate streamCallHistory activeEvents remainingStreams
                | EmitRerunEvent evtp ->
                    let evt = Option.get !evtp
                    let cmd = EventRecord (evt.EventObjectId, strmIdToProcess)
                    cmd :: generate streamCallHistory activeEvents remainingStreams
                | Perform cmd ->
                    // perform a non-synchronization operation
                    let streamCallHistory = strmIdToProcess :: streamCallHistory

                    // generate CUDA call template
                    let calls = callsForExecItem cmd cache strmIdToProcess

                    calls @ generate streamCallHistory activeEvents remainingStreams
                | RerunSatisfied _ | ExecUnitStart _ | ExecUnitEnd _ -> 
                    // ignore informational markers
                    generate streamCallHistory activeEvents remainingStreams
            else
                // streams are all empty
                []

        generate [] [] streams

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

        memAllocCalls @ streamAllocCalls @ eventAllocCalls @ textureAllocCalls, 
        eventDisposeCalls @ streamDisposeCalls @ memDisposeCalls @ textureDisposeCalls

    /// generate initalization CUDA calls
    let generateInitCalls initItems cache =
        [for cmd in initItems do
            yield! callsForExecItem cmd cache 0]

    /// builds a CUDA recipe for the given unified expression
    let build compileEnv expr =
        if Debug.TraceCompile then printfn "Start of compilation..."

        // generate execution units from unified expression
        if Debug.TraceCompile then printfn "Generating execution units..."
        let sw = Stopwatch.StartNew()
        let euData = CudaExecUnit.exprToCudaExecUnits compileEnv expr
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
        for KeyValue (elemFunc, opName) in compileEnv.ElemFuncsOpNames do
            tmplInstCache |> TmplInstCache.instElemOp elemFunc opName
        let timeForOps = sw.Elapsed

        // generate CUDA calls for execution and initialization
        if Debug.TraceCompile then printfn "Generating calls..."
        let sw = Stopwatch.StartNew()
        let execCalls = generateCalls streams tmplInstCache
        let allocCalls, disposeCalls = 
            generateAllocAndDispose compileEnv euData.MemAllocs (List.length streams) eventObjCnt
        let initCalls =
            allocCalls @ generateInitCalls euData.InitItems tmplInstCache
        let timeForCalls = sw.Elapsed

        // diagnostic output
        if Debug.TraceCompile then printfn "Compilation done."
        if Debug.Timing then
            printfn "Time for building CUDA recipe:"
            printfn "Execution units:        %A" timeForExecUnits
            printfn "Stream generation:      %A" timeForStreams
            printfn "Op generation:          %A" timeForOps
            printfn "Call generation:        %A" timeForCalls
        if Debug.ResourceUsage then
            let memUsage = euData.MemAllocs |> List.sumBy (fun ma -> ma.ByteSize)
            let cmdCounts = List.concat streams |> List.length
            printfn "Used CUDA memory:       %.3f MiB" (float memUsage / 2.**20.)
            printfn "Used CUDA streams:      %d" streams.Length
            printfn "Used CUDA events:       %d" eventObjCnt
            printfn "Total CUDA work calls:  %d" cmdCounts

        {
            KernelCode     = kernelModuleHeader + TmplInstCache.getCodeForDomain KernelFunc tmplInstCache
            CPPCode        = cppModuleHeader + TmplInstCache.getCodeForDomain CPPFunc tmplInstCache
            InitCalls      = initCalls
            DisposeCalls   = disposeCalls
            ExecCalls      = execCalls
            ConstantValues = compileEnv.ConstantValues |> Map.ofDictionary
        }


