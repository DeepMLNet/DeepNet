module CudaRecipe

open Util
open ExecUnitsGen
open CudaExecUnits
open StreamGen
open CudaBasics
open ManagedCuda

/// Header of generated CUDA module
let cudaModuleHeader =
    "#include \"NDSupport.cuh\"\n\
     #include \"Ops.cuh\"\n\n"

/// CUDA event object
type EventObjectT = int

/// device memory
type DevMemT =
    | DevMemAlloc of MemAllocT
    | DevExternalMem of ExternalMemT
/// device memory pointer
type DevMemPtrT = {Base: DevMemT;
                   Offset: int}

/// pre-allocated host memory 
type HostExternalMemT = {Name: string}
/// host memory pointer
type HostMemPtrT = {Base: HostExternalMemT;
                    Offset: int}

/// CUDA call flags
type CudaFlagsT = int

/// CUDA api call
type CudaCallT =
    // memory mangement
    | MemAlloc of MemAllocT
    | MemFree of MemAllocT
    // memory operations
    | MemcpyAsync of DevMemPtrT * DevMemPtrT * int * StreamT
    | MemcpyHtoDAsync of DevMemPtrT * HostMemPtrT * int * StreamT
    | MemcpyDtoHAsync of HostMemPtrT * DevMemPtrT * int * StreamT
    | MemsetD32Async of DevMemPtrT * single * int * StreamT
    // stream management
    | StreamCreate of StreamT * BasicTypes.CUStreamFlags
    | StreamDestory of StreamT
    | StreamWaitEvent of StreamT * EventObjectT 
    // event mangement
    | EventCreate of EventObjectT * BasicTypes.CUEventFlags
    | EventDestory of EventObjectT
    | EventRecord of EventObjectT * StreamT
    | EventSynchronize of EventObjectT
    // execution control
    | LaunchCPPKernel of TmplInstT * WorkDimT * int * StreamT * (obj list)
    | LaunchCKernel of string * WorkDimT * int * StreamT * (obj list)

/// function instantiation state
type KernelInstCacheT = {mutable Insts: (TmplInstT * string) list;
                         mutable Code: string} 

/// CUDA execution recipe
type CudaRecipeT = {KernelCode: string;
                    InitCalls: CudaCallT list;
                    DisposeCalls: CudaCallT list;
                    ExecCalls: CudaCallT list;}

module CudaRecipe =
    /// gets all CUDA C kernel launches performed 
    let getAllCKernelLaunches recipe =
        let extract = List.filter (fun c -> 
            match c with 
            | LaunchCKernel _ -> true
            | _ -> false)
        (extract recipe.InitCalls) @ (extract recipe.DisposeCalls) @ (extract recipe.ExecCalls)

/// instantiates a template CUDA C++ kernel with a C linkage function name and returns the C function name
let instTmplKernel cache (ti: TmplInstT) =  
    match cache.Insts |> List.tryFind (fun (cti, _) -> cti = ti) with
    | Some (_, cName) -> cName
    | None ->
        // generate C function name
        let nPrv = 
            cache.Insts 
            |> List.filter (fun (oti, name) -> oti.FuncName = ti.FuncName) 
            |> List.length
        let firstArgStr = 
            match ti.TmplArgs with
            | fa::_ -> fa.Replace("<", "_").Replace(">", "_").Replace(".", "_")
            | _ -> ""
        let cName = sprintf "%s_%s_%d" ti.FuncName firstArgStr nPrv
        cache.Insts <- (ti, cName)::cache.Insts

        // generate template instantiation with C linkage
        let instStr =
            if List.isEmpty ti.TmplArgs then ti.FuncName
            else sprintf "%s<%s>" ti.FuncName (ti.TmplArgs |> String.combineWith ", ")
        let argDeclStr = ti.ArgTypes |> List.mapi (fun i t -> sprintf "%s p%d" t i)  |> String.combineWith ", "
        let argCallStr = ti.ArgTypes |> List.mapi (fun i t -> sprintf "p%d" i) |> String.combineWith ", "
        let retCmd = if ti.RetType.Trim() = "void" then "" else "return"
        let declStr =
            sprintf "extern \"C\" __global__ %s %s (%s) {\n" ti.RetType cName argDeclStr
            + sprintf "  %s %s (%s);\n" retCmd ti.FuncName argCallStr
            //+ sprintf "  %s %s (%s);\n" retCmd instStr argCallStr
            + sprintf "}\n"
            + sprintf "\n"
        cache.Code <- cache.Code + declStr

        cName

/// generates a sequence of CUDA calls from streams
let generateCalls streams =    
    /// the number of times WaitOnEvent is called for a particular correlation
    let correlationIdWaiters =
        seq {
            for strm in streams do
                for exec in strm do
                    match exec with
                    | WaitOnEvent evt -> yield evt.CorrelationId
                    | _ -> ()
        } |> Seq.countBy id |> Map.ofSeq
        
    /// mutable kernel instantiation cache
    let cache = {Insts=[]; Code=""}

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
                streamsSorted 
                |> List.find (fun (strmId, strm) ->
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
            | EmitEvent evtp ->
                // add active event as many times as it will be waited upon
                let evt = Option.get !evtp
                let activeEvents = List.replicate correlationIdWaiters.[evt.CorrelationId] evt @ activeEvents

                let cmd = EventRecord (evt.EventObjectId, strmIdToProcess)
                cmd :: generate streamCallHistory activeEvents remainingStreams
            | Perform cmd ->
                // perform a non-synchronization operation
                let streamCallHistory = strmIdToProcess :: streamCallHistory

                let getDevMem t = 
                    match t.Memory with
                    | ExecUnitsGen.MemAlloc m -> {DevMemPtrT.Base = DevMemAlloc m; Offset = 0;}
                    | ExecUnitsGen.ExternalMem m -> {DevMemPtrT.Base = DevExternalMem m; Offset = 0;}

                let getHostMem t =
                    {HostMemPtrT.Base = {HostExternalMemT.Name = t}; Offset=0;}

                // generate CUDA call template
                let calls = 
                    match cmd with
                    | LaunchKernel(ti, workDim, args) -> 
                        [LaunchCKernel(instTmplKernel cache ti, workDim, 0, strmIdToProcess, args)]
                    | MemcpyDtoD(src, trgt) -> 
                        [MemcpyAsync(getDevMem trgt, getDevMem src, NDArrayView.nElems trgt, strmIdToProcess)]
                    | MemcpyHtoD(hostVarName, trgt) -> 
                        [MemcpyHtoDAsync(getDevMem trgt, getHostMem hostVarName, NDArrayView.nElems trgt, strmIdToProcess)]
                    | MemcpyDtoH(src, hostVarName) ->
                        [MemcpyDtoHAsync(getHostMem hostVarName, getDevMem src, NDArrayView.nElems src, strmIdToProcess)]   
                    | Memset(value, trgt) ->                        
                        [MemsetD32Async(getDevMem trgt, single value, NDArrayView.nElems trgt, strmIdToProcess)]            

                calls @ generate streamCallHistory activeEvents remainingStreams
            | ExecUnitStartInfo _ | ExecUnitEndInfo -> 
                generate streamCallHistory activeEvents remainingStreams
        else
            // streams are all empty
            []

    generate [] [] streams, cache


let generateInitAndDispose memAllocs streamCnt eventObjCnt =
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
    let streamDisposeCalls=
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

    memAllocCalls @ streamAllocCalls @ eventAllocCalls, eventDisposeCalls @ streamDisposeCalls @ memDisposeCalls

let buildCudaRecipe cudaEnv sizeSymbolEnv expr =
    let execUnits, exprRes, memAllocs = exprToCudaExecUnits cudaEnv sizeSymbolEnv expr
    let streams, eventObjCnt = execUnitsToStreamCommands execUnits
    let execCalls, krnlCache = generateCalls streams
    let initCalls, disposeCalls = generateInitAndDispose memAllocs (List.length streams) eventObjCnt

    {KernelCode = cudaModuleHeader + krnlCache.Code;
     InitCalls = initCalls;
     DisposeCalls = disposeCalls;
     ExecCalls = execCalls;}


