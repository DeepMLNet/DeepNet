module CudaRecipe

open Util
open ExecUnitsGen
open CudaExecUnits
open StreamGen
open CudaBasics
open ManagedCuda

/// CUDA event object
type EventObjectT = int

/// CUDA grid dimension
type GridDimT = int * int * int
/// CUDA block dimension
type BlockDimT = int * int * int

/// device memory allocation
type DevMemAllocT = {Id: int; Size: int}
/// device memory pointer
type DevMemPtrT = DevMemAllocT * int

/// host memory allocation
type HostMemAllocT = {Id: int; Size: int}
/// host memory pointer
type HostMemPtrT = HostMemAllocT * int

/// CUDA call flags
type CudaFlagsT = int

/// CUDA api call
type CudaCallT =
    // memory mangement
    | MemAlloc of DevMemPtrT
    | MemFree of DevMemPtrT
    // memory operations
    | MemcpyAsync of DevMemPtrT * DevMemPtrT * int * StreamT
    | MemcpyHtoDAsync of DevMemPtrT * HostMemPtrT * int * StreamT
    | MemcpyDtoHAsync of HostMemPtrT * DevMemPtrT * int * StreamT
    | MemsetD32Async of DevMemPtrT * uint32 * int * StreamT
    // stream management
    | StreamCreate of StreamT * CudaFlagsT
    | StreamDestory of StreamT
    | StreamWaitEvent of StreamT * EventObjectT * CudaFlagsT
    // event mangement
    | EventCreate of EventObjectT * CudaFlagsT
    | EventDestory of EventObjectT
    | EventRecord of EventObjectT * StreamT
    | EventSynchronize of EventObjectT
    // execution control
    | LaunchCPPKernel of TmplInstT * WorkDimT * int * StreamT * (obj list)
    | LaunchCKernel of string * WorkDimT * int * StreamT * (obj list)

/// function instantiation state
type KernelInstCacheT = {mutable Insts: (TmplInstT * string) list;
                         mutable Code: string} 

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

                let cmd = StreamWaitEvent (strmIdToProcess, evt.EventObjectId, 0)
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

                let calls = 
                    match cmd with
                    | LaunchKernel(ti, workDim, args) ->
                        // need to compute BlockDim and GridDim
                        // i.e. let CUDA suggest
                        // however for that the kernel must be compiled first
                        // so let it compile the kernel
                        // but then kernels are compiled one at a time
                        // which might be slow
                        // so kernels should be collected and compiled togehter
                        [LaunchCKernel(instTmplKernel cache ti, workDim, 0, strmIdToProcess, args)]
                    //| Memset(f, view) ->                        
                    | _ -> failwithf "CUDA command %A not implemented" cmd

                calls @ generate streamCallHistory activeEvents remainingStreams
            | ExecUnitStartInfo _ | ExecUnitEndInfo -> 
                generate streamCallHistory activeEvents remainingStreams
        else
            // streams are all empty
            []

    generate [] [] streams, cache


/// Compiles the given CUDA device code into a CUDA module, loads and jits it and returns
/// ManagedCuda.CudaKernel objects for the specified kernel names.
let loadCudaCode modName modCode krnlNames =
    let gpuArch = "compute_30"
    let includePath = assemblyDirectory

    use cmplr = new NVRTC.CudaRuntimeCompiler(modCode, modName)
    let cmplrArgs = [|sprintf "--gpu-architecture=%s" gpuArch; 
                      sprintf "--include-path=\"%s\"" includePath|]

    printfn "CUDA compilation of %s with arguments \"%s\":" modName (cmplrArgs |> String.combineWith " ")
    try
        cmplr.Compile(cmplrArgs)
    with
    | :? NVRTC.NVRTCException as cmplrError ->
        printfn "Compile error:"
        let log = cmplr.GetLogAsString()
        printfn "%s" log

        exit 1
    
    let log = cmplr.GetLogAsString()
    printfn "%s" log

    let ptx = cmplr.GetPTX()
    use jitOpts = new CudaJitOptionCollection()
    use jitInfoBuffer = new CudaJOInfoLogBuffer(10000)
    jitOpts.Add(jitInfoBuffer)
    use jitErrorBuffer = new CudaJOErrorLogBuffer(10000)   
    jitOpts.Add(jitErrorBuffer)
    use jitLogVerbose = new CudaJOLogVerbose(true)
    jitOpts.Add(jitLogVerbose)

    let cuMod = cudaCntxt.LoadModulePTX(ptx, jitOpts)

    jitOpts.UpdateValues()
    printfn "CUDA jitting of %s:" modName
    printfn "%s" jitErrorBuffer.Value
    printfn "%s" jitInfoBuffer.Value   
    jitErrorBuffer.FreeHandle()
    jitInfoBuffer.FreeHandle()

    let krnls =
        krnlNames
        |> Seq.fold (fun krnls name -> 
            krnls |> Map.add name (CudaKernel(name, cuMod, cudaCntxt))) 
            Map.empty
    krnls

let dumpCudaCode modName (modCode : string) =
    let filename = sprintf "%s.cu" modName
    use tw = new System.IO.StreamWriter(filename)
    tw.Write(modCode)
    printfn "Wrote CUDA module code to %s" filename


/// CUDA execution recipe
type CudaRecipeT = {KernelInst: KernelInstCacheT;
                    InitCalls: CudaCallT list;
                    DisposeCalls: CudaCallT list;
                    ExecCalls: CudaCallT list;}

let emptyCudaRecipe = {KernelInst = {Insts=[]; Code=""};
                       InitCalls = [];
                       DisposeCalls = [];
                       ExecCalls = []}


let mutable cudaModuleCounter = 0
let generateCudaModuleName () =
    cudaModuleCounter <- cudaModuleCounter + 1
    sprintf "mod%d" cudaModuleCounter

let cudaModuleHeader =
    "#include \"NDSupport.cuh\"\n\
     #include \"Ops.cuh\"\n\n"


let buildCudaRecipe sizeSymbolEnv expr =
    let execUnits, exprRes, memAllocs = exprToCudaExecUnits sizeSymbolEnv expr
    let streams, eventObjCnt = execUnitsToStreamCommands execUnits
    let calls, krnlCache = generateCalls streams
    let modName = generateCudaModuleName ()

    let modCode = cudaModuleHeader + krnlCache.Code

    dumpCudaCode modName modCode
    loadCudaCode modName modCode []
