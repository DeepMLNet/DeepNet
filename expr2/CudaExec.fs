module CudaExec

open Util
open ManagedCuda
open CudaBasics
open CudaRecipe
open ExecUnitsGen


/// generated CUDA module counter
let mutable cudaModCntr = 0

/// generates a CUDA module name
let generateCudaModName () =
    cudaModCntr <- cudaModCntr + 1
    sprintf "mod%d.cu" cudaModCntr

/// Compiles the given CUDA device code into a CUDA module, loads and jits it and returns
/// ManagedCuda.CudaKernel objects for the specified kernel names.
let loadCudaCode modName modCode krnlNames =
    let gpuArch = "compute_30"
    let includePath = assemblyDirectory

    use cmplr = new NVRTC.CudaRuntimeCompiler(modCode, modName)
    let cmplrArgs = [|"--std=c++11";
                      sprintf "--gpu-architecture=%s" gpuArch; 
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
    let ptx = cmplr.GetPTX()
    
    let log = cmplr.GetLogAsString()
    printfn "%s" log

    printfn "CUDA jitting of %s:" modName
    use jitOpts = new CudaJitOptionCollection()
    use jitInfoBuffer = new CudaJOInfoLogBuffer(10000)
    jitOpts.Add(jitInfoBuffer)
    use jitErrorBuffer = new CudaJOErrorLogBuffer(10000)   
    jitOpts.Add(jitErrorBuffer)
    use jitLogVerbose = new CudaJOLogVerbose(true)
    jitOpts.Add(jitLogVerbose)

    let cuMod = cudaCntxt.LoadModulePTX(ptx, jitOpts)

    jitOpts.UpdateValues()
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

/// dumps CUDA kernel code to a file
let dumpCudaCode (modName: string) (modCode: string) =
    let filename = modName
    use tw = new System.IO.StreamWriter(filename)
    tw.Write(modCode)
    printfn "Wrote CUDA module code to %s" filename


/// Computes CUDA launch dimensions from work dimensions and maximum block size.
/// It is possible that the calculated launch dimensions will be smaller than the
/// specified work dimensions, since the maximum block and grid sizes are limited.
let computeLaunchDim (workDim: CudaExecUnits.WorkDimT) maxBlockSize =
    let wx, wy, wz = workDim
    let mbx, mby, mbz = cudaMaxBlockDim
    let mgx, mgy, mgz = cudaMaxGridDim

    let bx = min mbx (min wx maxBlockSize)
    let by = min mby (min wy (maxBlockSize / bx))
    let bz = min mbz (min wz (maxBlockSize / (bx * by)))

    let gx = min mgx (wx / bx + 1)
    let gy = min mgy (wy / by + 1)
    let gz = min mgz (wz / bz + 1)

    {Block = bx, by, bz; Grid = gx, gy, gz;}
    

/// Workspace for evaluation of an expression compiled to a CudaRecipeT.
type CudaExprWorkspace(recipe: CudaRecipeT) =
    /// stream id to CUDA stream mapping
    let streams = new Dictionary<StreamGen.StreamT, CudaStream>()

    /// event id to CUDA event mapping
    let events = new Dictionary<EventObjectT, CudaEvent>()

    /// memory allocation to CUDA memory mapping
    let internalMem = new Dictionary<MemAllocT, CudaDeviceVariable<single>>()

    /// get memory allocation for device memory
    let getMem mem =
        match mem with
        | DevMemAlloc ma -> internalMem.[ma]
        | DevExternalMem em -> externalMem.[em]

    /// get memory allocation for host memory
    let getHostMem mem =
        hostMem.[mem]

    /// all kernel calls
    let kernelCalls = CudaRecipe.getAllCKernelLaunches recipe

    /// C function names of all kernels
    let kernelCNames = 
        kernelCalls 
        |> List.map (fun l ->
            match l with
            | LaunchCKernel(name, _, _, _, _) -> name
            | _ -> failwith "unexpected CUDA call")
        |> Set.ofList
        |> Set.toList

    /// kernel launches with distinct name/workDim combination
    let kernelDistinctLaunches =
        kernelCalls 
        |> List.map (fun l ->
            match l with
            | LaunchCKernel(name, workDim, _, _, _) -> name, workDim
            | _ -> failwith "unexpected CUDA call")
        |> Set.ofList

    // compile and load CUDA module
    let modName = generateCudaModName ()
    do
        dumpCudaCode modName recipe.KernelCode
    /// CUDA kernels
    let kernels = loadCudaCode modName recipe.KernelCode kernelCNames

    /// CUDA launch sizes for specified WorkDims
    let kernelLaunchDims =
        kernelDistinctLaunches
        |> Set.toSeq
        |> Seq.map (fun (name, workDim) ->
            let maxBlockSize = kernels.[name].GetOccupancyMaxPotentialBlockSize().blockSize
            (name, workDim), computeLaunchDim workDim maxBlockSize)
        |> Map.ofSeq
    
    /// executes the specified calls
    let execCalls (externalMem: Map<ExternalMemT, CudaDeviceVariable<single>>)
                  (hostMem:     Map<HostExternalMemT, CudaPageLockedHostMemory<single>>) 
                  calls =
        for call in calls do
            match call with 
            // memory management
            | CudaRecipe.MemAlloc mem -> 
                internalMem.Add(mem, new CudaDeviceVariable<single>(BasicTypes.SizeT(mem.Size)))
            | CudaRecipe.MemFree mem ->
                internalMem.[mem].Dispose()
                internalMem.Remove(mem) |> ignore

            // memory operations
            | MemcpyAsync (dst, src, length, strm) ->
                let {DevMemPtrT.Base = dstBase; Offset = dstOffset} = dst
                let {DevMemPtrT.Base = srcBase; Offset = srcOffset} = src
                (getMem dstBase).AsyncCopyToDevice(getMem srcBase, 
                                                   BasicTypes.SizeT(srcOffset * sizeof<single>), 
                                                   BasicTypes.SizeT(dstOffset * sizeof<single>), 
                                                   BasicTypes.SizeT(length * sizeof<single>), 
                                                   streams.[strm].Stream)
            | MemcpyHtoDAsync (dst, src, length, strm) ->
                let {DevMemPtrT.Base = dstBase; Offset = dstOffset} = dst
                let {HostMemPtrT.Base = srcBase; Offset = srcOffset} = src
                hostMem.[srcBase].AsyncCopyToDevice(getMem dstBase, 
                                                    BasicTypes.SizeT(srcOffset * sizeof<single>), 
                                                    BasicTypes.SizeT(dstOffset * sizeof<single>), 
                                                    BasicTypes.SizeT(length * sizeof<single>), 
                                                    streams.[strm].Stream)
            | MemcpyDtoHAsync (dst, src, length, strm) ->
                let {HostMemPtrT.Base = dstBase; Offset = dstOffset} = dst
                let {DevMemPtrT.Base = srcBase; Offset = srcOffset} = src
                hostMem.[dstBase].AsyncCopyFromDevice(getMem srcBase, 
                                                      BasicTypes.SizeT(srcOffset * sizeof<single>), 
                                                      BasicTypes.SizeT(dstOffset * sizeof<single>), 
                                                      BasicTypes.SizeT(length * sizeof<single>), 
                                                      streams.[strm].Stream)
            | MemsetD32Async (dst, value, length, strm) ->
                let {DevMemPtrT.Base = dstBase; Offset = dstOffset} = dst
                let intval = System.BitConverter.ToUInt32(System.BitConverter.GetBytes(value), 0)       
                (getMem dstBase).MemsetAsync(intval, streams.[strm].Stream)

            // stream management
            | StreamCreate (strm, flags) ->
                streams.Add(strm, new CudaStream(flags))
            | StreamDestory strm ->
                streams.[strm].Dispose()
                streams.Remove(strm) |> ignore
            | StreamWaitEvent (strm, evnt) ->
                streams.[strm].WaitEvent(events.[evnt].Event)

            // event management
            | EventCreate (evnt, flags) ->
                events.Add(evnt, new CudaEvent(flags))
            | EventDestory evnt ->
                events.[evnt].Dispose()
                events.Remove(evnt) |> ignore
            | EventRecord (evnt, strm) ->
                events.[evnt].Record(streams.[strm].Stream)
            | EventSynchronize evnt ->
                events.[evnt].Synchronize()

            // execution control
            | LaunchCKernel (krnl, workDim, smemSize, strm, args) ->
                let {Block=blockDim; Grid=gridDim} = kernelLaunchDims.[(krnl, workDim)]
                let argArray = args |> List.toArray
                kernels.[krnl].BlockDimensions <- toDim3 blockDim
                kernels.[krnl].GridDimensions <- toDim3 gridDim
                kernels.[krnl].DynamicSharedMemory <- uint32 smemSize
                kernels.[krnl].RunAsync(streams.[strm].Stream, argArray)
            | LaunchCPPKernel _ ->
                failwith "cannot launch C++ kernel from CudaExec"

    // initialize
    do
        execCalls Map.empty Map.empty recipe.InitCalls

    // finalizer
    interface System.IDisposable with
        member this.Dispose() = execCalls Map.empty Map.empty recipe.DisposeCalls

    /// Evaluate expression.
    member this.Eval(externalMem: Map<ExternalMemT, CudaDeviceVariable<single>>,
                     hostMem:     Map<HostExternalMemT, CudaPageLockedHostMemory<single>>) =
        execCalls externalMem hostMem recipe.ExecCalls



