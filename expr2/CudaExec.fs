module CudaExec

open Util
open ManagedCuda
open CudaBasics
open CudaRecipe
open CudaExecUnits
open ExecUnitsGen
open DiskMap


let hostCompilerDir = @"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64"


/// generated CUDA module counter
let mutable cudaModCntr = 0

/// generates a CUDA module name
let generateCudaModName () =
    cudaModCntr <- cudaModCntr + 1
    sprintf "mod%d.cu" cudaModCntr


/// dumps CUDA kernel code to a file
let dumpCode (modName: string) (modCode: string) =
    System.IO.File.WriteAllText(modName, modCode)
    printfn "Wrote module code to %s" modName

/// Compiles the given CUDA device code into a CUDA module, loads and jits it and returns
/// ManagedCuda.CudaKernel objects for the specified kernel names.
let loadKernelCode modName modCode krnlNames =
    let gpuArch = "compute_30"
    let includePath = assemblyDirectory

    use cmplr = new NVRTC.CudaRuntimeCompiler(modCode, modName)
    let cmplrArgs = [|"--std=c++11";
                      sprintf "--gpu-architecture=%s" gpuArch; 
                      sprintf "--include-path=\"%s\"" includePath|]

    printfn "nvrtc %s %s" (cmplrArgs |> String.combineWith " ") modName 
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
    //use jitLogVerbose = new CudaJOLogVerbose(true)
    //jitOpts.Add(jitLogVerbose)

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
    krnls, cuMod

/// unloads previously loaded CUDA kernel code
let unloadCudaCode cuMod =
    cudaCntxt.UnloadModule(cuMod)

type ModCacheKey = {Code: string; HeaderModTimes: Map<string, System.DateTime>}

let localAppData = System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData)
let cppModCacheDir = System.IO.Path.Combine(localAppData, "expr2", "CPPCache")
let cppModCache = DiskMap<ModCacheKey, byte[]> (cppModCacheDir, "code.dat", "mod.dll")

/// Compiles the given CUDA C++ device/host code into a module, loads it and returns
/// functions objects for the specified C function names.
let loadCppCode modName modCode (funcDelegates: Map<string, System.Type>)  =
    let gpuArch = "compute_30"
    let includePath = assemblyDirectory

    let libName = (System.IO.Path.GetFileNameWithoutExtension modName) + ".dll"

    let cmplrArgs = ["--shared";
                     sprintf "--compiler-bindir \"%s\"" hostCompilerDir;
                     sprintf "--gpu-architecture=%s" gpuArch; 
                     sprintf "--include-path=\"%s\"" includePath;
                     sprintf "-o \"%s\"" libName;
                     sprintf "\"%s\"" modName]
    let cmplrArgStr = cmplrArgs |> String.combineWith " "

    let headerModTimes =
        System.IO.Directory.EnumerateFiles(includePath, "*.cuh")
        |> Seq.map (fun headerFile ->
            System.IO.Path.GetFileName headerFile, System.IO.File.GetLastWriteTimeUtc headerFile)
        |> Map.ofSeq

    let cacheKey = modCode, headerModTimes

    match cppModCache.TryGet cacheKey with
    | Some libData ->
        System.IO.File.WriteAllBytes (libName, libData)
    | None ->
        printfn "nvcc %s" cmplrArgStr
        use prcs = new System.Diagnostics.Process()
        prcs.StartInfo.FileName <- "nvcc.exe"
        prcs.StartInfo.Arguments <- cmplrArgStr
        prcs.StartInfo.UseShellExecute <- false
        prcs.Start() |> ignore
        prcs.WaitForExit()
        if prcs.ExitCode <> 0 then
            printfn "Compile error"
            exit 1

        cppModCache.Set cacheKey (System.IO.File.ReadAllBytes libName)

    // load compiled library
    let libHndl = LoadLibrary(libName)
    if libHndl = System.IntPtr.Zero then
        raise (System.ComponentModel.Win32Exception(sprintf "LoadLibrary of %s failed" libName))

    // get function addresses and build delegates
    let funcs =
        funcDelegates
        |> Map.map (fun name delegateType ->
            let addr = GetProcAddress(libHndl, name)
            if addr = System.IntPtr.Zero then
                 raise (System.ComponentModel.Win32Exception(sprintf "GetProcAddress of %s in %s failed" name libName))
            System.Runtime.InteropServices.Marshal.GetDelegateForFunctionPointer (addr, delegateType))

    funcs, libHndl

/// unloads previously loaded CUDA C++ code
let unloadCppCode libHndl =
    let ret = FreeLibrary(libHndl)
    if not ret then
        raise (System.ComponentModel.Win32Exception("FreeLibrary failed"))        


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
    let internalMem = new Dictionary<MemAllocT, CudaDeviceVariable<byte>>()

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

    /// all C function calls
    let cppCalls = CudaRecipe.getAllCFuncCalls recipe

    /// Function names and delegate types of all C calls
    let cFuncDelegates =
        cppCalls
        |> List.map (fun l ->
            match l with
            | CudaCallT.CallCFunc(name, dgte, _) -> name, dgte
            | _ -> failwith "unexpected C call")
        |> Map.ofList

    // compile and load CUDA kernel module
    let modName = generateCudaModName ()
    do
        dumpCode modName recipe.KernelCode
    /// CUDA kernels
    let kernels, krnlModHndl = loadKernelCode modName recipe.KernelCode kernelCNames

    /// CUDA launch sizes for specified WorkDims
    let kernelLaunchDims =
        kernelDistinctLaunches
        |> Set.toSeq
        |> Seq.map (fun (name, workDim) ->
            let maxBlockSize = kernels.[name].GetOccupancyMaxPotentialBlockSize().blockSize
            (name, workDim), computeLaunchDim workDim maxBlockSize)
        |> Map.ofSeq

    // compile and load CUDA C++ host/device module
    let cppModName = generateCudaModName ()
    do
        dumpCode cppModName recipe.CPPCode
    /// C++ functions
    let cFuncs, cLibHndl = loadCppCode cppModName recipe.CPPCode cFuncDelegates
    
    /// executes the specified calls
    let execCalls (execEnv: CudaExecEnvT) calls =

        for call in calls do
            match call with 
            // memory management
            | CudaRecipe.MemAlloc mem -> 
                let sizeToAlloc = if mem.Size > 0 then mem.Size else 1
                internalMem.Add(mem, new CudaDeviceVariable<byte>(BasicTypes.SizeT(sizeToAlloc * 4)))
            | CudaRecipe.MemFree mem ->
                internalMem.[mem].Dispose()
                internalMem.Remove(mem) |> ignore

            // memory operations
            | MemcpyAsync (dst, src, strm) ->
                let {DeviceVar=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                let {DeviceVar=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                dstCudaVar.AsyncCopyToDevice(srcCudaVar, 
                                             BasicTypes.SizeT(srcOffset), 
                                             BasicTypes.SizeT(dstOffset), 
                                             BasicTypes.SizeT(length), 
                                             streams.[strm].Stream)
            | MemcpyHtoDAsync (dst, src, strm) ->
                let {DeviceVar=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                let {HostVar=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                use srcOffsetVar = new CudaRegisteredHostMemory<byte>(srcCudaVar.PinnedHostPointer + (nativeint srcOffset), 
                                                                      BasicTypes.SizeT(length))
                use dstOffsetVar = new CudaDeviceVariable<byte>(dstCudaVar.DevicePointer + (BasicTypes.SizeT dstOffset), 
                                                                BasicTypes.SizeT(length))
                srcOffsetVar.AsyncCopyToDevice(dstOffsetVar, streams.[strm].Stream)
            | MemcpyDtoHAsync (dst, src, strm) ->
                let {HostVar=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                let {DeviceVar=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                use srcOffsetVar = new CudaDeviceVariable<byte>(srcCudaVar.DevicePointer + (BasicTypes.SizeT srcOffset), 
                                                                BasicTypes.SizeT(length))
                use dstOffsetVar = new CudaRegisteredHostMemory<byte>(dstCudaVar.PinnedHostPointer + (nativeint dstOffset), 
                                                                      BasicTypes.SizeT(length))
                dstOffsetVar.AsyncCopyFromDevice(srcOffsetVar, streams.[strm].Stream)
            | MemsetD32Async (dst, value, strm) ->
                let {DeviceVar=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                use dstOffsetVar = new CudaDeviceVariable<byte>(dstCudaVar.DevicePointer + (BasicTypes.SizeT dstOffset), 
                                                                BasicTypes.SizeT(length))
                let intval = System.BitConverter.ToUInt32(System.BitConverter.GetBytes(value), 0)       
                dstOffsetVar.MemsetAsync(intval, streams.[strm].Stream)

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
            | LaunchCKernel (krnl, workDim, smemSize, strm, argTmpls) ->
                // instantiate args
                let args = argTmpls |> List.map (fun (arg: ICudaArgTmpl) -> arg.GetArg execEnv)
                let argArray = args |> List.toArray

                // launch configuration
                let {Block=blockDim; Grid=gridDim} = kernelLaunchDims.[(krnl, workDim)]
                kernels.[krnl].BlockDimensions <- toDim3 blockDim
                kernels.[krnl].GridDimensions <- toDim3 gridDim
                kernels.[krnl].DynamicSharedMemory <- uint32 smemSize

                kernels.[krnl].RunAsync(streams.[strm].Stream, argArray)
            | LaunchCPPKernel _ ->
                failwith "cannot launch C++ kernel from CudaExec"
            | CudaCallT.CallCFunc (name, _, argTmpls) ->
                // instantiate args
                let args = argTmpls |> List.map (fun (arg: ICudaArgTmpl) -> arg.GetArg execEnv)
                let argArray = args |> List.toArray
 
                let func = cFuncs.[name]   
                func.DynamicInvoke(argArray) |> ignore
            // CUBLAS
            | CublasSetStram strm ->
                cudaBlas.Stream <- streams.[strm].Stream
            | CublasSgemm (aOp, bOp, aFac, a, b, trgtFac, trgt) ->   
                let aVar = (a :> ICudaArgTmpl).GetArg execEnv :?> CudaDeviceVariable<single>            
                let bVar = (b :> ICudaArgTmpl).GetArg execEnv :?> CudaDeviceVariable<single>            
                let trgtVar = (trgt :> ICudaArgTmpl).GetArg execEnv :?> CudaDeviceVariable<single>            
                let m = a.GetRowsForOp execEnv aOp
                let n = b.GetColumnsForOp execEnv bOp
                let k = a.GetColumnsForOp execEnv aOp
                let ldA = a.GetLeadingDimension execEnv
                let ldB = b.GetLeadingDimension execEnv
                let ldTrgt = trgt.GetLeadingDimension execEnv
                cudaBlas.Gemm(aOp, bOp, m, n, k, aFac, aVar, ldA, bVar, ldB, trgtFac, trgtVar, ldTrgt)

    // initialize
    do
        execCalls {InternalMem=internalMem; ExternalVar=Map.empty; HostVar=Map.empty} recipe.InitCalls

    // finalizer
    interface System.IDisposable with
        member this.Dispose() = 
            execCalls {InternalMem=internalMem; ExternalVar=Map.empty; HostVar=Map.empty} recipe.DisposeCalls
            unloadCppCode cLibHndl
            unloadCudaCode krnlModHndl

    /// Evaluate expression.
    member this.Eval(externalVar: Map<Op.VarSpecT, NDArrayDev.NDArrayDev>,
                     hostVar:     Map<Op.VarSpecT, NDArray.NDArray>) =
        execCalls {InternalMem=internalMem; ExternalVar=externalVar; HostVar=hostVar} recipe.ExecCalls
        cudaCntxt.Synchronize () // TODO: remove and signal otherwise




