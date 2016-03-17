namespace SymTensor.Compiler.Cuda

open System.Runtime.InteropServices
open System.IO
open ManagedCuda
open ManagedCuda.BasicTypes
open Basics
open ArrayNDNS
open Basics.Cuda
open SymTensor
open SymTensor.Compiler
open Basics.Cuda
open DiskMap



module Compile = 

    type ModCacheKey = {Code: string; HeaderModTimes: Map<string, System.DateTime>}

    let hostCompilerDir = @"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64"

    let gpuArch = "compute_30"
    let gpuCode = "sm_30"
    let includePath = Util.assemblyDirectory

    let krnlPtxCacheDir = Path.Combine(Util.localAppData, "PTXCache")
    let krnlPtxCache = DiskMap<ModCacheKey, byte[]> (krnlPtxCacheDir, "code.dat", "mod.ptx")

    let cppModCacheDir = Path.Combine(Util.localAppData, "CPPCache")
    let cppModCache = DiskMap<ModCacheKey, byte[]> (cppModCacheDir, "code.dat", "mod.dll")


    /// modification time of C++ header files
    let headerModTimes =
        Directory.EnumerateFiles(includePath, "*.cuh")
        |> Seq.map (fun headerFile ->
            Path.GetFileName headerFile, File.GetLastWriteTimeUtc headerFile)
        |> Map.ofSeq

   
    /// generated CUDA module counter
    let mutable cudaModCntr = 0

    /// generates a CUDA module name
    let generateCudaModName () =
        cudaModCntr <- cudaModCntr + 1
        sprintf "mod%d.cu" cudaModCntr

    /// dumps CUDA kernel code to a file
    let dumpCode (modName: string) (modCode: string) =
        File.WriteAllText(modName, modCode)
        //printfn "Wrote module code to %s" modName

    /// Compiles the given CUDA device code into a CUDA module, loads and jits it and returns
    /// ManagedCuda.CudaKernel objects for the specified kernel names.
    let loadKernelCode modCode krnlNames =
        let modName = generateCudaModName ()

        use cmplr = new NVRTC.CudaRuntimeCompiler(modCode, modName)
        let cmplrArgs = [
            "--std=c++11";
            "-Xcudafe"; "--diag_suppress=declared_but_not_referenced";
            sprintf "--gpu-architecture=%s" gpuArch; 
            sprintf "--include-path=\"%s\"" includePath
        ]

        let dbgArgs = 
            if Debug.DebugCompile then ["--device-debug"; "--generate-line-info"]
            else []

        let cmplrArgs = cmplrArgs @ dbgArgs

        dumpCode modName modCode

        let cacheKey = modCode, headerModTimes, cmplrArgs
        let ptx =
            match krnlPtxCache.TryGet cacheKey with
            | Some ptx -> ptx
            | None ->
                printfn "nvrtc %s %s" (cmplrArgs |> String.concat " ") modName 
                try cmplr.Compile (Array.ofList cmplrArgs)
                with :? NVRTC.NVRTCException as cmplrError ->
                    printfn "Compile error:"
                    let log = cmplr.GetLogAsString()
                    printfn "%s" log
                    exit 1
                let log = cmplr.GetLogAsString()
                printfn "%s" log

                let ptx = cmplr.GetPTX()
                krnlPtxCache.Set cacheKey ptx
                ptx    

        #if !CUDA_DUMMY

        //printfn "CUDA jitting of %s:" modName
        use jitOpts = new CudaJitOptionCollection()
        use jitInfoBuffer = new CudaJOInfoLogBuffer(10000)
        jitOpts.Add(jitInfoBuffer)
        use jitErrorBuffer = new CudaJOErrorLogBuffer(10000)   
        jitOpts.Add(jitErrorBuffer)
        //use jitLogVerbose = new CudaJOLogVerbose(true)
        //jitOpts.Add(jitLogVerbose)

        let cuMod = CudaSup.context.LoadModulePTX(ptx, jitOpts)

        jitOpts.UpdateValues()
        //printfn "%s" jitErrorBuffer.Value
        //printfn "%s" jitInfoBuffer.Value   
        jitErrorBuffer.FreeHandle()
        jitInfoBuffer.FreeHandle()
        //printfn "JIT done."

        let krnls =
            krnlNames
            |> Seq.fold (fun krnls name -> 
                krnls |> Map.add name (CudaKernel(name, cuMod, CudaSup.context))) 
                Map.empty
        krnls, cuMod

        #else

        let krnls: Map<string, CudaKernel> = Map.empty
        krnls, CUmodule()

        #endif

    /// unloads previously loaded CUDA kernel code
    let unloadCudaCode cuMod =
        CudaSup.context.UnloadModule(cuMod)


    /// Compiles the given CUDA C++ device/host code into a module, loads it and returns
    /// functions objects for the specified C function names.
    let loadCppCode modCode (funcDelegates: Map<string, System.Type>)  =
        let modName = generateCudaModName ()
        let libName = (Path.GetFileNameWithoutExtension modName) + ".dll"

        // build argument list
        let baseCmplrArgs = [
            "--shared";
            "--machine 64";
            "-Xcudafe"; "--diag_suppress=declared_but_not_referenced";
            sprintf "--compiler-bindir \"%s\"" hostCompilerDir;                         
            sprintf "--gpu-architecture=%s" gpuArch; 
            sprintf "--gpu-code=%s" gpuCode;
            sprintf "--include-path=\"%s\"" includePath
        ]
        let dbgArgs = 
            if Debug.DebugCompile then ["--debug"; "--device-debug"; "--generate-line-info"]
            else ["--optimize 2"]
        let baseCmplrArgs = baseCmplrArgs @ dbgArgs
        let cmplrArgs = 
            baseCmplrArgs @ [
                sprintf "-o \"%s\"" libName;
                sprintf "\"%s\"" modName
            ]
        let cmplrArgStr = cmplrArgs |> String.concat " "

        dumpCode modName modCode

        let cacheKey = modCode, headerModTimes, baseCmplrArgs
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
        let libHndl = Native.LoadLibrary(libName)
        if libHndl = System.IntPtr.Zero then
            raise (System.ComponentModel.Win32Exception(sprintf "LoadLibrary of %s failed" libName))

        // get function addresses and build delegates
        let funcs =
            funcDelegates
            |> Map.map (fun name delegateType ->
                let addr = Native.GetProcAddress(libHndl, name)
                if addr = System.IntPtr.Zero then
                     raise (System.ComponentModel.Win32Exception(sprintf "GetProcAddress of %s in %s failed" name libName))
                System.Runtime.InteropServices.Marshal.GetDelegateForFunctionPointer (addr, delegateType))

        funcs, libHndl

    /// unloads previously loaded CUDA C++ code
    let unloadCppCode libHndl =
        let ret = Native.FreeLibrary(libHndl)
        if not ret then
            raise (System.ComponentModel.Win32Exception("FreeLibrary failed"))        


[<AutoOpen>]
module CudaExprWorkspaceTypes =    

    /// Workspace for evaluation of an expression compiled to a CudaRecipeT.
    type CudaExprWorkspace(recipe: CudaRecipeT) =

        do if Debug.DisableStreams then printfn "CudaExprWorkspace: redirecting all streams to null stream"

        /// execution environment
        let execEnv = {
            Stream = new Dictionary<StreamT, CudaStream>();
            Event = new Dictionary<EventObjectT, CudaEvent>();
            InternalMem = new Dictionary<MemAllocManikinT, CudaDeviceVariable<byte>>();
            ExternalVar = Map.empty;
            HostVar = Map.empty;
        }

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
                | CudaCallT.CallCFunc(name, dgte, _, _) -> name, dgte
                | _ -> failwith "unexpected C call")
            |> Map.ofList
        
        // compile and load CUDA kernel module
        /// CUDA kernels
        let kernels, krnlModHndl = Compile.loadKernelCode recipe.KernelCode kernelCNames

        #if !CUDA_DUMMY
        /// CUDA launch sizes for specified WorkDims
        let kernelLaunchDims =
            kernelDistinctLaunches
            |> Set.toSeq
            |> Seq.map (fun (name, workDim) ->
                let maxBlockSize = kernels.[name].GetOccupancyMaxPotentialBlockSize().blockSize
                (name, workDim), CudaSup.computeLaunchDim workDim maxBlockSize)
            |> Map.ofSeq
        #else
        let kernelLaunchDims = Map.empty    
        #endif

        // compile and load CUDA C++ host/device module
        /// C++ functions
        let cFuncs, cLibHndl = Compile.loadCppCode recipe.CPPCode cFuncDelegates
    
        let getStream strm = 
            if Debug.DisableStreams then CUstream.NullStream
            else execEnv.Stream.[strm].Stream

        /// executes the specified calls
        let execCalls calls =
            let mutable previousCall = None
            for call in calls do

                match call with 
                // memory management
                | CudaCallT.MemAlloc mem -> 
                    let typeSize = Marshal.SizeOf (TypeName.getType mem.TypeName)
                    let elements = if mem.Elements > 0 then mem.Elements else 1
                    execEnv.InternalMem.Add(mem, new CudaDeviceVariable<byte>(SizeT(elements * typeSize)))
                | CudaCallT.MemFree mem ->
                    execEnv.InternalMem.[mem].Dispose()
                    execEnv.InternalMem.Remove(mem) |> ignore

                // memory operations
                | MemcpyAsync (dst, src, strm) ->
                    let {DeviceMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    let {DeviceMem=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                    dstCudaVar.AsyncCopyToDevice(srcCudaVar, 
                                                 SizeT(srcOffset), 
                                                 SizeT(dstOffset), 
                                                 SizeT(length), 
                                                 getStream strm)
                | MemcpyHtoDAsync (dst, src, strm) ->
                    let {DeviceMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    let {HostMem=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                    use srcOffsetVar = new CudaRegisteredHostMemory<byte>(srcCudaVar.PinnedHostPointer + (nativeint srcOffset), 
                                                                          BasicTypes.SizeT(length))
                    use dstOffsetVar = new CudaDeviceVariable<byte>(dstCudaVar.DevicePointer + (BasicTypes.SizeT dstOffset), 
                                                                    BasicTypes.SizeT(length))
                    srcOffsetVar.AsyncCopyToDevice(dstOffsetVar, getStream strm)
                | MemcpyDtoHAsync (dst, src, strm) ->
                    let {HostMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    let {DeviceMem=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                    use srcOffsetVar = new CudaDeviceVariable<byte>(srcCudaVar.DevicePointer + (BasicTypes.SizeT srcOffset), 
                                                                    BasicTypes.SizeT(length))
                    use dstOffsetVar = new CudaRegisteredHostMemory<byte>(dstCudaVar.PinnedHostPointer + (nativeint dstOffset), 
                                                                          BasicTypes.SizeT(length))
                    dstOffsetVar.AsyncCopyFromDevice(srcOffsetVar, getStream strm)
                | MemsetD32Async (dst, value, strm) ->
                    let {DeviceMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    use dstOffsetVar = new CudaDeviceVariable<byte>(dstCudaVar.DevicePointer + (BasicTypes.SizeT dstOffset), 
                                                                    BasicTypes.SizeT(length))
                    let intval = System.BitConverter.ToUInt32(System.BitConverter.GetBytes(value), 0)       
                    dstOffsetVar.MemsetAsync(intval, getStream strm)

                // stream management
                | StreamCreate (strm, flags) ->
                    if not Debug.DisableStreams then
                        execEnv.Stream.Add(strm, new CudaStream(flags))
                | StreamDestory strm ->
                    if not Debug.DisableStreams then
                        execEnv.Stream.[strm].Dispose()
                        execEnv.Stream.Remove(strm) |> ignore
                | StreamWaitEvent (strm, evnt) ->
                    if not Debug.DisableStreams then
                        execEnv.Stream.[strm].WaitEvent(execEnv.Event.[evnt].Event)

                // event management
                | EventCreate (evnt, flags) ->
                    execEnv.Event.Add(evnt, new CudaEvent(flags))
                | EventDestory evnt ->
                    execEnv.Event.[evnt].Dispose()
                    execEnv.Event.Remove(evnt) |> ignore
                | EventRecord (evnt, strm) ->
                    execEnv.Event.[evnt].Record(getStream strm)
                | EventSynchronize evnt ->
                    execEnv.Event.[evnt].Synchronize()

                // execution control
                | LaunchCKernel (krnl, workDim, smemSize, strm, argTmpls) ->
                    // instantiate args
                    let args = argTmpls |> List.map (fun (tmpl: ICudaArgTmpl) -> tmpl.GetArg execEnv (getStream strm))
                    let argArray = args |> List.toArray

                    // launch configuration
                    let {Block=blockDim; Grid=gridDim} = kernelLaunchDims.[(krnl, workDim)]
                    kernels.[krnl].BlockDimensions <- CudaSup.toDim3 blockDim
                    kernels.[krnl].GridDimensions <- CudaSup.toDim3 gridDim
                    kernels.[krnl].DynamicSharedMemory <- uint32 smemSize

                    if Debug.TraceCalls then
                        printfn "Launching kernel %s on stream %d with work dims %A using block dims %A and grid dims %A" 
                            krnl (getStream strm).Pointer workDim blockDim gridDim

                    if Debug.DisableStreams then
                        kernels.[krnl].Run(argArray) |> ignore
                    else
                        kernels.[krnl].RunAsync(getStream strm, argArray)
                    
                | LaunchCPPKernel _ ->
                    failwith "cannot launch C++ kernel from CudaExec"
                | CudaCallT.CallCFunc (name, _, strm, argTmpls) ->
                    // instantiate args
                    let args = argTmpls |> List.map (fun (tmpl: ICudaArgTmpl) -> tmpl.GetArg execEnv (getStream strm))
                    let argArray = args |> List.toArray
 
                    if Debug.TraceCalls then
                        printfn "Calling C function %s on stream %d" name (getStream strm).Pointer

                    let func = cFuncs.[name]   
                    func.DynamicInvoke(argArray) |> ignore

                // CUBLAS 
                | CublasSgemm (aOp, bOp, aFac, a, b, trgtFac, trgt, strm) ->   
                    let aVar = (a :> ICudaArgTmpl).GetArg execEnv (getStream strm) :?> CudaDeviceVariable<single>            
                    let bVar = (b :> ICudaArgTmpl).GetArg execEnv (getStream strm) :?> CudaDeviceVariable<single>            
                    let trgtVar = (trgt :> ICudaArgTmpl).GetArg execEnv (getStream strm) :?> CudaDeviceVariable<single>            
                    let m = a.GetRowsForOp execEnv aOp
                    let n = b.GetColumnsForOp execEnv bOp
                    let k = a.GetColumnsForOp execEnv aOp
                    let ldA = a.GetLeadingDimension execEnv
                    let ldB = b.GetLeadingDimension execEnv
                    let ldTrgt = trgt.GetLeadingDimension execEnv
                    CudaSup.blas.Stream <- getStream strm
                    CudaSup.blas.Gemm(aOp, bOp, m, n, k, aFac, aVar, ldA, bVar, ldB, trgtFac, trgtVar, ldTrgt)

                // misc
                | Trace (uexpr, res) ->
                    CudaSup.context.Synchronize ()
                    let resDev = CudaExecEnv.getArrayNDForManikin execEnv res
                    let resHost = resDev.ToHost()
                    let msg = sprintf "previous call: %A" previousCall
                    Trace.exprEvaledWithMsg uexpr resHost msg

                previousCall <- Some call

        // initialize
        #if !CUDA_DUMMY
        do execCalls recipe.InitCalls
        #endif

        // finalizer
        interface System.IDisposable with
            member this.Dispose() = 
                execCalls recipe.DisposeCalls
                Compile.unloadCppCode cLibHndl
                Compile.unloadCudaCode krnlModHndl

        /// Evaluate expression.
        member this.Eval(externalVar: Map<UVarSpecT, IArrayNDT>,
                         hostVar:     Map<UVarSpecT, IArrayNDT>) =

            execEnv.ExternalVar <- externalVar |> Map.map (fun _ value -> value :?> IArrayNDCudaT)
            execEnv.HostVar <- hostVar |> Map.map (fun _ value -> value :?> IArrayNDHostT)
            
            execCalls recipe.ExecCalls

            CudaSup.context.Synchronize () // TODO: remove and signal otherwise




