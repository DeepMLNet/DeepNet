namespace SymTensor.Compiler.Cuda

open System.Diagnostics
open System.Runtime.InteropServices
open System.IO
open System.Reflection
open System.Security.Cryptography

open ManagedCuda
open ManagedCuda.BasicTypes
open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler
open UExprTypes
open Basics.Cuda
open DiskMap



module Compile = 

    type ModCacheKey = {
        Code:           string
        HeaderHashes:   Map<string, byte list>
        CompilerArgs:   string list
    }

    let hostCompilerDir = @"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64"

    let krnlPtxCacheDir = Path.Combine(Util.localAppData, "PTXCache")
    let krnlPtxCache = DiskMap<ModCacheKey, byte[]> (krnlPtxCacheDir, "code.dat", "mod.ptx")

    let cppModCacheDir = Path.Combine(Util.localAppData, "CPPCache")
    let cppModCache = DiskMap<ModCacheKey, byte[]> (cppModCacheDir, "code.dat", "mod.dll")

    let compileDirRoot = Path.Combine(Util.localAppData, "Compile")

    /// prepares a compile directory
    let prepareCompileDir code =        
        // create temp directory
        let rec getTempDir () =  
            let dirname = Path.Combine(compileDirRoot, Path.GetRandomFileName())
            if not (Directory.Exists dirname) then dirname
            else getTempDir()
        let compileDir = getTempDir ()
        Directory.CreateDirectory compileDir |> ignore

        // get embedded header files from out assembly
        let asmbly = Assembly.GetExecutingAssembly()
        let headers =
            asmbly.GetManifestResourceNames ()
            |> Seq.filter (fun s -> s.EndsWith(".cuh") || s.EndsWith(".h"))

        // calculate MD5 sum of headers
        let headerHashes =
            headers
            |> Seq.map (fun header -> 
                use strm = asmbly.GetManifestResourceStream(header)
                use md5 = MD5.Create()
                (header, md5.ComputeHash strm |> Array.toList))
            |> Map.ofSeq

        // write headers to compile directory
        headers
        |> Seq.iter (fun header -> 
            use strm = asmbly.GetManifestResourceStream(header)
            let filename = Path.Combine (compileDir, header)
            use fileStrm = File.OpenWrite filename
            strm.CopyTo fileStrm)

        // write module code
        let modPath = Path.Combine (compileDir, "mod.cu")
        File.WriteAllText(modPath, code)

        compileDir, modPath, headerHashes

    /// removes a compile directory
    let removeCompileDir compileDir =
        Directory.Delete(compileDir, true)       

    /// code files writted so far
    let mutable codeDumpCnt = 0

    /// writes code to a file
    let dumpCode code =
        let filename = sprintf "mod%d.cu" codeDumpCnt
        File.WriteAllText (filename, code)
        printfn "Dumped code to %s" (Path.GetFullPath filename)
        codeDumpCnt <- codeDumpCnt + 1
   
    /// Compiles the given CUDA device code into a CUDA module, loads and jits it and returns
    /// ManagedCuda.CudaKernel objects for the specified kernel names.
    let loadKernelCode modCode krnlNames =
        if Debug.DumpCode then dumpCode modCode

        let compileDir, modPath, headerHashes = prepareCompileDir modCode

        use cmplr = new NVRTC.CudaRuntimeCompiler(modCode, modPath)
        let baseCmplrArgs = [
            "--std=c++11"
            "-DWIN32_LEAN_AND_MEAN"
            "-Xcudafe"; "--diag_suppress=declared_but_not_referenced"
            sprintf "--gpu-architecture=%s" CudaSup.nvccArch
        ]
        let dbgArgs = 
            if Debug.DebugCompile then ["--device-debug"; "--generate-line-info"]
            else []
        let baseCmplrArgs = baseCmplrArgs @ dbgArgs
        let cmplrArgs = 
            baseCmplrArgs @ [ 
                sprintf "--include-path=\"%s\"" compileDir
            ]

        let cacheKey = {Code=modCode; HeaderHashes=headerHashes; CompilerArgs=baseCmplrArgs}
        let ptx =
            match krnlPtxCache.TryGet cacheKey with
            | Some ptx -> ptx
            | None ->
                let sw = Stopwatch.StartNew ()
                if Debug.TraceCompile then
                    printfn "nvrtc %s %s" (cmplrArgs |> String.concat " ") modPath 
                try cmplr.Compile (Array.ofList cmplrArgs)
                with :? NVRTC.NVRTCException as cmplrError ->
                    printfn "Compile error:"
                    let log = cmplr.GetLogAsString()
                    printfn "%s" log
                    exit 1
                if Debug.TraceCompile then
                    let log = cmplr.GetLogAsString()
                    printf "%s" log
                if Debug.Timing then printfn "nvrtc took %A" sw.Elapsed

                let ptx = cmplr.GetPTX()
                krnlPtxCache.Set cacheKey ptx                
                ptx    

        #if !CUDA_DUMMY

        let sw = Stopwatch.StartNew ()
        if Debug.TraceCompile then
            printfn "JITing PTX code..."
       
        use jitOpts = new CudaJitOptionCollection()
        use jitInfoBuffer = new CudaJOInfoLogBuffer(10000)
        jitOpts.Add(jitInfoBuffer)
        use jitErrorBuffer = new CudaJOErrorLogBuffer(10000)   
        jitOpts.Add(jitErrorBuffer)
        use jitLogVerbose = new CudaJOLogVerbose(true)
        jitOpts.Add(jitLogVerbose)

        let cuMod = CudaSup.context.LoadModulePTX(ptx, jitOpts)

        jitOpts.UpdateValues()
        if Debug.PtxasInfo then
            printfn "%s" jitErrorBuffer.Value
            printfn "%s" jitInfoBuffer.Value   
        jitErrorBuffer.FreeHandle()
        jitInfoBuffer.FreeHandle()

        if Debug.Timing then printfn "JITing PTX code took %A" sw.Elapsed

        let krnls =
            krnlNames
            |> Seq.fold (fun krnls name -> 
                krnls |> Map.add name (CudaKernel(name, cuMod, CudaSup.context))) 
                Map.empty
        krnls, cuMod, compileDir

        #else

        let krnls: Map<string, CudaKernel> = Map.empty
        krnls, CUmodule(), compileDir

        #endif

    /// unloads previously loaded CUDA kernel code
    let unloadCudaCode cuMod =
        CudaSup.context.UnloadModule(cuMod)


    /// Compiles the given CUDA C++ device/host code into a module, loads it and returns
    /// functions objects for the specified C function names.
    let loadCppCode modCode (funcDelegates: Map<string, System.Type>) =
        if Debug.DumpCode then dumpCode modCode

        let compileDir, modPath, headerHashes = prepareCompileDir modCode
        let libPath = Path.Combine (compileDir, "mod.dll")

        // build argument list
        let baseCmplrArgs = [
            "--shared"
            "--machine 64"
            "-DWIN32_LEAN_AND_MEAN"
            "-Xcudafe"; "--diag_suppress=declared_but_not_referenced";
            sprintf "--compiler-bindir \"%s\"" hostCompilerDir                        
            sprintf "--gpu-architecture=%s" CudaSup.nvccArch 
            sprintf "--gpu-code=%s" CudaSup.nvccCode
        ]
        let dbgArgs = 
            if Debug.DebugCompile then ["--debug"; "--device-debug"; "--generate-line-info"]
            else ["--optimize 2"]
        let baseCmplrArgs = baseCmplrArgs @ dbgArgs
        let cmplrArgs = 
            baseCmplrArgs @ [
                sprintf "--include-path=\"%s\"" compileDir
                sprintf "-o \"%s\"" libPath;
                sprintf "\"%s\"" modPath
            ]
        let cmplrArgStr = cmplrArgs |> String.concat " "

        let cacheKey = {Code=modCode; HeaderHashes=headerHashes; CompilerArgs=baseCmplrArgs}
        match cppModCache.TryGet cacheKey with
        | Some libData ->
            System.IO.File.WriteAllBytes (libPath, libData)
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

            cppModCache.Set cacheKey (System.IO.File.ReadAllBytes libPath)

        // load compiled library
        let libHndl = Native.LoadLibrary(libPath)
        if libHndl = System.IntPtr.Zero then
            raise (System.ComponentModel.Win32Exception(sprintf "LoadLibrary of %s failed" libPath))

        // get function addresses and build delegates
        let funcs =
            funcDelegates
            |> Map.map (fun name delegateType ->
                let addr = Native.GetProcAddress(libHndl, name)
                if addr = System.IntPtr.Zero then
                     raise (System.ComponentModel.Win32Exception(sprintf "GetProcAddress of %s in %s failed" name libPath))
                System.Runtime.InteropServices.Marshal.GetDelegateForFunctionPointer (addr, delegateType))

        funcs, libHndl, compileDir

    /// unloads previously loaded CUDA C++ code
    let unloadCppCode libHndl =
        let ret = Native.FreeLibrary(libHndl)
        if not ret then
            raise (System.ComponentModel.Win32Exception("FreeLibrary failed"))        


[<AutoOpen>]
module CudaExprWorkspaceTypes =    

    /// Workspace for evaluation of an expression compiled to a CudaRecipeT.
    type CudaExprWorkspace (recipe: CudaRecipeT) =
        let mutable disposed = false

        //do if Debug.DisableStreams then printfn "CudaExprWorkspace: redirecting all streams to null stream"

        /// execution environment
        let execEnv = {
            Stream         = new Dictionary<StreamT, CudaStream>()
            Event          = new Dictionary<EventObjectT, CudaEvent>()
            InternalMem    = new Dictionary<MemAllocManikinT, CudaDeviceVariable<byte>>()
            RegHostMem     = new Dictionary<MemAllocManikinT, RegHostMemT>()
            ExternalVar    = Map.empty
            HostVar        = Map.empty
            TextureObject  = new Dictionary<TextureObjectT, CudaTexObjectAndArray>()
            ConstantValues = recipe.ConstantValues
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
        let kernels, krnlModHndl, krnlCompileDir = 
            Compile.loadKernelCode recipe.KernelCode kernelCNames

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
        let cFuncs, cLibHndl, cCompileDir = 
            if not (Map.isEmpty cFuncDelegates) then
                let cFuncs, cLibHndl, cCompileDir = Compile.loadCppCode recipe.CPPCode cFuncDelegates
                cFuncs, Some cLibHndl, Some cCompileDir
            else
                Map.empty, None, None
    
        /// get CUstream of stream object
        let getStream strm = 
            if Debug.DisableStreams then CUstream.NullStream
            else execEnv.Stream.[strm].Stream

        /// executes the specified calls
        let execCalls calls =
            let mutable previousCall = None
            for call in calls do
                if Debug.TraceCalls then
                    match call with
                    | ExecItem (Trace _, _) -> ()
                    | _ -> printfn "CUDA call: %A" call

                match call with 
                // memory management
                | CudaCallT.MemAlloc mem -> 
                    let typeSize = Marshal.SizeOf (TypeName.getType mem.TypeName)
                    let elements = if mem.Elements > 0 then mem.Elements else 1
                    match mem.Kind with
                    | MemAllocDev ->
                        try
                            execEnv.InternalMem.Add(mem, new CudaDeviceVariable<byte>(SizeT (elements * typeSize)))
                        with :? CudaException as e when e.CudaError = CUResult.ErrorOutOfMemory ->
                            failwithf "Out of CUDA memory while allocating %d bytes" (elements * typeSize)
                    | MemAllocRegHost ->
                        let sizeInBytes = elements * typeSize
                        let ptr = Marshal.AllocHGlobal sizeInBytes
                        let cudaRegMem = new CudaRegisteredHostMemory<byte> (ptr, SizeT sizeInBytes)
                        cudaRegMem.Register (CUMemHostRegisterFlags.None)
                        execEnv.RegHostMem.Add(mem, {Ptr=ptr; CudaRegHostMem=cudaRegMem})
                | CudaCallT.MemFree mem ->
                    match mem.Kind with
                    | MemAllocDev ->
                        if execEnv.InternalMem.ContainsKey mem then
                            execEnv.InternalMem.[mem].Dispose()
                            execEnv.InternalMem.Remove mem |> ignore
                    | MemAllocRegHost ->
                        if execEnv.RegHostMem.ContainsKey mem then
                            execEnv.RegHostMem.[mem].CudaRegHostMem.Unregister ()
                            execEnv.RegHostMem.[mem].CudaRegHostMem.Dispose()
                            Marshal.FreeHGlobal execEnv.RegHostMem.[mem].Ptr
                            execEnv.RegHostMem.Remove mem |> ignore

                // stream management
                | StreamCreate (strm, flags) ->
                    if not Debug.DisableStreams then
                        execEnv.Stream.Add(strm, new CudaStream(flags))
                | StreamDestory strm ->
                    if not Debug.DisableStreams then
                        if execEnv.Stream.ContainsKey strm then
                            execEnv.Stream.[strm].Dispose()
                            execEnv.Stream.Remove(strm) |> ignore
                | StreamWaitEvent (strm, evnt) ->
                    if not Debug.DisableStreams then
                        execEnv.Stream.[strm].WaitEvent(execEnv.Event.[evnt].Event)

                // event management
                | EventCreate (evnt, flags) ->
                    if not Debug.DisableEvents then
                        execEnv.Event.Add(evnt, new CudaEvent(flags))
                | EventDestory evnt ->
                    if not Debug.DisableEvents then
                        if execEnv.Event.ContainsKey evnt then
                            execEnv.Event.[evnt].Dispose()
                            execEnv.Event.Remove(evnt) |> ignore
                | EventRecord (evnt, strm) ->
                    if not Debug.DisableEvents then
                        execEnv.Event.[evnt].Record(getStream strm)
                | EventSynchronize evnt ->
                    if not Debug.DisableEvents then
                        execEnv.Event.[evnt].Synchronize()

                // texture object management
                | TextureCreate tex ->
                    let devVar, _ = CudaExecEnv.getDevMemForManikin execEnv tex.Contents
                    let devAry, resDsc =
                        match tex.Contents.NDims with
                        | 1 ->
                            if (ArrayND.stride tex.Contents).[0] <> 1 then
                                failwith "texture contents must be continuous"
                            let nElems = tex.Contents.Shape.[0]
                            let devAry = new CudaArray1D
                                           (CUArrayFormat.Float, SizeT nElems, CudaArray1DNumChannels.One)
                            devAry.CopyFromDeviceToArray1D 
                                (devVar.DevicePointer, SizeT (nElems * sizeof<single>), 
                                 SizeT (ArrayND.offset tex.Contents * sizeof<single>))
                            (devAry :> System.IDisposable), CudaResourceDesc (devAry)
                        | 2 ->
                            if (ArrayND.stride tex.Contents).[1] <> 1 then
                                failwith "texture contents must be continuous in last dimension"
                            let devAry = new CudaArray2D(CUArrayFormat.Float, 
                                                         SizeT tex.Contents.Shape.[1],
                                                         SizeT tex.Contents.Shape.[0], 
                                                         CudaArray2DNumChannels.One)
                            use pdv = 
                                new CudaPitchedDeviceVariable<single> 
                                    (devVar.DevicePointer + SizeT (ArrayND.offset tex.Contents * sizeof<single>), 
                                     SizeT tex.Contents.Shape.[1], SizeT tex.Contents.Shape.[0], 
                                     SizeT ((ArrayND.stride tex.Contents).[0]) * sizeof<single>) 
                            devAry.CopyFromDeviceToThis (pdv)
                            (devAry :> System.IDisposable), CudaResourceDesc (devAry)
                        | 3 ->
                            if (ArrayND.stride tex.Contents).[2] <> 1 then
                                failwith "texture contents must be continuous in last dimension"
                            if (ArrayND.stride tex.Contents).[0] <> 
                               (ArrayND.stride tex.Contents).[1] * tex.Contents.Shape.[1] then
                                failwith "texture contents must be continuous in first dimension"
                            let devAry = new CudaArray3D(CUArrayFormat.Float, 
                                                         SizeT tex.Contents.Shape.[2],
                                                         SizeT tex.Contents.Shape.[1],
                                                         SizeT tex.Contents.Shape.[0], 
                                                         CudaArray3DNumChannels.One,
                                                         CUDAArray3DFlags.None)
                            devAry.CopyFromDeviceToThis 
                                (devVar.DevicePointer + SizeT (ArrayND.offset tex.Contents * sizeof<single>),
                                 SizeT sizeof<single>, 
                                 SizeT ((ArrayND.stride tex.Contents).[1]) * sizeof<single>)
                            (devAry :> System.IDisposable), CudaResourceDesc (devAry)
                        | d -> failwithf "unsupported number of dimensions for texture: %d" d
                    let texObj = new CudaTexObject (resDsc, tex.Descriptor)
                    execEnv.TextureObject.[tex] <- {TexObject=texObj; TexArray=devAry}
                | TextureDestroy tex -> 
                    if execEnv.TextureObject.ContainsKey tex then
                        execEnv.TextureObject.[tex].TexObject.Dispose()
                        execEnv.TextureObject.[tex].TexArray.Dispose()
                        execEnv.TextureObject.Remove(tex) |> ignore

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
                            krnl strm workDim blockDim gridDim

                    kernels.[krnl].RunAsync(getStream strm, argArray)                   
                | LaunchCPPKernel _ ->
                    failwith "cannot launch C++ kernel from CudaExec"
                | CudaCallT.CallCFunc (name, _, strm, argTmpls) ->
                    // instantiate args
                    let args = argTmpls |> List.map (fun (tmpl: ICudaArgTmpl) -> tmpl.GetArg execEnv (getStream strm))
                    let argArray = args |> List.toArray
 
                    if Debug.TraceCalls then
                        printfn "Calling C function %s on stream %d" name strm

                    let func = cFuncs.[name]   
                    func.DynamicInvoke(argArray) |> ignore

                // ======================= ExecItems execution ===================================================

                | ExecItem (CudaExecItemT.LaunchKernel _, _)
                | ExecItem (CudaExecItemT.CallCFunc _, _)
                    -> failwith "these ExecItems must be translated to CudaExecItems"

                // memory operations
                | ExecItem (MemcpyDtoD (src, dst), strm) ->
                    let {DeviceMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    let {DeviceMem=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                    dstCudaVar.AsyncCopyToDevice(srcCudaVar, 
                                                 SizeT(srcOffset), 
                                                 SizeT(dstOffset), 
                                                 SizeT(length), 
                                                 getStream strm)
                | ExecItem (MemcpyHtoD (src, dst), strm) ->
                    let {DeviceMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    let {HostMem=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                    use srcOffsetVar = new CudaRegisteredHostMemory<byte>(srcCudaVar.PinnedHostPointer + (nativeint srcOffset), 
                                                                          BasicTypes.SizeT(length))
                    use dstOffsetVar = new CudaDeviceVariable<byte>(dstCudaVar.DevicePointer + (BasicTypes.SizeT dstOffset), 
                                                                    BasicTypes.SizeT(length))
                    srcOffsetVar.AsyncCopyToDevice(dstOffsetVar, getStream strm)
                    if Debug.TraceCalls then
                        printfn "MemcpyHtoD of %d bytes on stream %d" length strm
                | ExecItem (MemcpyDtoH (src, dst), strm) ->
                    let {HostMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    let {DeviceMem=srcCudaVar; OffsetInBytes=srcOffset} = src.GetRng execEnv
                    use srcOffsetVar = new CudaDeviceVariable<byte>(srcCudaVar.DevicePointer + (BasicTypes.SizeT srcOffset), 
                                                                    BasicTypes.SizeT(length))
                    use dstOffsetVar = new CudaRegisteredHostMemory<byte>(dstCudaVar.PinnedHostPointer + (nativeint dstOffset), 
                                                                          BasicTypes.SizeT(length))
                    dstOffsetVar.AsyncCopyFromDevice(srcOffsetVar, getStream strm)
                    if Debug.TraceCalls then
                        printfn "MemcpyDtoH of %d bytes on stream %d" length strm
                | ExecItem (MemsetSingle (value, dst), strm) ->
                    let {DeviceMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    use dstOffsetVar = new CudaDeviceVariable<byte>(dstCudaVar.DevicePointer + (BasicTypes.SizeT dstOffset), 
                                                                    BasicTypes.SizeT(length))
                    let intval = System.BitConverter.ToUInt32(System.BitConverter.GetBytes(value), 0)       
                    dstOffsetVar.MemsetAsync(intval, getStream strm)
                | ExecItem (MemsetUInt32 (value, dst), strm) ->
                    let {DeviceMem=dstCudaVar; OffsetInBytes=dstOffset; LengthInBytes=length} = dst.GetRng execEnv
                    use dstOffsetVar = new CudaDeviceVariable<byte>(dstCudaVar.DevicePointer + (BasicTypes.SizeT dstOffset), 
                                                                    BasicTypes.SizeT(length))    
                    dstOffsetVar.MemsetAsync(value, getStream strm)

                // CUBLAS 
                | ExecItem (BlasGemm (aOp, bOp, aFac, a, b, trgtFac, trgt), strm) ->   
                    use aVar = a.GetVar execEnv
                    use bVar = b.GetVar execEnv
                    use trgtVar = trgt.GetVar execEnv
                    let m = a.GetRowsForOp execEnv aOp.CudaBlasOperation
                    let n = b.GetColumnsForOp execEnv bOp.CudaBlasOperation
                    let k = a.GetColumnsForOp execEnv aOp.CudaBlasOperation
                    let ldA = a.GetLeadingDimension execEnv
                    let ldB = b.GetLeadingDimension execEnv
                    let ldTrgt = trgt.GetLeadingDimension execEnv
                    CudaSup.blas.Stream <- getStream strm
                    CudaSup.blas.Gemm(aOp.CudaBlasOperation, bOp.CudaBlasOperation, 
                                      m, n, k, aFac, aVar, ldA, bVar, ldB, trgtFac, 
                                      trgtVar, ldTrgt)

                | ExecItem (BlasGemmBatched (aOp, bOp, aFac, a, b, trgtFac, trgt), strm) ->   
                    use aAry = a.GetPointerArrayDevice execEnv
                    use bAry = b.GetPointerArrayDevice execEnv
                    use trgtAry = trgt.GetPointerArrayDevice execEnv                    
                    let m = a.GetRowsForOp aOp.CudaBlasOperation
                    let n = b.GetColumnsForOp bOp.CudaBlasOperation
                    let k = a.GetColumnsForOp aOp.CudaBlasOperation
                    let ldA = a.LeadingDimension 
                    let ldB = b.LeadingDimension 
                    let ldTrgt = trgt.LeadingDimension 

                    if Debug.TraceCalls then
                        printfn "Executing GemmBatched on stream %d with m=%d, n=%d, k=%d, \
                                 ldA=%d, ldB=%d, ldTrgt=%d, nSamples=%d" 
                            strm m n k ldA ldB ldTrgt a.NSamples

                    CudaSup.blas.Stream <- getStream strm
                    CudaSup.blas.GemmBatched(aOp.CudaBlasOperation, bOp.CudaBlasOperation, 
                                             m, n, k, aFac, aAry, ldA, bAry, ldB, trgtFac, 
                                             trgtAry, ldTrgt, a.NSamples)

                | ExecItem (BlasGetrfBatched (a, pivot, info), strm) ->
                    use aAry = a.GetPointerArrayDevice execEnv
                    let n = a.Rows 
                    let ldA = a.LeadingDimension 
                    let pVar = pivot.GetVar execEnv
                    let infoVar = info.GetVar execEnv
                    CudaSup.blas.Stream <- getStream strm
                    CudaSup.blas.GetrfBatchedS (n, aAry, ldA, pVar, infoVar, a.NSamples)

                | ExecItem (BlasGetriBatched (a, pivot, trgt, info), strm) ->
                    use aAry = a.GetPointerArrayDevice execEnv
                    let n = a.Rows 
                    let ldA = a.LeadingDimension 
                    let pVar = pivot.GetVar execEnv
                    let trgtAry = trgt.GetPointerArrayDevice execEnv
                    let ldC = trgt.LeadingDimension 
                    let infoVar = info.GetVar execEnv
                    CudaSup.blas.Stream <- getStream strm
                    CudaSup.blas.GetriBatchedS (n, aAry, ldA, pVar, trgtAry, ldC, infoVar, 
                                                a.NSamples)

                | ExecItem (BlasInitPointerArray (aryTmpl), strm) ->
                    let cacheKey = aryTmpl.PointerArrayCacheKey execEnv
                    if aryTmpl.PointerArrayCacheKeyOnDevice <> Some cacheKey then
                        let ptrAryValues = aryTmpl.GetPointerArrayValues execEnv
                        let {Ptr=ptrAryHostPtr; CudaRegHostMem=ptrAryHostVar} = aryTmpl.GetPointerArrayHost execEnv 
                        for n = 0 to ptrAryValues.Length - 1 do
                            Marshal.StructureToPtr (ptrAryValues.[n], 
                                                    ptrAryHostPtr + nativeint (n * sizeof<CUdeviceptr>), false)

                        use ptrAryDevVar = aryTmpl.GetPointerArrayDevice execEnv
                        ptrAryHostVar.AsyncCopyToDevice (ptrAryDevVar.DevicePointer, getStream strm)
                        aryTmpl.PointerArrayCacheKeyOnDevice <- Some cacheKey

                        if Debug.TraceCalls then
                            printfn "Initializing BLAS pointer array on stream %d" strm

                | ExecItem (ExtensionExecItem eei, strm) ->
                    eei.Execute execEnv strm

                | ExecItem (PrintWithMsg (msg, res), strm) ->
                    CudaSup.context.Synchronize ()
                    let resDev = CudaExecEnv.getArrayNDForManikin execEnv res
                    let resHost = resDev.ToHost()    
                    printfn "%s=\n%A\n" msg resHost                

                | ExecItem (DumpValue (name, res), strm) ->
                    if Dump.isActive () then
                        CudaSup.context.Synchronize ()
                        let resDev = CudaExecEnv.getArrayNDForManikin execEnv res
                        let resHost = resDev.ToHost()    
                        Dump.dumpValue name resHost

                | ExecItem (CheckNonFiniteCounter (name, counter), strm) ->
                    if Debug.TerminateWhenNonFinite then
                        // source counter
                        let counterVarByteVar, _ = CudaExecEnv.getDevMemForManikin execEnv counter
                        use counterVarIntVar = new CudaDeviceVariable<int> (counterVarByteVar.DevicePointer)

                        // temporary destination
                        let counterVar : int[] = Array.zeroCreate 1
                        let gcHnd = GCHandle.Alloc(counterVar, GCHandleType.Pinned)
                        use counterVarPinned = new CudaRegisteredHostMemory<int>(gcHnd.AddrOfPinnedObject(), SizeT 1)

                        // copy from source counter to temporary destination
                        counterVarPinned.AsyncCopyFromDevice(counterVarIntVar, getStream strm)

                        // add callback when copy is finished
                        let callback (hStream: CUstream) (status: CUResult) (userData: System.IntPtr) =
                            gcHnd.Free()
                            if counterVar.[0] <> 0 then
                                printfn "Infinity or NaN encountered in %d elements of %s." counterVar.[0] name 
                                exit 1
                        use stream = new CudaStream (getStream strm)
                        stream.AddCallback (CUstreamCallback callback, nativeint 0, CUStreamAddCallbackFlags.None)

                // trace
                | ExecItem (Trace (uexpr, res), _) ->
                    try
                        CudaSup.context.Synchronize ()
                    with :? CudaException as ex ->
                        printfn "CUDA exception during trace: %A" ex
                        match previousCall with
                        | Some pc -> printfn "Last call was %A" pc
                        | None -> ()
                        printfn "Expression was %A" uexpr

                        let crashTraceFile = "crash_trace.txt"
                        use tw = File.CreateText crashTraceFile
                        Trace.dumpActiveTrace tw
                        printfn "Dumped active trace to %s" (Path.GetFullPath crashTraceFile)
                        reraise()

                    let resDev = CudaExecEnv.getArrayNDForManikin execEnv res
                    let resHost = resDev.ToHost()
                    let msg = 
                        match previousCall with
                        | Some (ExecItem (Trace _, _)) | None -> "no previous call"
                        | Some pc -> sprintf "previous call: %A" pc                        
                    Trace.exprEvaledWithMsg uexpr resHost msg

                previousCall <- Some call

                // synchronize to make sure that CUDA errors occur here
                if Debug.SyncAfterEachCudaCall then
                    try
                        CudaSup.context.Synchronize ()
                    with :? CudaException as ex ->
                        printfn "CUDA exception: %A" ex
                        match previousCall with
                        | Some pc -> printfn "Last call was %A" pc
                        | None -> ()
                        reraise()

        // initialize
        #if !CUDA_DUMMY
        do 
            CudaSup.checkContext ()
            execCalls recipe.InitCalls
        #endif

        // finalizer
        interface System.IDisposable with
            member this.Dispose() = 
                if disposed then raise (System.ObjectDisposedException("CudaExprWorkspace"))

                try 
                    // execute dummy CUDA function to check that CUDA context is not
                    // disposed yet
                    CudaSup.context.PushContext ()
                    CudaSup.context.GetDeviceInfo() |> ignore

                    // cleanup CUDA resources
                    execCalls recipe.DisposeCalls
                    Compile.unloadCudaCode krnlModHndl
                    CudaSup.context.PopContext ()
                with :? System.ObjectDisposedException -> ()

                match cLibHndl, cCompileDir with
                | Some cLibHndl, Some cCompileDir ->
                    Compile.unloadCppCode cLibHndl
                    Compile.removeCompileDir cCompileDir
                | _ -> ()
                Compile.removeCompileDir krnlCompileDir
                disposed <- true

        override this.Finalize() =
            if not disposed then
                (this :> System.IDisposable).Dispose()

        /// Evaluate expression.
        member this.Eval(externalVar: Map<UVarSpecT, IArrayNDT>,
                         hostVar:     Map<UVarSpecT, IArrayNDT>) =
            if disposed then raise (System.ObjectDisposedException("CudaExprWorkspace"))
            CudaSup.checkContext ()

            lock this (fun () ->

                execEnv.ExternalVar <- externalVar |> Map.map (fun _ value -> value :?> IArrayNDCudaT)
                execEnv.HostVar <- hostVar |> Map.map (fun _ value -> value :?> IArrayNDHostT)

                // TODO: implement proper synchronization.
                // For now we synchronize the whole context to make sure that data transfers
                // from and to the GPU do not overlap with the computation that may involve
                // the targets/sources of these transfers as input/output variables.
                CudaSup.context.Synchronize () 
                execCalls recipe.ExecCalls
                CudaSup.context.Synchronize () 

            )


