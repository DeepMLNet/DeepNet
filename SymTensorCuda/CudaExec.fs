namespace SymTensor.Compiler.Cuda

open System.Runtime.InteropServices
open System.IO
open System.Reflection
open System.Security.Cryptography

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

    type ModCacheKey = {Code: string; HeaderHashes: Map<string, byte list>; CompilerArgs: string list}

    let hostCompilerDir = @"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64"

    let krnlPtxCacheDir = Path.Combine(Util.localAppData, "PTXCache")
    let krnlPtxCache = DiskMap<ModCacheKey, byte[]> (krnlPtxCacheDir, "code.dat", "mod.ptx")

    let cppModCacheDir = Path.Combine(Util.localAppData, "CPPCache")
    let cppModCache = DiskMap<ModCacheKey, byte[]> (cppModCacheDir, "code.dat", "mod.dll")

    let compileDirRoot = Path.Combine(Util.localAppData, "Compile")

    /// modification time of C++ header files
    //let headerModTimes =
    //    let includePath = Util.assemblyDirectory
    //    Directory.EnumerateFiles(includePath, "*.cuh")
    //    |> Seq.map (fun headerFile ->
    //        Path.GetFileName headerFile, File.GetLastWriteTimeUtc headerFile)
    //    |> Map.ofSeq

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
   
    /// Compiles the given CUDA device code into a CUDA module, loads and jits it and returns
    /// ManagedCuda.CudaKernel objects for the specified kernel names.
    let loadKernelCode modCode krnlNames =
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
                printfn "nvrtc %s %s" (cmplrArgs |> String.concat " ") modPath 
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
    let loadCppCode modCode (funcDelegates: Map<string, System.Type>)  =
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
    type CudaExprWorkspace(recipe: CudaRecipeT) =
        let mutable disposed = false

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
        let cFuncs, cLibHndl, cCompileDir = Compile.loadCppCode recipe.CPPCode cFuncDelegates
    
        let getStream strm = 
            if Debug.DisableStreams then CUstream.NullStream
            else execEnv.Stream.[strm].Stream

        /// executes the specified calls
        let execCalls calls =
            let mutable previousCall = None
            for call in calls do
                //printfn "CUDA call: %A" call

                match call with 
                // memory management
                | CudaCallT.MemAlloc mem -> 
                    let typeSize = Marshal.SizeOf (TypeName.getType mem.TypeName)
                    let elements = if mem.Elements > 0 then mem.Elements else 1
                    execEnv.InternalMem.Add(mem, new CudaDeviceVariable<byte>(SizeT (elements * typeSize)))
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
                    // WORKAROUND: disposing events currently causes a CUDA access violation
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
                if disposed then raise (System.ObjectDisposedException("CudaExprWorkspace"))

                try 
                    // execute dummy CUDA function to check that CUDA context is not
                    // disposed yet
                    CudaSup.context.GetDeviceInfo() |> ignore
                    execCalls recipe.DisposeCalls
                    Compile.unloadCudaCode krnlModHndl
                with :? System.ObjectDisposedException -> ()

                Compile.unloadCppCode cLibHndl
                Compile.removeCompileDir cCompileDir
                Compile.removeCompileDir krnlCompileDir
                disposed <- true

        override this.Finalize() =
            if not disposed then
                (this :> System.IDisposable).Dispose()

        /// Evaluate expression.
        member this.Eval(externalVar: Map<UVarSpecT, IArrayNDT>,
                         hostVar:     Map<UVarSpecT, IArrayNDT>) =
            if disposed then raise (System.ObjectDisposedException("CudaExprWorkspace"))

            execEnv.ExternalVar <- externalVar |> Map.map (fun _ value -> value :?> IArrayNDCudaT)
            execEnv.HostVar <- hostVar |> Map.map (fun _ value -> value :?> IArrayNDHostT)

            // TODO: implement proper synchronization.
            // For now we synchronize the whole context to make sure that data transfers
            // from and to the GPU do not overlap with the computation that may involve
            // the targets/sources of these transfers as input/output variables.
            CudaSup.context.Synchronize () 
            execCalls recipe.ExecCalls
            CudaSup.context.Synchronize () 




