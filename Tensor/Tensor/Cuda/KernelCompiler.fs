namespace Tensor.Cuda

open System
open System.IO
open System.Threading
open System.Reflection
open System.Reflection.Emit
open System.Runtime.InteropServices
open System.Security.Cryptography
open System.Text
open System.Text.RegularExpressions

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor.Utils



/// CUDA module caching key.
type internal ModCacheKey = {
    Code:           string
    HeaderHashes:   Map<string, byte list>
    CompilerArgs:   string list
} with
    member this.Bytes =
        let sb = StringBuilder()
        sb.AppendLine(sprintf "// CompilerArgs=%s" (String.concat " " this.CompilerArgs)) |> ignore
        for KeyValue(k, v) in this.HeaderHashes do
            sb.Append(sprintf "// HeaderHashes[%s]=" k) |> ignore
            for h in v do
                sb.Append(sprintf "%x " h) |> ignore
            sb.AppendLine() |> ignore
        sb.AppendLine(sprintf "// Code=\n%s" this.Code) |> ignore
        Encoding.UTF8.GetBytes(sb.ToString())



/// compiles CUDA C++ code to CUDA kernels.
module internal KernelCompiler =

    let krnlPtxCacheDir = Path.Combine(Util.localAppData "Tensor", "PTXCache")
    let krnlPtxCache = DiskBinaryMap (krnlPtxCacheDir, "code.cu", "mod.ptx")
    let compileDirRoot = Path.Combine(Util.localAppData "Tensor", "Compile")

    /// prepares a compile directory
    let private prepareCompileDir code =        
        // create temp directory
        let rec getTempDir () =  
            let dirname = Path.Combine(compileDirRoot, Path.GetRandomFileName())
            if not (Directory.Exists dirname) then dirname
            else getTempDir()
        let compileDir = getTempDir ()
        Directory.CreateDirectory compileDir |> ignore

        // Get embedded header resources from our assembly.
        // They are in the form: Tensor.Cuda.FILENAME.cuh, i.e. slashes have been replaced by dots.
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
            // extract header file name from resource name
            let filename = Regex.Match(header, @"\.(\w+\.\w+)$").Groups.[1].Value
            let path = Path.Combine (compileDir, filename)
            use fileStrm = File.OpenWrite path
            use strm = asmbly.GetManifestResourceStream(header)
            strm.CopyTo fileStrm)

        // write module code
        let modPath = Path.Combine (compileDir, "mod.cu")
        File.WriteAllText(modPath, code)

        compileDir, modPath, headerHashes

    /// removes a compile directory
    let private removeCompileDir compileDir =
        Directory.Delete(compileDir, true)     

    /// Compiles the given CUDA device code into a CUDA module, loads and jits it and returns
    /// ManagedCuda.CudaKernel objects for the specified kernel names.
    let load modCode krnlNames =
        let compileDir, modPath, headerHashes = prepareCompileDir modCode

        use cmplr = new NVRTC.CudaRuntimeCompiler(modCode, modPath) 
        let baseCmplrArgs = [
            yield "--std=c++11"
            yield "-DWIN32_LEAN_AND_MEAN"
            yield "-Xcudafe"; yield "--diag_suppress=declared_but_not_referenced"
            yield sprintf "--gpu-architecture=%s" Cuda.nvccArch
            if Cfg.FastKernelMath then yield "--use_fast_math"
            if Cfg.RestrictKernels then yield "--restrict"
            if Cfg.DebugCompile then yield "--device-debug"
            if Cfg.DebugCompile then yield "--generate-line-info"
        ] 
        let cmplrArgs =
            baseCmplrArgs @ [sprintf "--include-path=\"%s\"" compileDir]

        let cacheKey = {Code=modCode; HeaderHashes=headerHashes; CompilerArgs=baseCmplrArgs}
        let cacheKeyBytes = cacheKey.Bytes
        let ptx =
            match krnlPtxCache.TryGet cacheKeyBytes with
            | Some ptx when not Cfg.DisableKernelCache -> ptx
            | _ ->
                if Cfg.DebugCompile then
                   printfn "nvrtc %s %s" (cmplrArgs |> String.concat " ") modPath 
                try cmplr.Compile (Array.ofList cmplrArgs)
                with :? NVRTC.NVRTCException as cmplrError ->
                    let log = cmplr.GetLogAsString()
                    let log = log.Replace ("\n\n", "\n")
                    failwithf "nvrtc compile error:\n%s" log
                if Cfg.DebugCompile then
                    let log = cmplr.GetLogAsString()
                    printf "%s" log
                let ptx = cmplr.GetPTX()
                krnlPtxCache.Set cacheKeyBytes ptx                
                ptx    

        if not Cfg.DebugCompile then 
            removeCompileDir compileDir
      
        use jitOpts = new CudaJitOptionCollection()
        use jitInfoBuffer = new CudaJOInfoLogBuffer(10000)
        jitOpts.Add(jitInfoBuffer)
        use jitErrorBuffer = new CudaJOErrorLogBuffer(10000)   
        jitOpts.Add(jitErrorBuffer)
        use jitLogVerbose = new CudaJOLogVerbose(true)
        jitOpts.Add(jitLogVerbose)

        let cuMod = Cuda.context.LoadModulePTX(ptx, jitOpts)

        jitOpts.UpdateValues()
        if Cfg.DebugCompile then
            printfn "%s" jitErrorBuffer.Value
            printfn "%s" jitInfoBuffer.Value   
        jitErrorBuffer.FreeHandle()
        jitInfoBuffer.FreeHandle()

        let krnls =
            (Map.empty, krnlNames)
            ||> Seq.fold (fun krnls name -> 
                krnls |> Map.add name (CudaKernel(name, cuMod, Cuda.context))) 

        krnls, cuMod

    /// unloads previously loaded CUDA kernel code
    let unload cuMod =
        Cuda.context.UnloadModule(cuMod)

    

/// Argument type of a CUDA kernel
type internal KernelArgType = 
    | ArgTypeTensor of NativeTensorInfo
    | ArgTypeIdxTensors of NativeIdxTensorsInfo
    | ArgTypeScalar of Type

/// Argument type of a CUDA kernel
module internal KernelArgType =
    let cppType at =
        match at with
        | ArgTypeTensor nti -> NativeTensor.cppName nti
        | ArgTypeIdxTensors niti -> NativeIdxTensors.cppName niti
        | ArgTypeScalar t -> Cpp.cppTypeInst t

    let mangleName at =
        match at with
        | ArgTypeTensor nti -> NativeTensor.mangledName nti
        | ArgTypeIdxTensors niti -> NativeIdxTensors.mangledName niti
        | ArgTypeScalar t -> Cpp.cppTypeInst t

    let marshal at (av: obj) =
        match at, av with
        | ArgTypeTensor nti, (:? NativeTensor as nt) when NativeTensor.validInstance nti nt ->
            NativeTensor.marshal nt
        | ArgTypeIdxTensors niti, (:? NativeIdxTensors as nit) when NativeIdxTensors.validInstance niti nit ->
            NativeIdxTensors.marshal nit
        | ArgTypeScalar t, av when av.GetType() = t -> 
            if t = typeof<bool> then
                if av :?> bool then box 1uy else box 0uy
            else av
        | _ -> failwithf "cannot marshal %A as %A" av at



/// A CUDA module built from source containing kernel functions.
type internal CudaModule () =

    let wrapperCodes = Dictionary<string, string> ()
    let mutable kernels : Map<string, CudaKernel> option = None
    let mutable cuMod = None

    member this.GetKernel (funcName: string) (argTypes: KernelArgType list) =
        let mangledArgTypes =
            argTypes
            |> List.map KernelArgType.mangleName
            |> String.concat "__"
        let mangledName = funcName + "__" + mangledArgTypes
        let argDeclStr =
            argTypes
            |> List.mapi (fun i typ -> sprintf "%s arg%d" (KernelArgType.cppType typ) i)
            |> String.concat ", "
        let argCallStr = 
            argTypes
            |> List.mapi (fun i _ -> sprintf "arg%d" i)
            |> String.concat ", "
        let declCode =
            sprintf "extern \"C\" __global__ void %s (%s)" mangledName argDeclStr
        let callCode = 
            sprintf "%s (%s);" funcName argCallStr
        let wrapperCode =
            sprintf "%s { %s }\n" declCode callCode
        wrapperCodes.[mangledName] <- wrapperCode

        let argTypes = List.toArray argTypes
        (fun (stream: CUstream, workDim: Cuda.WorkDim, [<ParamArray>] argVals: obj[]) ->
            match kernels with
            | Some kernels ->
                let kernel = kernels.[mangledName]
                let maxBlockSize = kernel.GetOccupancyMaxPotentialBlockSize().blockSize
                let launchDim = Cuda.computeLaunchDim workDim maxBlockSize
                kernel.BlockDimensions <- Cuda.toDim3 launchDim.Block
                kernel.GridDimensions <- Cuda.toDim3 launchDim.Grid
                kernel.DynamicSharedMemory <- 0u

                if argVals.Length <> argTypes.Length then
                    failwithf "incorrect number of arguments for %s" funcName
                let kernelArgs =
                    (argTypes, argVals) ||> Array.map2 KernelArgType.marshal
                kernel.RunAsync (stream, kernelArgs)
            | None -> failwith "CudaModule was not built"
        )

    member this.Build (headers: string list) =
        match kernels with
        | Some _ -> failwith "CudaModule already built"
        | None -> 
            let headerCode = 
                headers 
                |> List.map (fun h -> sprintf "#include \"%s\"" h)
                |> String.concat "\n"
            let wrapperCode =
                wrapperCodes
                |> Seq.map (fun (KeyValue(_, code)) -> code)
                |> String.concat "\n"
            let code = headerCode + "\n" + wrapperCode
            let wrapperNames = 
                wrapperCodes
                |> Seq.map (fun (KeyValue(name, _)) -> name)
            let krnls, m = KernelCompiler.load code wrapperNames
            kernels <- Some krnls
            cuMod <- Some m
            
    override this.Finalize() =
        match cuMod with
        | Some cm ->
            try KernelCompiler.unload cm
            with _ -> ()
            cuMod <- None
        | None -> ()


type internal InstanceCache<'T, 'K when 'K: equality> (create: 'K -> 'T) =
    let cache = Dictionary<'K, 'T>()
    member this.Get (key: 'K) =
        lock cache (fun () ->
            match cache.TryFind key with
            | Some inst -> inst
            | None ->
                let inst = create key
                cache.[key] <- inst
                inst
        )



module internal KernelHelpers =

    /// returns the CUDA work dimensions (x, y, z) for work of given size
    let workDimForWorkSize workSize hetero : Cuda.WorkDim =
        match List.length workSize with
        | _ when hetero -> (List.fold (*) 1L workSize, 1L, 1L)
        | 0 -> (1L, 1L, 1L)
        | 1 -> (workSize.[0], 1L, 1L)
        | 2 -> (workSize.[1], workSize.[0], 1L)
        | 3 -> (workSize.[2], workSize.[1], workSize.[0])
        | d ->
            let rest = {0 .. d-3} |> Seq.map (fun i -> workSize.[i]) |> Seq.fold (*) 1L 
            (workSize.[d-1], workSize.[d-2], rest)

    /// returns the CUDA work dimensions (x, y, z) for an element-wise operation
    let workDimForElemwise (trgt: NativeTensor) =
        workDimForWorkSize trgt.Shape false

    // useful type classifications
    let fpTypes = [typeof<single>; typeof<double>]
    let intTypes = [typeof<int8>; typeof<int16>; typeof<int32>; typeof<int64>
                    typeof<uint8>; typeof<uint16>; typeof<uint32>; typeof<uint64>]
    let numTypes = fpTypes @ intTypes
    let boolTypes = [typeof<bool>]

