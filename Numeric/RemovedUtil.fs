
open System
open System.Reflection
open System.IO
open System.Runtime.InteropServices
open System.Collections.Concurrent
open FSharp.Reflection






module internal Seq = 

    /// every n-th element of the given sequence
    let everyNth n (input:seq<_>) = 
      seq { use en = input.GetEnumerator()
            // Call MoveNext at most 'n' times (or return false earlier)
            let rec nextN n = 
              if n = 0 then true
              else en.MoveNext() && (nextN (n - 1)) 
            // While we can move n elements forward...
            while nextN n do
              // Retrun each nth element
              yield en.Current }

    /// shuffles a finite sequence using the given seed
    let shuffle seed (input:seq<_>) =
        let rand = System.Random seed

        let swap (a: _[]) x y =
            let tmp = a.[x]
            a.[x] <- a.[y]
            a.[y] <- tmp

        let a = Array.ofSeq input
        Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a        
        a |> Seq.ofArray

    /// sequence counting from given value to infinity
    let countingFrom from = seq {
        let mutable i = from
        while true do
            yield i
            i <- i + 1
    }

    /// sequence counting from zero to infinity
    let counting = countingFrom 0



module internal List =
    /// removes all elements with the given value
    let withoutValue value lst =
        lst |> List.filter ((<>) value)


    /// removes the first element with the given value
    let rec removeValueOnce value lst =
        match lst with
        | v::vs when v = value -> vs
        | v::vs -> v :: removeValueOnce value vs
        | [] -> []
        
    /// transposes a list of lists
    let rec transpose = function
        | (_::_)::_ as m -> List.map List.head m :: transpose (List.map List.tail m)
        | _ -> []

    /// sets element with index elem to given value
    let rec set elem value lst =
        match lst, elem with
            | l::ls, 0 -> value::ls
            | l::ls, _ -> l::(set (elem-1) value ls)
            | [], _ -> invalidArg "elem" "element index out of bounds"

    /// swaps the elements at the specified positions
    let swap elem1 elem2 lst =
        lst
        |> set elem1 lst.[elem2]
        |> set elem2 lst.[elem1]

    /// Elementwise addition of two list. If one list is shorter, it is padded with zeros.
    let inline addElemwise a b =
        let rec build a b =
            match a, b with
            | av::ra, bv::rb -> (av+bv)::(build ra rb)
            | av::ra, [] -> av::(build ra [])
            | [], bv::rb -> bv::(build [] rb)
            | [], [] -> []
        build a b

    /// Elementwise summation of lists. All lists are padded to the same size with zeros.
    let inline sumElemwise (ls: 'a list seq) =
        ([], ls) ||> Seq.fold addElemwise




module internal Map = 

    /// adds all items from q to p
    let join (p:Map<'a,'b>) (q:Map<'a,'b>) = 
        Map(Seq.concat [(Map.toSeq p); (Map.toSeq q)])    

    /// merges two maps by adding the values of the same key
    let inline addSameKeys (p:Map<'a,'b>) (q:Map<'a,'b>) = 
        (p, q) ||> Map.fold (fun res key value ->
            match res |> Map.tryFind key with
            | Some oldValue -> res |> Map.add key (oldValue + value)
            | None -> res |> Map.add key value)

    /// merges two maps by summing the values of the same key      
    let inline sum (ps:Map<'a,'b> seq) =
        (Map.empty, ps) ||> Seq.fold addSameKeys

    /// Creates a map from a System.Collection.Generic.Dictionary<_,_>.
    let ofDictionary dictionary = 
        (dictionary :> seq<_>)
        |> Seq.map (|KeyValue|)
        |> Map.ofSeq    




module internal String =

    /// combines sequence of string with given seperator but returns empty if sequence is empty
    let concatButIfEmpty empty sep items =
        if Seq.isEmpty items then empty
        else String.concat sep items

    /// object x converted to a string and capped to a maximum length
    let truncObj x =
        let maxLen = 80
        let s = sprintf "%A" x
        let s = s.Replace ("\n", " ")
        if String.length s > maxLen then s.[0..maxLen-3-1] + "..."
        else s


module internal Array2D =

    /// returns a transposed copy of the matrix
    let transpose m = 
        Array2D.init (Array2D.length2 m) (Array2D.length1 m) (fun y x -> m.[x, y])





[<AutoOpen>]
module internal Extensions =

    type System.Collections.Generic.HashSet<'T> with
        member this.LockedContains key =
            lock this (fun () -> this.Contains key)

        member this.LockedAdd key =
            lock this (fun () -> this.Add key)





    type System.Collections.Generic.IReadOnlyDictionary<'TKey, 'TValue> with
        member this.TryFindReadOnly key =
            let value = ref (Unchecked.defaultof<'TValue>)
            if this.TryGetValue (key, value) then Some !value
            else None

    type System.Collections.Generic.Queue<'T> with
        member this.TryPeek =
            if this.Count > 0 then Some (this.Peek())
            else None



module Util =

    /// Default value for options. Returns b if a is None, else the value of a.
    let inline (|?) (a: 'a option) b = 
        match a with
        | Some v -> v
        | None -> b

    /// all BindingFlags, i.e. public and non-public, static and instance
    let allBindingFlags = 
        BindingFlags.Public ||| BindingFlags.NonPublic ||| 
        BindingFlags.Static ||| BindingFlags.Instance

    /// Compares two objects of possibly different types.
    let compareObjs (this: 'A when 'A :> System.IComparable<'A>) (other: obj) =
        if this.GetType() = other.GetType() then
            (this :> System.IComparable<'A>).CompareTo (other :?> 'A)
        else compare (this.GetType().FullName) (other.GetType().FullName)

    /// iterates function f n times
    let rec iterate f n x =
        match n with
        | 0 -> x
        | n when n > 0 -> iterate f (n-1) (f x)
        | _ -> failwithf "cannot execute negative iterations %d" n

    /// Returns the contents of a blittable structure as a byte array.
    let structToBytes (s: 'S when 'S: struct) =
        let size = Marshal.SizeOf(typeof<'S>)
        let byteAry : byte[] = Array.zeroCreate size

        let tmpPtr = Marshal.AllocHGlobal(size)
        Marshal.StructureToPtr(s, tmpPtr, false)
        Marshal.Copy(tmpPtr, byteAry, 0, size)
        Marshal.DestroyStructure(tmpPtr, typeof<'S>)
        Marshal.FreeHGlobal(tmpPtr)

        byteAry


    /// Returns "Some key" when a key was pressed, otherwise "None".
    let getKey () =
        try
            if Console.KeyAvailable then Some (Console.ReadKey().KeyChar)
            else None
        with :? InvalidOperationException -> 
            // InvalidOperationException is thrown when process does not have a console or 
            // input is redirected from a file.
            None

    /// matches integral values (e.g. 2, 2.0 or 2.0f, etc.)
    let (|Integral|_|) (x: 'T) =
        match typeof<'T> with
        | t when t = typeof<int> ->
            Some (x |> box |> unbox<int>)
        | t when t = typeof<byte> ->
            Some (x |> box |> unbox<byte> |> int)
        | t when t = typeof<float> ->
            let f = x |> box |> unbox<float>
            if abs (f % 1.0) < System.Double.Epsilon then
                Some (f |> round |> int)
            else None
        | t when t = typeof<single> ->
            let f = x |> box |> unbox<single>
            if abs (f % 1.0f) < System.Single.Epsilon then
                Some (f |> round |> int)
            else None
        | _ -> None


    /// part of a limited string length format specifier
    type LimitedStringPart =
        /// a delimiter that is always inserted
        | Delim of string
        /// a formatter that is replaced by "..." if string becomes too long
        | Formatter of (int -> string)

    /// builds a string of approximate maximum length from the given format specifier parts
    let limitedToString maxLength parts =
        ("", parts)
        ||> List.fold (fun s p ->
            match p with
            | Delim d -> s + d
            | Formatter fmtFn -> 
                match maxLength - s.Length with
                | remLength when remLength > 0 -> s + fmtFn remLength
                | _ -> s + "..."
            )
                       

    /// directory of our assembly
    let assemblyDirectory = 
        // http://stackoverflow.com/questions/52797/how-do-i-get-the-path-of-the-assembly-the-code-is-in
        let codeBase = System.Reflection.Assembly.GetExecutingAssembly().CodeBase
        let uri = new System.UriBuilder(codeBase)
        let path = System.Uri.UnescapeDataString(uri.Path)
        System.IO.Path.GetDirectoryName(path)

    /// true if running on Windows
    let onWindows =
        System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(OSPlatform.Windows)

    /// true if running on Linux
    let onLinux =
        System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(OSPlatform.Linux)


    [<Flags>]
    type ErrorModes = 
        | SYSTEM_DEFAULT = 0x0
        | SEM_FAILCRITICALERRORS = 0x0001
        | SEM_NOALIGNMENTFAULTEXCEPT = 0x0004
        | SEM_NOGPFAULTERRORBOX = 0x0002
        | SEM_NOOPENFILEERRORBOX = 0x8000

    [<DllImport("kernel32.dll")>]
    extern ErrorModes private SetErrorMode(ErrorModes mode)

    /// disables the Windows WER dialog box on crash of this application
    let disableCrashDialog () =
        if onWindows then        
            SetErrorMode(ErrorModes.SEM_NOGPFAULTERRORBOX |||
                         ErrorModes.SEM_FAILCRITICALERRORS |||
                         ErrorModes.SEM_NOOPENFILEERRORBOX)
            |> ignore



module Cuda =

    open ManagedCuda
    open ManagedCuda.BasicTypes

    /// CUDA context
    let context = 
        let cudaCntxt = 
            try
                new CudaContext(createNew=false)
            with e ->
                failwithf "Cannot create CUDA context: %s" e.Message
        cudaCntxt

    /// CUDA device info
    let deviceInfo =
        context.GetDeviceInfo()

    /// prints CUDA info
    let printInfo () =
        let di = deviceInfo
        printfn "CUDA device:                                         %s" di.DeviceName
        printfn "CUDA driver version:                                 %A" di.DriverVersion
        printfn "CUDA device global memory:                           %A bytes" di.TotalGlobalMemory
        printfn "CUDA device free memory:                             %A bytes" (context.GetFreeDeviceMemorySize())
        printfn "CUDA device compute capability:                      %A" di.ComputeCapability
        printfn "CUDA device maximum block size:                      %A" di.MaxThreadsPerBlock                       
        printfn "CUDA device maximum block dimensions:                %A" di.MaxBlockDim
        printfn "CUDA device maximum grid dimensions:                 %A" di.MaxGridDim    
        printfn "CUDA device async engine count:                      %d" di.AsyncEngineCount
        printfn "CUDA device can execute kernels concurrently:        %A" di.ConcurrentKernels
        printfn "CUDA device can overlap kernels and memory transfer: %A" di.GpuOverlap

    /// prints short CUDA device information
    let printDevice () =
        let di = deviceInfo
        printfn "Using CUDA device \"%s\" with %d multiprocessors @ %.2f GHz and %d MB memory." 
            di.DeviceName di.MultiProcessorCount 
            (float di.ClockRate / 10.0**6.0) (int64 di.TotalGlobalMemory / pown 2L 20)

    /// Ensures that CUDA is initialized. Multiple calls are allowed and have no effect.
    let init () =       
        // make a dummy call on the context to ensure that it is created
        context.GetSharedMemConfig() |> ignore

    /// shutsdown CUDA (necessary for correct profiler results)  
    let shutdown () =
        context.Synchronize ()
        CudaContext.ProfilerStop ()
        context.Synchronize ()
        blas.Dispose ()
        context.Dispose ()

    /// Checks that the thread's current CUDA context is the CUDA context that was active
    /// or created while this module was initialized.
    let checkContext () =
        let ctx = ref (CUcontext ())
        if DriverAPINativeMethods.ContextManagement.cuCtxGetCurrent (ctx) <> CUResult.Success then
            failwith "cuCtxGetCurrent failed"
        if context.Context <> (!ctx) then
            failwithf "Current CUDA context %A does not match library initialization CUDA context %A"
                (!ctx).Pointer context.Context.Pointer
                        