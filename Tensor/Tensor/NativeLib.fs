namespace Tensor.Utils

open System
open System.Runtime.InteropServices
open System.Linq.Expressions
open System.IO


module private OSLoader =

    let unsupPlatform () = raise (PlatformNotSupportedException "NativeLib does not support this platform.")

    module private Unix =

        let RTLD_LAZY = nativeint 1
        let RTLD_NOW = nativeint 2

        [<DllImport("dl")>]
        extern IntPtr dlopen (string filename, nativeint flags)

        [<DllImport("dl")>]
        extern nativeint dlclose (IntPtr handle)

        [<DllImport("dl")>]
        extern IntPtr dlsym (IntPtr handle, string symbol)

        [<DllImport("dl")>]
        extern IntPtr dlerror ()

        let dlerrorString () =
            dlerror () |> Marshal.PtrToStringAnsi 

    module private Windows =

        [<DllImport("kernel32", SetLastError=true)>]
        extern IntPtr LoadLibraryEx (string lpFileName, IntPtr hReservedNull, nativeint dwFlags)

        [<DllImport("kernel32", SetLastError=true)>]
        extern bool FreeLibrary (IntPtr hModule)

        [<DllImport("kernel32", SetLastError=true)>]
        extern IntPtr GetProcAddress (IntPtr hModule, string procName)   

    let load (filename: string) =
        if RuntimeInformation.IsOSPlatform OSPlatform.Linux || RuntimeInformation.IsOSPlatform OSPlatform.OSX then
            let hnd = Unix.dlopen (filename, Unix.RTLD_NOW)
            if hnd <> IntPtr.Zero then Ok hnd
            else Error (Unix.dlerrorString ())
        elif RuntimeInformation.IsOSPlatform OSPlatform.Windows then
            let hnd = Windows.LoadLibraryEx (filename, IntPtr.Zero, nativeint 0)
            if hnd <> IntPtr.Zero then Ok hnd
            else Error (sprintf "HResult: %d" (Marshal.GetHRForLastWin32Error()))
        else
            unsupPlatform ()

    let free (hnd: IntPtr) =
        if RuntimeInformation.IsOSPlatform OSPlatform.Linux || RuntimeInformation.IsOSPlatform OSPlatform.OSX then
            Unix.dlclose (hnd) |> ignore
        elif RuntimeInformation.IsOSPlatform OSPlatform.Windows then
            Windows.FreeLibrary (hnd) |> ignore
        else
            unsupPlatform ()
        
    let getAddress (hnd: IntPtr) (name: string) =
        if RuntimeInformation.IsOSPlatform OSPlatform.Linux || RuntimeInformation.IsOSPlatform OSPlatform.OSX then
            let ptr = Unix.dlsym (hnd, name)
            if ptr <> IntPtr.Zero then Ok ptr
            else Error (Unix.dlerrorString ())           
        elif RuntimeInformation.IsOSPlatform OSPlatform.Windows then
            let ptr = Windows.GetProcAddress (hnd, name)
            if ptr <> IntPtr.Zero then Ok ptr
            else Error (sprintf "HResult: %d" (Marshal.GetHRForLastWin32Error()))            
        else
            unsupPlatform ()


/// The native library could not be loaded.
exception NativeLibNotLoadable of filename:string * msg:string with
    override __.Message = sprintf "Native library %s could not be loaded: %s" __.filename __.msg

/// The specified symbol was not found in the native library.
exception SymbolNotFound of filename:string * symbol:string * msg:string with
    override __.Message = sprintf "Symbol %s could not be found in native library %s: %s" __.symbol __.filename __.msg


/// <summary>Specifies a native library name.</summary>
[<RequireQualifiedAccess>]
type NativeLibName =
    /// <summary>The exact name is passed to the dynamic loader.</summary>
    | Exact of string
    /// <summary>The library name is translated in an OS-specific way.</summary>
    /// <remarks>If the specified name is <c>XXX</c> then Linux uses
    /// <c>libXXX.so</c>, Mac OS uses <c>libXXX.dylib</c> and Windows uses <c>XXX.dll</c></remarks>
    | Translated of string
    /// <summary>The library name is translated in an OS-specific way and searched for in the specific
    /// NuGet package directories.</summary>
    | Packaged of string


/// <summary>A native library.</summary>
/// <param name="libName">The name of the native library.</param>
/// <exception cref="NativeLibNotLoadable">The native library could not be loaded.</exception>
type NativeLib (libName: NativeLibName) =

    let translate name = 
        if RuntimeInformation.IsOSPlatform OSPlatform.Linux then sprintf "lib%s.so" name
        elif RuntimeInformation.IsOSPlatform OSPlatform.OSX then sprintf "lib%s.dylib" name
        elif RuntimeInformation.IsOSPlatform OSPlatform.Windows then sprintf "%s.dll" name
        else OSLoader.unsupPlatform ()
            
    let resolve filename =
        let rid =
            if RuntimeInformation.IsOSPlatform OSPlatform.Linux then "linux-x64"
            elif RuntimeInformation.IsOSPlatform OSPlatform.OSX then "osx-x64"
            elif RuntimeInformation.IsOSPlatform OSPlatform.Windows then "win-x64"
            else OSLoader.unsupPlatform ()
        let cands = 
            [yield Path.Combine (Util.assemblyDir, filename)
             yield Path.Combine (Util.assemblyDir, "..", "..", "runtimes", rid, "native", filename)]
        match cands |> List.tryFind File.Exists with
        | Some path -> 
            //printfn "Resolved library %s to %s." filename path
            path
        | None ->
            let msg = sprintf "Cannot resolve library path for %s. Candidates are %A." filename cands
            raise (NativeLibNotLoadable (filename, msg))       

    let filename =
        match libName with
        | NativeLibName.Exact filename -> filename
        | NativeLibName.Translated name -> translate name
        | NativeLibName.Packaged name -> name |> translate |> resolve

    let hnd = 
        match OSLoader.load filename with
        | Ok hnd -> hnd
        | Error msg -> raise (NativeLibNotLoadable (filename, msg))

    interface IDisposable with
        /// <summary>Frees the native library.</summary>
        /// <remarks>
        /// <para>This will crash your program if you use the obtained delegates after freeing the library.
        /// Also, the library might spawn threads, which will then crash your program, even if
        /// you do not perform any function call into the library.</para>
        /// <para>It is usually best to not free the library.</para>
        /// </remarks>
        member __.Dispose () =
            OSLoader.free hnd

    /// <summary>Get delegate to native function.</summary>
    /// <typeparam name="'F">Delegate type of the native function.</typeparam>
    /// <param name="symbol">The symbol name.</param>
    /// <returns>A delegate to the native function or <c>Error msg</c> if the function does not exist.</returns>
    member __.TryFunc<'F> (symbol: string) =
        match OSLoader.getAddress hnd symbol with
        | Ok ptr -> Marshal.GetDelegateForFunctionPointer<'F> (ptr) |> Ok
        | Error msg -> Error msg

    /// <summary>Get delegate to native function.</summary>
    /// <typeparam name="'F">Delegate type of the native function.</typeparam>
    /// <param name="symbol">The symbol name.</param>
    /// <returns>A delegate to the native function.</returns>
    /// <exception cref="SymbolNotFound">The specified symbol was not found in the library.</exception>
    member this.Func<'F> (symbol: string) =
        match this.TryFunc<'F> symbol with
        | Ok f -> f
        | Error msg -> raise (SymbolNotFound (filename, symbol, msg))

    /// <summary>Get delegate to native function failing at invocation if function does not exists.</summary>
    /// <typeparam name="'F">Delegate type of the native function.</typeparam>
    /// <param name="symbol">The symbol name.</param>
    /// <returns>A delegate to the native function.</returns>
    /// <exception cref="SymbolNotFound">The specified symbol was not found in the library.</exception>
    member this.LazyFunc<'F> (symbol: string) =
        match this.TryFunc<'F> symbol with
        | Ok f -> f
        | Error msg -> 
            let func = typeof<'F>.GetMethod("Invoke")
            let pars = func.GetParameters() |> Seq.map (fun p -> Expression.Parameter p.ParameterType)           
            let thrw = Expression.Throw(Expression.Constant(SymbolNotFound (filename, symbol, msg)))
            let expr = Expression.Lambda<'F>(thrw, pars)
            expr.Compile()

    /// <summary>Checks if the native function exists in the library.</summary>
    /// <param name="symbol">The symbol name.</param>
    /// <returns>true if the functions exists; false otherwise.</returns>
    member __.HasFunc (symbol: string) =
        match OSLoader.getAddress hnd symbol with
        | Ok _ -> true
        | Error _ -> false
