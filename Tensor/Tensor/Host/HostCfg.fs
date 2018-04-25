namespace Tensor.Host

open System
open System.Threading

open Tensor.Utils

/// <summary>BLAS and LAPACK libraries.</summary>
/// <seealso cref="Cfg.BLAS"/>
[<RequireQualifiedAccess>]
type BLASLib =
    /// <summary>Vendor BLAS and LAPACK.</summary>
    /// <remarks>This uses the system BLAS and LAPACK libraries by loading the shared libaries with the following names.
    /// Windows: blas.dll and lapacke.dll.
    /// Linux: libblas.so and liblapacke.so.
    /// Mac OS: libblas.dylib and liblapacke.dylib.
    /// </remarks>
    | Vendor
    /// <summary>Intel MKL BLAS and LAPACK (shipped in NuGet package)</summary>
    | IntelMKL
    /// <summary>Custom BLAS and LAPACK libraries.</summary>
    /// <param name="blas">Name of BLAS native library.</param>
    /// <param name="lapack">Name of LAPACK native library.</param>
    | Custom of blas:NativeLibName * lapack:NativeLibName


/// <summary>Options for configuring operations performed on hosts tensors.</summary>
/// <seealso cref="HostTensor"/>
type Cfg private () = 

    static let mutable blasLib : BLASLib = BLASLib.IntelMKL

    static let blasLibChangedEvent = new Event<_>()
    
    /// <summary>The BLAS and LAPACK library to use.</summary>
    /// <remarks>
    /// <para>This setting is global to the all threads.</para>
    /// <para>The default BLAS library is Intel MKL (shipped in NuGet package).</para>
    /// </remarks>
    static member BLASLib
        with get() = blasLib
        and set(v) = 
            blasLib <- v    
            blasLibChangedEvent.Trigger (blasLib)

    /// <summary>BLAS library was changed.</summary>
    [<CLIEvent>]
    static member internal BLASLibChangedEvent = blasLibChangedEvent.Publish

