namespace Tensor.AssemblyInfo

open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

// General Information about an assembly is controlled through the following 
// set of attributes. Change these attribute values to modify the information
// associated with an assembly.
[<assembly: AssemblyTitle("F# Tensor Library")>]
[<assembly: AssemblyDescription("Tensor library for F#. Provides n-dimensional arrays on host \
                                 and CUDA GPU devices with reshape and slicing functionality.\n\n\
                                 Make sure to set the platform of your project to x64.")>]
[<assembly: AssemblyConfiguration("")>]
[<assembly: AssemblyCompany("Deep.Net developers")>]
[<assembly: AssemblyProduct("Deep.Net")>]
[<assembly: AssemblyCopyright("Copyright © Deep.Net Developers. Licensed under the Apache 2.0 license. \
                               Includes HDF5 binaries that are licensed under the terms specified at
                               https://www.hdfgroup.org/HDF5/doc/Copyright.html")>]
[<assembly: AssemblyTrademark("")>]
[<assembly: AssemblyCulture("")>]

// Setting ComVisible to false makes the types in this assembly not visible 
// to COM components.  If you need to access a type in this assembly from 
// COM, set the ComVisible attribute to true on that type.
[<assembly: ComVisible(false)>]

// The following GUID is for the ID of the typelib if this project is exposed to COM
[<assembly: Guid("82de3bae-bcec-4df8-9c46-07b7faf4e31a")>]

// Version information for an assembly consists of the following four values:
// 
//       Major Version
//       Minor Version 
//       Build Number
//       Revision
// 
// You can specify all the values or you can default the Build and Revision Numbers 
// by using the '*' as shown below:
// [<assembly: AssemblyVersion("1.0.*")>]
[<assembly: AssemblyVersion("0.3.*")>]
[<assembly: AssemblyFileVersion("0.3.*")>]

do
    ()