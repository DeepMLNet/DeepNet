namespace DeepNet.AssemblyInfo

open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

// General Information about an assembly is controlled through the following 
// set of attributes. Change these attribute values to modify the information
// associated with an assembly.
[<assembly: AssemblyTitle("Deep.Net")>]
[<assembly: AssemblyDescription("Deep learning library for F#. Provides symbolic model differentiation, \
                                 automatic differentiation and compilation to CUDA GPUs. Includes optimizers \
                                 and model blocks used in deep learning.\n\n\
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
[<assembly: Guid("7c53d1c7-8ed2-4e04-8f2d-cf0384c51f33")>]

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
[<assembly: AssemblyVersion("0.1.*")>]
[<assembly: AssemblyFileVersion("0.1.*")>]

do
    ()