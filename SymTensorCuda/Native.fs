namespace Basics

open System
open System.Runtime.InteropServices


/// native methods
module Native =

    [<DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Ansi)>]
    extern IntPtr LoadLibrary([<MarshalAs(UnmanagedType.LPStr)>] string dllToLoad)

    [<DllImport("kernel32.dll", CharSet=CharSet.Ansi, ExactSpelling=true, SetLastError=true)>]
    extern IntPtr GetProcAddress(IntPtr hModule, string procedureName)

    [<DllImport("kernel32.dll", SetLastError=true)>]
    extern [<return: MarshalAs(UnmanagedType.Bool)>] bool FreeLibrary(IntPtr hModule)



