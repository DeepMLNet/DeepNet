module NDArrayCuda

open System
open System.Runtime.InteropServices
open ManagedCuda
open CudaBasics
open CudaRecipe
open NDArray


[<DllImport("kernel32.dll", SetLastError=true)>]
extern bool VirtualLock(IntPtr lpAddress, UIntPtr dwSize)

[<DllImport("kernel32.dll", SetLastError=true)>]
extern bool VirtualUnlock(IntPtr lpAddress, UIntPtr dwSize)

let getLastWin32ErrText () =
    let we = System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error())
    we.Message

/// constructs a CudaPageLockedHostMemory object for the given array
type PageLockedData<'a when 'a: (new: unit -> 'a) and 'a: struct and 'a:> ValueType> (data: 'a array) =
    // pin managed memory so that address cannot change
    let pinnedDataHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
    let dataAddr = pinnedDataHnd.AddrOfPinnedObject ()
    let dataElements = Array.length data
    let dataByteSize = dataElements * sizeof<'a>

    do
        // lock memory
        if not (VirtualLock(dataAddr, UIntPtr (uint64 dataByteSize))) then
            failwithf "VirtualLock failed: %s" (getLastWin32ErrText())

    // construct cuda memory handle
    let cudaMem = new CudaPageLockedHostMemory<'a>(dataAddr, BasicTypes.SizeT(dataElements))

    interface IDisposable with
        member this.Dispose() =            
            // release cuda memory handle
            cudaMem.Dispose()
            // unlock memory
            if not (VirtualUnlock(dataAddr, UIntPtr (uint64 dataByteSize))) then
                failwithf "VirtualUnlock failed: %s" (getLastWin32ErrText())
            // unpin managed memory
            pinnedDataHnd.Free()

    /// the data array
    member this.Data = data

    /// the CudaPageLockedHostMemory
    member this.CudaMem = cudaMem


let lockedNDArray (a: NDArray) =
    new PageLockedData<single>(a.Data)


type PageLockedVarEnv (varEnv: OpEval.VarEnvT) =
    
    let pageLockedDatas =     
        varEnv 
        |> Map.toList
        |> List.map (fun (name, ary) -> {HostExternalMemT.Name = name}, lockedNDArray ary)

    let hostMem = 
        pageLockedDatas
        |> List.map (fun (emem, pld) -> emem, pld.CudaMem)
        |> Map.ofList

    member this.HostMem = hostMem

    interface IDisposable with
        member this.Dispose () =
            for _, pld in pageLockedDatas do
                (pld :> IDisposable).Dispose()

                




