module CudaRegMem

open System
open System.Runtime.InteropServices
open ManagedCuda
open Util
open NDArray


/// CUDA registered memory for data arrays
module DataLock =

    // TODO: make thread safe

    /// registration count
    let registeredCount = new Dictionary<obj, int>()

    /// master data registrations
    let dataRegistrations = new Dictionary<obj, obj>()

    /// decreases reference count for page locked data
    let decrRefCount<'a> (data: 'a) =
        registeredCount.[data] <- registeredCount.[data] - 1
        if registeredCount.[data] = 0 then
            dataRegistrations.Remove[data] |> ignore
            true
        else false

    /// a data lock handle
    type DataLockT<'a when 'a: (new: unit -> 'a) and 'a: struct and 'a:> ValueType> 
        (data: obj, pinHnd: GCHandle, cudaMem: CudaRegisteredHostMemory<'a>) =
           
        let mutable disposed = false

        interface IDisposable with
            member this.Dispose() =          
                disposed <- true
                if decrRefCount data then            
                    // unregister memory
                    cudaMem.Unregister() 
                    // release cuda memory handle 
                    cudaMem.Dispose()
                    // unpin managed memory
                    pinHnd.Free()

        /// the data array
        member this.Data = 
            if disposed then failwith "DataLock is disposed"
            data

        member this.Data_ = data

        /// GC memory pin handle
        member this.PinHnd = 
            if disposed then failwith "DataLock is disposed"
            pinHnd

        member this.PinHnd_ = pinHnd

        /// the CudaRegisteredHostMemory
        member this.CudaRegisteredMemory = 
            if disposed then failwith "DataLock is disposed"
            cudaMem

        member this.CudaRegisteredMemory_ = cudaMem

    /// gets handle for already locked data           
    let get<'a when 'a: (new: unit -> 'a) and 'a: struct and 'a:> ValueType> (data: 'a array) =      
        if not (dataRegistrations.ContainsKey(data)) then
            failwithf "%A is not registered data" data
        registeredCount.[data] <- registeredCount.[data] + 1
        let dr = dataRegistrations.[data] :?> DataLockT<'a>
        new DataLockT<'a>(dr.Data_, dr.PinHnd_, dr.CudaRegisteredMemory_)   
        
    /// gets the CudaRegisteredMemory for already locked data without increment the reference count
    let getCudaRegisteredMemory<'a when 'a: (new: unit -> 'a) and 'a: struct and 'a:> ValueType> (data: 'a array) =
        if not (dataRegistrations.ContainsKey(data)) then
            failwithf "%A is not registered data" data
        let dr = dataRegistrations.[data] :?> DataLockT<'a>
        dr.CudaRegisteredMemory
            
    /// locks data (multiple registrations are okay) and returns the handle
    let lock<'a when 'a: (new: unit -> 'a) and 'a: struct and 'a:> ValueType> (data: 'a array) = 
        if dataRegistrations.ContainsKey(data) then get data      
        else
            // pin managed memory so that address cannot change
            let pinHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
            let dataAddr = pinHnd.AddrOfPinnedObject ()
            let dataElements = Array.length data
            let dataByteSize = dataElements * sizeof<'a>

            // construct cuda memory handle
            let cudaMem = new CudaRegisteredHostMemory<'a>(dataAddr, BasicTypes.SizeT(dataByteSize))    
            // register memory
            cudaMem.Register(BasicTypes.CUMemHostRegisterFlags.None)

            let dr = new DataLockT<'a>(data, pinHnd, cudaMem)     
            dataRegistrations.[data] <- dr
            dr

    /// unlocks data
    let unlock dataLock =
        (dataLock :> IDisposable).Dispose()


/// Methods for locking an NDArray into memory and registering the memory with CUDA
/// for fast data transfers with GPU device.
module NDArrayLock =
    /// get DataLock for already locked NDArray
    let get (ary: NDArray) =
        DataLock.get ary.Data

    /// Gets the CudaRegisteredMemory for an already locked NDArray.
    /// It becomes invalid if the NDArray is unlocked.
    let getCudaRegisteredMemory (ary: NDArray) =
        DataLock.getCudaRegisteredMemory ary.Data

    /// lock NDArray and get corresponding DataLock
    let lock (ary: NDArray) =
        DataLock.lock ary.Data

    /// unlock given DataLock
    let unlock aryLock =
        DataLock.unlock aryLock



/// Locks all variables in a VarEnv.
type VarEnvLock (varEnv: OpEval.VarEnvT) =   
    let varLocks =
        varEnv
        |> Map.toList
        |> List.map (fun (name, ary) -> NDArrayLock.lock ary)

    interface IDisposable with
        member this.Dispose () =
            for lck in varLocks do
                (lck :> IDisposable).Dispose()

                
