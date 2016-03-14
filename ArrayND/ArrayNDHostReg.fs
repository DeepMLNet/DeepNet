namespace ArrayNDNS

open System
open System.Runtime.InteropServices
open ManagedCuda
open ManagedCuda.BasicTypes
open Basics
open ArrayNDHost


module CudaRegMemSupport =
    // TODO: make thread safe

    /// registration count
    let registeredCount = new Dictionary<obj, int>()

    /// master data registrations
    let dataRegistrations = new Dictionary<obj, obj>()

    /// decreases reference count for page locked data
    let decrRefCount (data: IHostStorage) =
        registeredCount.[data] <- registeredCount.[data] - 1
        if registeredCount.[data] = 0 then
            dataRegistrations.Remove[data] |> ignore
            true
        else false


[<AutoOpen>]
module CudaRegMemTypes =

    /// CUDA registered memory for data arrays handle
    type CudaRegMemHnd (data: IHostStorage, pinHnd: IPinnedMemory, cudaMem: CudaRegisteredHostMemory<byte>) =
           
        let mutable disposed = false

        interface IDisposable with
            member this.Dispose() =          
                disposed <- true
                if CudaRegMemSupport.decrRefCount data then            
                    // unregister memory
                    cudaMem.Unregister() 
                    // release cuda memory handle 
                    cudaMem.Dispose()
                    // unpin managed memory
                    pinHnd.Dispose()

        /// the data array
        member this.Data = 
            if disposed then failwith "DataLock is disposed"
            data
        member this.DataPriv = data

        /// GC memory pin handle
        member this.PinHnd = 
            if disposed then failwith "DataLock is disposed"
            pinHnd
        member this.PinHndPriv = pinHnd

        /// the CudaRegisteredHostMemory
        member this.CudaRegisteredMemory = 
            if disposed then failwith "DataLock is disposed"
            cudaMem
        member this.CudaRegisteredMemoryPriv = cudaMem



/// CUDA registered memory for data arrays
module CudaRegMem =
    open CudaRegMemSupport

    /// gets handle for already locked data           
    let get (data: IHostStorage) =      
        if not (dataRegistrations.ContainsKey(data)) then
            failwithf "%A is not registered data" data
        registeredCount.[data] <- registeredCount.[data] + 1
        let dr = dataRegistrations.[data] :?> CudaRegMemHnd
        new CudaRegMemHnd(dr.DataPriv, dr.PinHndPriv, dr.CudaRegisteredMemoryPriv)   
        
    /// gets the CudaRegisteredMemory for already locked data without increment the reference count
    let getCudaRegisteredMemory (data: IHostStorage) =
        if not (dataRegistrations.ContainsKey(data)) then
            failwithf "the specified array is not registered data" 
        let dr = dataRegistrations.[data] :?> CudaRegMemHnd
        dr.CudaRegisteredMemory
            
    /// locks data (multiple registrations are okay) and returns the handle
    let lock (data: IHostStorage) = 
        if dataRegistrations.ContainsKey(data) then get data      
        else
            // pin managed memory so that address cannot change
            let pinHnd = data.Pin ()
            let dataAddr = pinHnd.Ptr
            let dataByteSize = data.SizeInBytes

            // construct cuda memory handle
            let cudaMem = new CudaRegisteredHostMemory<byte>(dataAddr, SizeT(dataByteSize))    
            // register memory
            cudaMem.Register(BasicTypes.CUMemHostRegisterFlags.None)

            // create handle object
            let dr = new CudaRegMemHnd(data, pinHnd, cudaMem)     
            dataRegistrations.[data] <- dr
            registeredCount.[data] <- 1
            dr

    /// unlocks data
    let unlock dataLock =
        (dataLock :> IDisposable).Dispose()


/// Methods for locking an NDArray into memory and registering the memory with CUDA
/// for fast data transfers with GPU device.
module ArrayNDHostReg =
    /// get DataLock for already locked NDArray
    let get (ary: IArrayNDHostT) =
        CudaRegMem.get ary.Storage

    /// Gets the CudaRegisteredMemory for an already locked NDArray.
    /// It becomes invalid if the NDArray is unlocked.
    let getCudaRegisteredMemory (ary: IArrayNDHostT) =
        CudaRegMem.getCudaRegisteredMemory ary.Storage

    /// lock NDArray and get corresponding DataLock
    let lock (ary: IArrayNDHostT) =
        CudaRegMem.lock ary.Storage

    /// unlock given DataLock
    let unlock aryLock =
        CudaRegMem.unlock aryLock



                
