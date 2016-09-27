namespace ArrayNDNS

open System
open System.Runtime.InteropServices
open ManagedCuda
open ManagedCuda.BasicTypes
open Basics
open ArrayNDHost


module internal CudaRegMemSupport =

    /// synchronization lock
    let internal syncLock = obj ()

    /// registration count
    let internal registeredCount = new Dictionary<IArrayNDHostT, int>()

    /// master data registrations
    let internal dataRegistrations = new Dictionary<IArrayNDHostT, obj>()

    /// decreases reference count for page locked data
    let internal decrRefCount data  =
        registeredCount.[data] <- registeredCount.[data] - 1
        if registeredCount.[data] = 0 then
            dataRegistrations.Remove data |> ignore
            true
        else false


[<AutoOpen>]
/// CUDA registered memory for data arrays handle
module CudaRegMemTypes =

    /// cannot register memory, probably because it is not properly aligned
    exception CannotRegisterMemory

    /// CUDA registered memory for data arrays handle
    type CudaRegMemHnd (hostArray:  IArrayNDHostT, 
                        pinHnd:     PinnedMemoryT, 
                        cudaMem:    CudaRegisteredHostMemory<byte>) =
           
        let mutable disposed = false

        interface IDisposable with
            member this.Dispose() =          
                lock CudaRegMemSupport.syncLock (fun () ->
                    if not disposed then 
                        if CudaRegMemSupport.decrRefCount hostArray then            
                            // unregister memory
                            cudaMem.Unregister() 
                            // release cuda memory handle 
                            cudaMem.Dispose()
                            // unpin managed memory
                            (pinHnd :> IDisposable).Dispose()
                    disposed <- true
                )

        override this.Finalize () =
            (this :> IDisposable).Dispose()

        /// the data array
        member this.HostArray = 
            if disposed then failwith "CudaRegMemHnd is disposed"
            hostArray
        member internal this.HostArrayPriv = hostArray

        /// GC memory pin handle
        member this.PinHnd = 
            if disposed then failwith "CudaRegMemHnd is disposed"
            pinHnd
        member internal this.PinHndPriv = pinHnd

        /// pointer to data 
        member this.Ptr =
            this.PinHnd.Ptr

        /// the CudaRegisteredHostMemory
        member this.CudaRegisteredMemory = 
            if disposed then failwith "CudaRegMemHnd is disposed"
            cudaMem
        member internal this.CudaRegisteredMemoryPriv = cudaMem


/// Methods for locking an NDArray into memory and registering the memory with CUDA
/// for fast data transfers with GPU device.
module ArrayNDHostReg =
    open CudaRegMemSupport

    /// get CudaRegMemHnd for already locked ArrayNDHostT          
    let get data =      
        lock CudaRegMemSupport.syncLock (fun () ->
            if not (dataRegistrations.ContainsKey data) then
                failwith "the specified tensor is not registered with CUDA for data transfer" 
            registeredCount.[data] <- registeredCount.[data] + 1
            let dr = dataRegistrations.[data] :?> CudaRegMemHnd
            new CudaRegMemHnd(dr.HostArrayPriv, dr.PinHndPriv, dr.CudaRegisteredMemoryPriv)   
        )
        
    /// gets the CudaRegisteredMemory for already locked ArrayNDHostT without 
    /// incrementing the reference count
    let getCudaRegisteredMemory data =
        lock CudaRegMemSupport.syncLock (fun () ->
            if not (dataRegistrations.ContainsKey data) then
                failwith "the specified tensor is not registered with CUDA for data transfer" 
            let dr = dataRegistrations.[data] :?> CudaRegMemHnd
            dr.CudaRegisteredMemory
        )            

    /// locks ArrayNDHostT (multiple registrations are okay) and returns the corresponding CudaRegMemHnd
    let lock (data: IArrayNDHostT) = 
        lock CudaRegMemSupport.syncLock (fun () ->
            if dataRegistrations.ContainsKey data then get data      
            else
                // pin managed memory so that address cannot change
                let pinHnd = data.Pin ()
                let dataAddr = pinHnd.Ptr
                let dataByteSize = data.DataSizeInBytes

                // construct cuda memory handle and register it
                let cudaMem = new CudaRegisteredHostMemory<byte>(dataAddr, SizeT dataByteSize)    
                try
                    cudaMem.Register(BasicTypes.CUMemHostRegisterFlags.None)
                with :? CudaException as ex ->
                    if ex.CudaError = CUResult.ErrorInvalidValue then
                        /// probably memory is not properly aligned
                        raise CannotRegisterMemory
                    else reraise ()

                // create handle object
                let dr = new CudaRegMemHnd(data, pinHnd, cudaMem)     
                dataRegistrations.[data] <- dr
                registeredCount.[data] <- 1
                dr
        )

    /// unlocks a ArrayNDHostT
    let unlock dataLock =
        (dataLock :> IDisposable).Dispose()




                
