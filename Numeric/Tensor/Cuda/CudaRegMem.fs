namespace Tensor.Cuda

open System
open System.IO
open System.Threading
open System.Reflection
open System.Reflection.Emit
open System.Runtime.InteropServices
open System.Security.Cryptography
open System.Text
open System.Text.RegularExpressions

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor.Utils
open Tensor.Host


/// cannot register host memory with CUDA, maybe because it is not properly aligned
exception CannotCudaRegisterMemory of msg:string with override __.Message = __.msg



/// CUDA registered memory support
module internal CudaRegMemSupport =

    /// synchronization lock
    let syncLock = obj ()

    /// registration count
    let registeredCount = new Dictionary<ITensorHostStorage, int>()

    /// master data registrations
    let dataRegistrations = new Dictionary<ITensorHostStorage, obj>()

    /// decreases reference count for page locked data
    let decrRefCount data  =
        registeredCount.[data] <- registeredCount.[data] - 1
        if registeredCount.[data] = 0 then
            dataRegistrations.Remove data |> ignore
            true
        else false


/// CUDA registered memory for fast data transfer.
/// Dispose to unregister memory with CUDA.
type CudaRegMemHnd internal (hostArray:  ITensorHostStorage, 
                             pinHnd:     PinnedMemory, 
                             cudaMem:    CudaRegisteredHostMemory<byte>) =
           
    let mutable disposed = false
    let checkDisposed () =
        if disposed then raise (ObjectDisposedException "CudaRegMemHnd")

    interface IDisposable with
        member this.Dispose() =          
            lock CudaRegMemSupport.syncLock (fun () ->
                if not disposed then 
                    if CudaRegMemSupport.decrRefCount hostArray then            
                        // unregister memory
                        try cudaMem.Unregister() 
                        with :? CudaException -> ()
                        // release cuda memory handle 
                        try cudaMem.Dispose()
                        with :? CudaException -> ()
                        // unpin managed memory
                        (pinHnd :> IDisposable).Dispose()
                disposed <- true)

    override this.Finalize () =
        (this :> IDisposable).Dispose()

    /// the data array
    member this.HostArray = 
        checkDisposed ()
        hostArray
    member internal this.HostArrayPriv = hostArray

    /// GC memory pin handle
    member this.PinHnd = 
        checkDisposed ()
        pinHnd
    member internal this.PinHndPriv = pinHnd

    /// pointer to data 
    member this.Ptr =
        this.PinHnd.Ptr

    /// the CudaRegisteredHostMemory
    member this.CudaRegisteredMemory = 
        checkDisposed ()
        cudaMem
    member internal this.CudaRegisteredMemoryPriv = cudaMem


/// Methods for locking a TensorHostStorage into memory and registering the memory with CUDA
/// for fast data transfers with GPU device.
module CudaRegMem =
    open CudaRegMemSupport

    /// get CudaRegMemHnd for already locked TensorHostStorage          
    let get data =      
        lock syncLock (fun () ->
            if not (dataRegistrations.ContainsKey data) then
                failwith "the specified TensorHostStorage is not registered with CUDA for fast data transfer" 
            registeredCount.[data] <- registeredCount.[data] + 1
            let dr = dataRegistrations.[data] :?> CudaRegMemHnd
            new CudaRegMemHnd(dr.HostArrayPriv, dr.PinHndPriv, dr.CudaRegisteredMemoryPriv)   
        )
        
    /// gets the CudaRegisteredMemory for already locked TensorHostStorage without 
    /// incrementing the reference count
    let getCudaRegisteredMemory data =
        lock syncLock (fun () ->
            if not (dataRegistrations.ContainsKey data) then
                failwith "the specified TensorHostStorage is not registered with CUDA for fast data transfer" 
            let dr = dataRegistrations.[data] :?> CudaRegMemHnd
            dr.CudaRegisteredMemory
        )            

    /// registers a TensorHostStorage (multiple registrations are okay) and returns the corresponding CudaRegMemHnd
    let register (data: ITensorHostStorage) = 
        lock syncLock (fun () ->
            if dataRegistrations.ContainsKey data then get data      
            else
                // pin managed memory so that address cannot change
                let pinHnd = data.Pin ()
                let dataAddr = pinHnd.Ptr
                let dataByteSize = data.DataSizeInBytes

                // construct cuda memory handle and register it
                let cudaMem = new CudaRegisteredHostMemory<byte> (dataAddr, SizeT dataByteSize)    
                try cudaMem.Register (BasicTypes.CUMemHostRegisterFlags.None)
                with :? CudaException as ex ->
                    if ex.CudaError = CUResult.ErrorInvalidValue then
                        // probably memory is not properly aligned
                        raise (CannotCudaRegisterMemory ex.Message)
                    else reraise ()

                // create handle object
                let dr = new CudaRegMemHnd(data, pinHnd, cudaMem)     
                dataRegistrations.[data] <- dr
                registeredCount.[data] <- 1
                dr
        )
              
