module NumpyFilesTest

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS


[<Fact>]
let ``Loading of .npz files`` () =
    let path = Util.assemblyDirectory + "/TestData/NPZ1.npz"
    printfn "Trying to load npz file at %s" path

    use npz = new NPZFile (path)
    let names = npz.Names |> List.ofSeq
    printfn "Entry names: %A" names

    let curvePos : Tensor<float> = npz.Get "curve_pos"
    printfn "curve_pos:\n%A" curvePos

    let dt : Tensor<float> = npz.Get "dt"
    printfn "dt: %A" dt

    let distortionActives : Tensor<bool> = npz.Get "distortion_actives"
    printfn "distortion_actives: %A" distortionActives

    let hdfPath = Util.assemblyDirectory + "/TestData/NPZ1.h5"
    //#if WRITE_HDF5_REF
    using (new HDF5 (hdfPath, HDF5Overwrite)) (fun hdf ->
        HostTensor.write hdf "curve_pos" curvePos
        HostTensor.write hdf "dt" dt
        HostTensor.write hdf "distortion_actives" distortionActives
    )
    printfn "Wrote HDF5 reference to %s" hdfPath
    //#endif

    printfn "Loading HDF5 reference from %s" hdfPath
    use hdf = new HDF5 (hdfPath)
    let curvePosRef : Tensor<float> = HostTensor.read hdf "curve_pos" 
    let dtRef : Tensor<float> = HostTensor.read hdf "dt"
    let distortionActivesRef : Tensor<bool> = HostTensor.read hdf "distortion_actives"

    let curvePosEqual = curvePos ==== curvePosRef
    let dtRefEqual = dt ==== dtRef
    let distortionActivesEqual = distortionActives ==== distortionActivesRef

    printfn "curvePosEqual:\n%A" curvePosEqual
    printfn "dtRefEqual:\n%A" dtRefEqual
    printfn "distortionActivesEqual:\n%A" distortionActivesEqual

    Tensor.all curvePosEqual |> should equal true
    Tensor.all dtRefEqual |> should equal true
    Tensor.all distortionActivesEqual |> should equal true






