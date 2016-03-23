module NumpyFilesTest

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS


[<Fact>]
let ``Loading of .npz files`` () =
    let path = Util.assemblyDirectory + "/NPZ1.npz"
    printfn "Trying to load npz file at %s" path

    use npz = new NPZFile (path)
    let names = npz.Names |> List.ofSeq
    printfn "Entry names: %A" names

    let curvePos : ArrayNDHostT<float> = npz.Get "curve_pos"
    printfn "curve_pos:\n%A" curvePos

    let dt : ArrayNDHostT<float> = npz.Get "dt"
    printfn "dt: %A" dt

    let distortionActives : ArrayNDHostT<bool> = npz.Get "distortion_actives"
    printfn "distortion_actives: %A" distortionActives

    let hdfPath = Util.assemblyDirectory + "/NPZ1.h5"
    //#if WRITE_HDF5_REF
    using (new HDF5 (hdfPath, HDF5Overwrite)) (fun hdf ->
        ArrayNDHDF.write hdf "curve_pos" curvePos
        ArrayNDHDF.write hdf "dt" dt
        ArrayNDHDF.write hdf "distortion_actives" distortionActives
    )
    printfn "Wrote HDF5 reference to %s" hdfPath
    //#endif

    printfn "Loading HDF5 reference from %s" hdfPath
    use hdf = new HDF5 (hdfPath)
    let curvePosRef : ArrayNDHostT<float> = ArrayNDHDF.read hdf "curve_pos" 
    let dtRef : ArrayNDHostT<float> = ArrayNDHDF.read hdf "dt"
    let distortionActivesRef : ArrayNDHostT<bool> = ArrayNDHDF.read hdf "distortion_actives"

    let curvePosEqual = curvePos ==== curvePosRef
    let dtRefEqual = dt ==== dtRef
    let distortionActivesEqual = distortionActives ==== distortionActivesRef

    printfn "curvePosEqual:\n%A" curvePosEqual
    printfn "dtRefEqual:\n%A" dtRefEqual
    printfn "distortionActivesEqual:\n%A" distortionActivesEqual

    ArrayND.all curvePosEqual |> ArrayND.value |> should equal true
    ArrayND.all dtRefEqual |> ArrayND.value |> should equal true
    ArrayND.all distortionActivesEqual |> ArrayND.value |> should equal true






