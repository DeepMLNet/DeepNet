namespace FracSigmoid

open Basics
open ArrayNDNS


module FracSigmoid =

    let rInner = 1e-4
    let maxIters = 50

    let inline sigmoid x =
        1.0 / (1.0 + exp -x)

    let inline dSigmoid x =
        (1.0 - sigmoid x) * sigmoid x

    let inline sigmoidInv y =
        log (y / (1.0 - y))

    let sigmoidFixPoint = 
        (0.6, {0..600})
        ||> Seq.fold (fun x _ -> sigmoid x)

    let dSigmoidFixPoint =
        dSigmoid sigmoidFixPoint

    let inline xi x =
        let rec calc iter fac z = 
            if iter > maxIters then
                System.Double.MaxValue
            elif abs (z - sigmoidFixPoint) <= rInner then
                fac * (z - sigmoidFixPoint)
            else
                calc (iter + 1) (fac / dSigmoidFixPoint) (sigmoid z)
        calc 0 1.0 x

    let inline xiInv y =
        let rec calc iter z =
            if iter > maxIters then
                System.Double.MaxValue
            elif abs z <= rInner then
                let mutable z = z + sigmoidFixPoint
                for j=0 to iter - 1 do
                    z <- sigmoidInv z
                z
            else
                calc (iter + 1) (z * dSigmoidFixPoint)
        calc 0 y

    let inline fracSigmoid n x =
        xiInv (dSigmoidFixPoint ** n * xi x)
        

[<AutoOpen>]
module FracSigmoidTableTypes =

    type Info = {
        NMin:       float
        NMax:       float
        NPoints:    int
        XMin:       float
        XMax:       float
        XPoints:    int
    }

    type FracSigmoidTable = {
        /// [nIdx, xIdx]
        Points:     ArrayNDHostT<single>
        Info:       Info
    }


module FracSigmoidTable =
     
    let generate (info: Info) =
        let xMin, xMax = info.XMin, info.XMax
        let xPoints = info.XPoints

        let nMin, nMax = info.NMin, info.NMax
        let nPoints = info.NPoints

        let xs = ArrayNDHost.linSpaced xMin xMax xPoints |> ArrayNDHost.toArray
        let ns = ArrayNDHost.linSpaced nMin nMax nPoints |> ArrayNDHost.toArray

        // idx = nIdx * xPoints + xIdx
        let tblFlat = 
            Array.Parallel.init (nPoints * xPoints) (fun idx ->
                let xIdx = idx % xPoints
                let nIdx = idx / xPoints
                let x, n = xs.[xIdx], ns.[nIdx] 
                FracSigmoid.fracSigmoid n x
            )
        let tbl = tblFlat |> ArrayNDHost.ofArray |> ArrayND.reshape [nPoints; xPoints]

        {
            Points = tbl |> ArrayND.convert :> ArrayNDT<single> :?> ArrayNDHostT<single>
            Info   = info
        }

    let save hdf name (tbl: FracSigmoidTable) =
        tbl.Points |> ArrayNDHDF.write hdf name
        hdf.SetRecord (name, tbl.Info)

    let load hdf name =
        {
            Points = ArrayNDHDF.read hdf name
            Info   = hdf.GetRecord name
        }

