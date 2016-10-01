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

    let inline dXi x =
        let rec calc iter prod z =
            if iter > maxIters then
                System.Double.MaxValue
            elif abs (z - sigmoidFixPoint) <= rInner then
                prod
            else
                calc (iter + 1) ((1.0 - sigmoid z) * sigmoid z / dSigmoidFixPoint) (sigmoid z)
        calc 0 1.0 x

    let inline dXiInv y =
        1.0 / dXi (xiInv y)

    let inline fracSigmoid n x =
        xiInv (dSigmoidFixPoint ** n * xi x)

    let inline dFracSigmoidDX n x =
        dXiInv (dSigmoidFixPoint ** n * xi x) * dSigmoidFixPoint ** n * dXi x

    let inline dFracSigmoidDN n x =
        dXiInv (dSigmoidFixPoint ** n * xi x) * dSigmoidFixPoint ** n * xi x * log dSigmoidFixPoint

        

[<AutoOpen>]
module FracSigmoidTableTypes =

    type InterpolatedFunction =
        | FracSigmoid
        | Sigmoid

    type Info = {
        NMin:       float
        NMax:       float
        NPoints:    int
        XMin:       float
        XMax:       float
        XPoints:    int
        Function:   InterpolatedFunction
        WithDeriv:  bool
    }

    type FracSigmoidTable = {
        /// [nIdx, xIdx]
        Points:     ArrayNDHostT<single>
        DPoints:    (ArrayNDHostT<single> * ArrayNDHostT<single>) option
        Info:       Info
    }


module FracSigmoidTable =
     
    let generate (info: Info) =
        printfn "Generating interpolation table for\n%A" info

        let xMin, xMax = info.XMin, info.XMax
        let xPoints = info.XPoints

        let nMin, nMax = info.NMin, info.NMax
        let nPoints = info.NPoints

        let xs = ArrayNDHost.linSpaced xMin xMax xPoints |> ArrayNDHost.toArray
        let ns = ArrayNDHost.linSpaced nMin nMax nPoints |> ArrayNDHost.toArray

        // idx = nIdx * xPoints + xIdx
        let tbl = 
            Array.Parallel.init (nPoints * xPoints) (fun idx ->
                let xIdx = idx % xPoints
                let nIdx = idx / xPoints
                let x, n = xs.[xIdx], ns.[nIdx] 

                match info.Function with
                | FracSigmoid -> FracSigmoid.fracSigmoid n x
                | Sigmoid -> FracSigmoid.sigmoid x
            )
            |> ArrayNDHost.ofArray 
            |> ArrayND.reshape [nPoints; xPoints]
            |> ArrayND.convert :> ArrayNDT<single> :?> ArrayNDHostT<single>

        let dtbls = 
            if info.WithDeriv then
                let dTbldN = 
                    Array.Parallel.init (nPoints * xPoints) (fun idx ->
                        let xIdx = idx % xPoints
                        let nIdx = idx / xPoints
                        let x, n = xs.[xIdx], ns.[nIdx] 

                        match info.Function with
                        | FracSigmoid -> FracSigmoid.dFracSigmoidDN n x
                        | Sigmoid -> 0.0
                    )
                    |> ArrayNDHost.ofArray 
                    |> ArrayND.reshape [nPoints; xPoints]
                    |> ArrayND.convert :> ArrayNDT<single> :?> ArrayNDHostT<single>
                let dTbldX = 
                    Array.Parallel.init (nPoints * xPoints) (fun idx ->
                        let xIdx = idx % xPoints
                        let nIdx = idx / xPoints
                        let x, n = xs.[xIdx], ns.[nIdx] 

                        match info.Function with
                        | FracSigmoid -> FracSigmoid.dFracSigmoidDX n x
                        | Sigmoid -> FracSigmoid.dSigmoid x
                    )
                    |> ArrayNDHost.ofArray 
                    |> ArrayND.reshape [nPoints; xPoints]
                    |> ArrayND.convert :> ArrayNDT<single> :?> ArrayNDHostT<single>
                Some (dTbldN, dTbldX)
            else None

        {
            Points = tbl 
            DPoints = dtbls
            Info   = info
        }

//    let save hdf name (tbl: FracSigmoidTable) =
//        tbl.Points |> ArrayNDHDF.write hdf name
//        tbl.DPoints |> ArrayNDHDF.write hdf ("d" + name)
//        hdf.SetRecord (name, tbl.Info)
//
//    let load hdf name =
//        {
//            Points  = ArrayNDHDF.read hdf name
//            DPoints = Some (ArrayNDHDF.read hdf ("d" + name))
//            Info    = hdf.GetRecord name
//        }

