module GPTransferOps

open ArrayNDNS
open SymTensor

type SquaredExponentialCovarianceMatrixUOp =
    | SquaredExponentialCovarianceMatrixUOp

    interface IUOp


type SquaredExponentialCovarianceMatrixOp<'T> =
    | SquaredExponentialCovarianceMatrixOp

    // sources: 0: trnX        [gp, trn_smpl]
    //          1: lengthscale [gp]
    // result:                 [gp, trn_smpl1, trn_smpl2]
    interface IOp<'T> with
        member this.Shape argShapes = 
            let trnXshp = argShapes.[0]
            let nGps, nTrnSmpls = trnXshp.[0], trnXshp.[1]
            [nGps; nTrnSmpls; nTrnSmpls]

        member this.CheckArgs argShapes = 
            let trnXshp, lengthscaleScape = argShapes.[0], argShapes.[1]
            match trnXshp, lengthscaleScape with
            | [nGps; nTrnSmpls], [nGps2] when nGps=nGps2 -> ()
            | _ -> 
                failwithf "trnX (%A) must be of shape [gp, trn_smpl] and lengthscale (%A) must be of shape [gp]"
                    trnXshp lengthscaleScape
                    
        member this.SubstSymSizes _ = this :> IOp<'T>
        
        member this.CanEvalAllSymSizes = true
        
        member this.ToUExpr expr makeOneUop = 
            makeOneUop SquaredExponentialCovarianceMatrixUOp

        member this.Deriv dOp args = 
            [Expr.Nary (Expr.ExtensionOp SquaredExponentialCovarianceMatrixWRTtrnXOp, args @ [dOp])
             Expr.Nary (Expr.ExtensionOp SquaredExponentialCovarianceMatrixWRTtrnLOp, args @ [dOp])]

        member this.EvalSimple args =
            let trnX, lengthscale = args.[0], args.[1]
            let trnXAry = trnX |> ArrayNDHost.toArray2D
            let lengthscaleAry = lengthscale |> ArrayNDHost.toArray

            MathInterface.link.PutFunction ("KSEMat", 2)
            MathInterface.link.Put (trnXAry, null)
            MathInterface.link.Put (lengthscaleAry, null)
            MathInterface.link.EndPacket ()
            MathInterface.link.WaitForAnswer () |> ignore
            let cmAry = MathInterface.link.GetArray (typeof<'T>, 3) :?> 'T[,,]

            let cm = cmAry |> ArrayNDHost.ofArray3D                      
            printfn "Result shape is %A" cm.Shape                
            cm


and SquaredExponentialCovarianceMatrixWRTtrnXOp<'T> =
    | SquaredExponentialCovarianceMatrixWRTtrnXOp
    // sources: 0: trnX        [gp, trn_smpl]
    //          1: lengthscale [gp]
    //          2: dOp         [out, n_gps * n_trn_smpls * n_trn_smpls]
    // result:                 [out, n_gps * n_trn_smpls]
    interface IOp<'T> with
        member this.Shape argShapes = 
            let trnXshp = argShapes.[0]
            let nGps, nTrnSmpls = trnXshp.[0], trnXshp.[1]
            let nOut = argShapes.[2].[0]
            [nOut; nGps * nTrnSmpls]

        member this.CheckArgs argShapes = ()                   
        member this.SubstSymSizes _ = this :> IOp<'T>        
        member this.CanEvalAllSymSizes = true
        
        member this.ToUExpr expr makeOneUop = 
            makeOneUop SquaredExponentialCovarianceMatrixUOp

        member this.Deriv dOp args = failwith "not impl"

        member this.EvalSimple args =
            let trnX, lengthscale, dOpFlat = args.[0], args.[1], args.[2]
            let nGps, nTrnSmpls = trnX.Shape.[0], trnX.Shape.[1]
            let nOut = dOpFlat.Shape.[0]
            let dOp = dOpFlat |> ArrayND.reshape [nOut; nGps; nTrnSmpls; nTrnSmpls]
            
            let trnXAry = trnX |> ArrayNDHost.toArray2D
            let lengthscaleAry = lengthscale |> ArrayNDHost.toArray
            let dOpAry = dOp |> ArrayNDHost.toArray4D 

            MathInterface.link.PutFunction ("dKSEMatdX", 3)
            MathInterface.link.Put (trnXAry, null)
            MathInterface.link.Put (lengthscaleAry, null)
            MathInterface.link.Put (dOpAry, null)
            MathInterface.link.EndPacket ()
            MathInterface.link.WaitForAnswer () |> ignore
            let dXAry = MathInterface.link.GetArray (typeof<'T>, 3) :?> 'T[,,]

            let dX = dXAry |> ArrayNDHost.ofArray3D
            let dXflat = dX |> ArrayND.reshape [nOut; nGps * nTrnSmpls]
            dXflat

and SquaredExponentialCovarianceMatrixWRTtrnLOp<'T> =
    | SquaredExponentialCovarianceMatrixWRTtrnLOp
    // sources: 0: trnX        [gp, trn_smpl]
    //          1: lengthscale [gp]
    //          2: dOp         [out, n_gps * n_trn_smpls * n_trn_smpls]
    // result:                 [out, n_gps]
    interface IOp<'T> with
        member this.Shape argShapes = 
            let trnXshp = argShapes.[0]
            let nGps, nTrnSmpls = trnXshp.[0], trnXshp.[1]
            let nOut = argShapes.[2].[0]
            [nOut; nGps]

        member this.CheckArgs argShapes = ()                   
        member this.SubstSymSizes _ = this :> IOp<'T>        
        member this.CanEvalAllSymSizes = true
        
        member this.ToUExpr expr makeOneUop = 
            makeOneUop SquaredExponentialCovarianceMatrixUOp

        member this.Deriv dOp args = failwith "not impl"

        member this.EvalSimple args =
            let trnX, lengthscale, dOpFlat = args.[0], args.[1], args.[2]
            let nGps, nTrnSmpls = trnX.Shape.[0], trnX.Shape.[1]
            let nOut = dOpFlat.Shape.[0]
            let dOp = dOpFlat |> ArrayND.reshape [nOut; nGps; nTrnSmpls; nTrnSmpls]
            
            let trnXAry = trnX |> ArrayNDHost.toArray2D
            let lengthscaleAry = lengthscale |> ArrayNDHost.toArray
            let dOpAry = dOp |> ArrayNDHost.toArray4D 

            MathInterface.link.PutFunction ("dKSEMatdL", 3)
            MathInterface.link.Put (trnXAry, null)
            MathInterface.link.Put (lengthscaleAry, null)
            MathInterface.link.Put (dOpAry, null)
            MathInterface.link.EndPacket ()
            MathInterface.link.WaitForAnswer () |> ignore
            let dLAry = MathInterface.link.GetArray (typeof<'T>, 2) :?> 'T[,]

            let dL = dLAry |> ArrayNDHost.ofArray2D
            let dLflat = dL |> ArrayND.reshape [nOut; nGps]
            dLflat
            

let squaredExponentialCovarianceMatrix (trnX: ExprT<'T>) (lengthscale: ExprT<'T>) =
    Expr.Nary (Expr.ExtensionOp SquaredExponentialCovarianceMatrixOp, [trnX; lengthscale])
    |> Expr.check

