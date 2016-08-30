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

        member this.Deriv dOp args = failwith "not impl"

        member this.EvalSimple args =
            let trnXshp = args.[0].Shape
            let nGps, nTrnSmpls = trnXshp.[0], trnXshp.[1]
            
            let cm = ArrayNDHost.zeros [nGps; nTrnSmpls; nTrnSmpls]
                        
                
            cm

            

let squaredExponentialCovarianceMatrix (trnX: ExprT<'T>) (lengthscale: ExprT<'T>) =
    Expr.Nary (Expr.ExtensionOp SquaredExponentialCovarianceMatrixOp, [trnX; lengthscale])
    |> Expr.check

