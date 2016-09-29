namespace Models

open ArrayNDNS
open SymTensor


module GaussianProcess =
    
    /// Kernek
    type Kernel =
        /// linear kernel
        | Linear
        /// squared exponential kernel
        | SquaredExponential of l:single*sign:single
    
    /// GP hyperparameters
    type HyperPars = {
        Kernel :    Kernel}
    
    /// GP parameters
    type Pars = {
        HyperPars:  HyperPars}

    let pars (mb: ModelBuilder<_>) hp = {
        HyperPars = hp
    }

    /// calculates Matrix between two vectors using linear kernel
    let linearCovariance (x:ExprT) (y:ExprT) =
        x .* y
    
    /// calculates Matrix between two vectors using linear kernel
    let squaredExpCovariance (l:single,sigf:single) (x:ExprT) (y:ExprT) =
        let x_smpl  = ElemExpr.idx 0
        let y_smpl  = ElemExpr.idx 1
        let xvec       = ElemExpr.argElem<single> 0
        let yvec       = ElemExpr.argElem<single> 1
        let kse = sigf * (exp -((xvec[x_smpl] - yvec[y_smpl])***2.0f)/(2.0f * l **2.0f))
        let sizeX = Expr.nElems x
        let sizeY = Expr.nElems y
        Expr.elements [sizeX;sizeY] kse [x; y]

    let regression pars x y sigmaNs x_star =
        let covMat z z' =
            match pars.HyperPars.Kernel with
            | Linear -> linearCovariance z z'
            | SquaredExponential (l,sign) -> squaredExpCovariance (l,sign) z z'
        let K           = (covMat x x) + Expr.diagMat sigmaNs
        let Kinv        = Expr.invert K
        let K_star      = covMat x x_star
        let K_starT     = Expr.transpose K_star
        let K_starstar  = covMat x_star x_star
        
        let mean = K_starT .* Kinv .* y
        let cov = K_starstar - K_starT .* Kinv .* K_star
        mean,cov