namespace GPTransfer

open ArrayNDNS
open SymTensor
open System

module InvTest =
    let cmplr = DevHost.Compiler, CompileEnv.empty


    ///Generates a batch of diagonal matrices from a batch of vectors 
    let diag v =        
        let vec: ExprT<single> = Expr.var "vec" [SizeSpec.symbol "m";SizeSpec.symbol "n"]
        let di = Expr.diagMat vec
        let funDiag = Func.make cmplr di |> arg1 vec
        funDiag v
    
    let inversionTest a =
        let A: ExprT<'T> = Expr.var "A" [SizeSpec.symbol "n"; SizeSpec.symbol "n"]
        let testInv = (Expr.invert A) .* A
        let testInversion = Func.make cmplr testInv |> arg1 A
        testInversion a

    let difference a b =
        let A: ExprT<'T> = Expr.var "A" [SizeSpec.symbol "n"; SizeSpec.symbol "n"]
        let B: ExprT<'T> = Expr.var "B" [SizeSpec.symbol "m";SizeSpec.symbol "n"; SizeSpec.symbol "n"]
        let res = A - B
        let calcDiff = Func.make cmplr res |> arg2 A B
        calcDiff a b
    
    let absError a b =
        let n = SizeSpec.symbol "n"
        let M: ExprT<'T> = Expr.var "M" [n; n]
        let i = ElemExpr.idx 0
        let j = ElemExpr.idx 1
        let e = ElemExpr.argElem 0
        let absValue = abs(e [i;j])
        let res =  Expr.elements [n;n] absValue [M]
        let calcAbsValue = Func.make cmplr res |> arg1 M
        difference a b |> calcAbsValue
    
    let randomTest (rand:Random) dim range = 
        let inMat = rand.UniformArrayND range [dim;dim]
        let id = ArrayNDHost.identity dim
        let testMat = inversionTest inMat
        let diff = difference testMat id
        let absTestError = absError testMat id
        printfn "inputMatrix =\n%A" inMat
        printfn "testInverse =\n%A" testMat
        printfn "difference =\n%A" diff
        printfn "absoluteValue =\n%A" absTestError

    let inversionTest2 a =
        let A: ExprT<'T> = Expr.var "A" [SizeSpec.symbol "m";SizeSpec.symbol "n"; SizeSpec.symbol "n"]
        let testInv = (Expr.invert A) .* A
        let testInversion = Func.make cmplr testInv |> arg1 A
        testInversion a

    let difference2 a b =
        let A: ExprT<'T> = Expr.var "A" [SizeSpec.symbol "m";SizeSpec.symbol "n"; SizeSpec.symbol "n"]
        let B: ExprT<'T> = Expr.var "B" [SizeSpec.symbol "m";SizeSpec.symbol "n"; SizeSpec.symbol "n"]
        let res = A - B
        let calcDiff = Func.make cmplr res |> arg2 A B
        calcDiff a b
    
    let absError2 a b =
        let n = SizeSpec.symbol "n"
        let m = SizeSpec.symbol "m"
        let M: ExprT<'T> = Expr.var "M" [m; n; n]
        let i = ElemExpr.idx 0
        let j = ElemExpr.idx 1
        let k = ElemExpr.idx 2
        let e = ElemExpr.argElem 0
        let absValue = abs(e [i;j;k])
        let res =  Expr.elements [m;n;n] absValue [M]
        let calcAbsValue = Func.make cmplr res |> arg1 M
        difference2 a b |> calcAbsValue
            
    let randomTest2 (rand:Random) num dim range = 
        let inMat = rand.UniformArrayND range [num;dim;dim]
        let idD = ArrayNDHost.ones [num;dim]
        let id = diag idD
        let testMat = inversionTest2 inMat
        let diff = difference2 testMat id
        let absTestError = absError2 testMat id
        printfn "inputMatrix =\n%A" inMat
        printfn "testInverse =\n%A" testMat
        printfn "difference =\n%A" diff
        printfn "absoluteValue =\n%A" absTestError

