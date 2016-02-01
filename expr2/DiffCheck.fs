module DiffCheck

open Op
open ExprReverseDiff
open NumGrad

    

let checkReverseDiff evalEnv expr = 
    let evalEnv = evalEnv |> OpEval.EvalEnv.addSizeSymbolsFromExpr expr
    //printfn "using evalEnv %A" evalEnv
    let rec checkSubExpr expr = 
        match expr with
        | Leaf(_) -> ()
        | Unary(_, a) -> checkSubExpr a
        | Binary(_, a, b) -> 
            checkSubExpr a
            checkSubExpr b

        //printfn "checking %A" expr
        if not (reverseDiffDeviationsOkay evalEnv expr) then
            failwithf "deviation between numeric and symbolic derivative too large in op %A" (extractOp expr)

    checkSubExpr expr





