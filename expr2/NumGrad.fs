module NumGrad

open NDArray


/// evaluates the Jacobian of f at x numerically with specified finite difference step
let numGradEpsilon epsilon f x =
    let y = f x
    let xShp, yShp = shape x, shape y
    let xElems, yElems = nElems xShp, nElems yShp
    let xf, yf = x |> reshape [xElems], y |> reshape [yElems]

    let j = zeros [yElems; xElems]
    for xi = 0 to xElems - 1 do
        let xdf = copy xf
        xdf |> set [xi] ((xf |> get [xi]) + epsilon)
        let ydf = xdf |> reshape xShp |> f |> reshape [yElems]
        let d = (ydf - yf) / epsilon       
        j |> view [All; Elem xi] |> copyTo d
    j

/// evaluates the Jacobian of f at x numerically
let numGrad = numGradEpsilon 1e-5f

let exprGradDiff evalEnv wrt expr =
    let g = ExprForwardDiff.grad wrt expr
    let exprFun = (expr |> OpEval.toFun |> OpEval.addArg wrt) |> OpEval.usingEvalEnv evalEnv
    let gradFun = (g |> OpEval.toFun |> OpEval.addArg wrt) |> OpEval.usingEvalEnv evalEnv

    let value = evalEnv.VarEnv.[Op.extractVar wrt]
    let symGradVal = gradFun value
    let exprGradVal = numGrad exprFun value
    let gradDiff = abs (symGradVal - exprGradVal)
    sum gradDiff |> NDArray.value


let reverseDiffDeviations evalEnv expr =
    let mutable devs = Map.empty
    let rDiffs = ExprReverseDiff.reverseDiff expr
    for wrt, rDiff in rDiffs |> Map.toSeq do
        let exprFun = (expr |> OpEval.toFun |> OpEval.addArg (Op.makeVar wrt)) |> OpEval.usingEvalEnv evalEnv
        let rDiffFun = (rDiff |> OpEval.toFun |> OpEval.addArg (Op.makeVar wrt)) |> OpEval.usingEvalEnv evalEnv

        let value = evalEnv.VarEnv.[wrt]
        let symGradVal = rDiffFun value
        let exprGradVal = numGrad exprFun value
        let gradDiff = abs (symGradVal - exprGradVal)
        devs <- devs |> Map.add (Op.VarSpec.name wrt) (sum gradDiff |> NDArray.value)
    devs

let reverseDiffDeviationsOkay evalEnv expr =
    let maxDeviation = 1e-4f
    reverseDiffDeviations evalEnv expr |> Map.iter
        (fun name dev -> if dev > maxDeviation then printfn "deviation wrt %s = %f" name dev)
    reverseDiffDeviations evalEnv expr |> Map.forall (fun _ dev -> dev < maxDeviation) 


