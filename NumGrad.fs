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
let numGrad = numGradEpsilon 1e-5

let exprGradDiff expr env wrt =
    let g = OpGrad.grad wrt expr
    let exprFun = (expr |> OpEval.toFun |> OpEval.addArg wrt) env
    let gradFun = (g |> OpEval.toFun |> OpEval.addArg wrt) env

    fun value ->
        let symGradVal = gradFun value
        let exprGradVal = numGrad exprFun value
        let gradDiff = abs (symGradVal - exprGradVal)
        sum gradDiff |> NDArray.value




