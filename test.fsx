#r "bin\debug\expr2.exe"

let Aval = NDArray.zeros [2; 3]
NDArray.set Aval [1; 1] 3.0
let bval = NDArray.zeros [2; 1]
NDArray.set Aval [1] 4.0

let xval = NDArray.zeros [3; 1]
NDArray.set xval [1; 0] 2.0
let tval = NDArray.zeros [2; 1]
NDArray.set tval [1; 0] 1.0

let env = ["A", Aval; "b", bval; "x", xval; "t", tval;]
          |> Map.ofList 

let out = Op.Add(Op.Dot (Op.Var "A", Op.Var "x"), Op.Var "b")
let loss = Op.Sum(Op.Power(Op.Substract(out, Op.Var "t"), Op.Const 2.0))

let gradA = Op.grad loss "A"
let gradb = Op.grad loss "b"

//let opt = Op.Multiply (Op.Var "v1", Op.Var "v2")
//Op.eval env opt
//
//let g = Op.grad opt "v1"
//Op.eval env g

Op.eval env loss

Op.eval env gradA
Op.eval env gradb
