module MathInterface

open Wolfram.NETLink

let mfilePath = @"C:\Local\surban\dev\Deep\DeepNet\TestExes\GPTransfer\GPTransfer.m"

let mathArgs = [| @"-linkname"
                  @"C:\Program Files\Wolfram Research\Mathematica\11.0\math.exe -mathlink"  |]
                  //@"C:\Program Files\Wolfram Research\Mathematica\11.0\MathKernel.exe -mathlink"  |]

// connect to Mathematica
let link = MathLinkFactory.CreateKernelLink mathArgs
link.WaitAndDiscardAnswer ()

// load .m file with our functions
link.PutFunctionAndArgs("Import", mfilePath)
link.EndPacket ()
link.WaitAndDiscardAnswer ()


let eval (funcName: string) (args: 'T[] list) =
    link.PutFunction (funcName, List.length args)
    for arg in args do
        link.Put(arg :> System.Array, null)
    link.EndPacket ()
    link.WaitForAnswer () |> ignore
    link.GetArray (typeof<'T>, 1) :?> 'T[]

let doMathTest () =
    link.PutFunctionAndArgs("TestFunc", 3)
    link.EndPacket()
    link.WaitForAnswer() |> ignore
    let res = link.GetInteger ()
    printfn "TestFunc[3]=%d" res


let doMathTest2 () =
    let inAry = [| 1.0; 2.0; 3.0 |]
    let resAry = eval "TestFunc2" [inAry]
    printfn "resAry=%A" resAry

