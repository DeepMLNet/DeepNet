module CudaCodeGen

open Op
open ExprEvalSequencer
open ManagedCuda


/// CUDA context
let cuda = new CudaContext(createNew=false)


module CudaNDArray =   
    let cuMod = cuda.LoadModule("NDArray.cu.obj")

    let scalarConstKernel =
        CudaKernel("scalarConst", cuMod, cuda)

    let scalarConst value ndArray =
        scalarConstKernel.Run([| value |])




//let generateCudaCalls eseq =
//    
//    match eseq with
//    | ExeSequenceItem(exeOp, expr) ->
//        match exeOp with
//        | LeafExe(target, op) ->
//            match op with 
//            | ScalarConst f ->
                

let testMe () =
    printfn "hallo"
    let cuda = new CudaContext(createNew=false)
    let cuMod = cuda.LoadModulePTX("NDArray.cu.obj")
    let myKernel = CudaKernel("scalarConst", cuMod, cuda)
    myKernel.Run([| 1 |])
    //CudaNDArray.scalarConst 1 ()


