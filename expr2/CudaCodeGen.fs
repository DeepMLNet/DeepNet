﻿module CudaCodeGen

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
    let cuda = new CudaContext(createNew=false)
    let cuMod = cuda.LoadModulePTX("NDArray.cu.obj")
    let myKernel = CudaKernel("sayHi", cuMod, cuda)
    myKernel.Run()
    //CudaNDArray.scalarConst 1 ()


