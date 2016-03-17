module ModelTest

open System.IO


open Basics
open ArrayNDNS

open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Datasets


let mnist = Mnist.load @"..\..\..\Data\MNIST"


let ``Test neural net`` device =
    let mc = ModelBuilder<single> "NeuralNetModel"
    
    // symbolic sizes
    let batchSize  = mc.Size "BatchSize"
    let nInput     = mc.Size "nInput"
    let nTarget    = mc.Size "nTarget"

    // model parameters
    let pars = NeuralLayer.pars (mc.Module "Layer1") nInput nTarget
    
    // input / output variables
    let input =  mc.Var "Input"  [nInput;  batchSize]
    let target = mc.Var "Target" [nTarget; batchSize]

    let mc = mc.ParametersComplete ()

    // expressions
    let loss = NeuralLayer.loss pars input target |> mc.Subst
    //let dLoss = mc.WrtParameters loss

    //let loss = Optimizer.optimize loss
    //let dLoss = Optimizer.optimize dLoss

    //printfn "loss:\n%A" loss
    //printfn "dLoss:\n%A" dLoss

    let post x =
        if device = DevCuda then ArrayNDCuda.toDev x :> ArrayNDT<'T>
        else x :> ArrayNDT<'T>

    // MNIST dataset
    let tstImgs =  
        mnist.TstImgs
        |> ArrayND.reorderAxes [2; 0; 1] 
        |> ArrayND.reshape [-1; (ArrayND.shape mnist.TstImgs).[0]]
        |> fun x -> x.[*, 0..10]
        |> post
    let tstLbls =  
        mnist.TstLbls
        |> ArrayND.reorderAxes [1; 0] 
        |> fun x -> x.[*, 0..10]
        |> post

    // infer sizes and variable locations from dataset
    mc.UseTmplVal input tstImgs     
    mc.UseTmplVal target tstLbls

    printfn "inferred sizes: %A" mc.SymSizeEnv
    printfn "inferred locations: %A" mc.VarLocs

    // instantiate model
    let mi = mc.Instantiate device
    //let mi = mc.Instantiate DevHost

    // compile functions
    let lossFun = mi.Func (loss) |> arg2 input target
    //let dLossFun = mi.Func (dLoss) |> arg2 input target

    // calcualte test loss on MNIST
    let tstLoss = lossFun tstImgs tstLbls
    printfn "Test loss on MNIST=%A" tstLoss

    let opt = Optimizers.gradientDescent {Step=1e-1f} loss mc.ParameterSet.Flat   
    let optFun = mi.Func opt |> arg2 input target
    
    printfn "Optimizing..."
    for itr = 0 to 2 do
        optFun tstImgs tstLbls |> ignore
        let l = lossFun tstImgs tstLbls
        printfn "Loss afer %d iterations: %A" itr l

    ()


let compareHostCuda func =
    printfn "Evaluating on host..."
    Trace.startSession "Host"
    func DevHost
    let hostTrace = Trace.endSession ()
    use tw = File.CreateText("Host.txt")
    Trace.dump tw hostTrace

    printfn "Evaluating on CUDA device..."
    Trace.startSession "CUDA"
    func DevCuda
    let cudaTrace = Trace.endSession ()
    use tw = File.CreateText("CUDA.txt")
    Trace.dump tw cudaTrace
    printfn "Done."

    Trace.compare hostTrace cudaTrace

    //printfn "Host trace:\n%A" hostTrace
    //printfn "CUDA trace:\n%A" cudaTrace



let ``Test Autoencoder`` () =
    let mc = ModelBuilder<single> "Autoencoder"
    
    // symbolic sizes
    let batchSize  = mc.Size "BatchSize"
    let nInput     = mc.Size "nInput"
    
    let pars = Autoencoder.pars (mc.Module "Autoencoder1") {NVisible=nInput; NLatent=mc.Fix 50; Tied=false}
    
    let input  =     mc.Var "Input"  [nInput; batchSize]

    let loss = Autoencoder.loss pars input 

    printfn "Autoencoder:\n%A" loss
    let dloss = Deriv.compute loss
    ()
    //printfn "%A" dloss



[<EntryPoint>]
let main argv = 
    Basics.Cuda.CudaSup.init ()
    ManagedCuda.CudaContext.ProfilerStart()

    //``Test neural net`` ()
    compareHostCuda ``Test neural net``


    //``Test Autoencoder`` ()

    // need code to load data and perform regression
        

    printfn "CUDA shutdown."
    Basics.Cuda.CudaSup.shutdown ()
    0
