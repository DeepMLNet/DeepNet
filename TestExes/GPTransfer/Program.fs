namespace GPTransfer

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System
open Datasets
open Models

module Program =
  
    ///Load the ablone dataset, classify gender from data
    let abalone = DataParser.loadSingleDataset "abalone.data.txt" [0] ','
    
    ///classified the dataset using a MLP with one hidden layer
    ///(analogous to Lern Mnist Project)
    let classificationMLP()=
    
        let mb = ModelBuilder<single> "NeuralNetModel"

        // define symbolic sizes
        let nBatch  = mb.Size "nBatch"
        let nInput  = mb.Size "nInput"
        let nClass  = mb.Size "nClass"
        let nHidden = mb.Size "nHidden"
       
       // define model parameters
        let mlp = 
            MLP.pars (mb.Module "MLP") 
                { Layers = [{NInput=nInput; NOutput=nHidden; TransferFunc=NeuralLayer.Tanh}
                            {NInput=nHidden; NOutput=nClass; TransferFunc=NeuralLayer.SoftMax}]
                  LossMeasure = LossLayer.CrossEntropy }
        // define variables
        let input  : ExprT<single> = mb.Var "Input"  [nBatch; nInput]
        let target : ExprT<single> = mb.Var "Target" [nBatch; nClass]

        // instantiate model
        mb.SetSize nInput abalone.Trn.[0].Img.Shape.[0]
        mb.SetSize nClass mnist.Trn.[0].Lbl.Shape.[0]
        mb.SetSize nHidden 100
        let mi = mb.Instantiate DevCuda
        ()

    [<EntryPoint>]
    let main argv = 

        TestFunctions.testDatasetParser()
//        TestFunctions.testMultiGPLayer DevHost
//        TestFunctions.testMultiGPLayer DevCuda
   
//        TestUtils.evalHostCuda TestFunctions.testMultiGPLayer
//        TestUtils.compareTraces TestFunctions.testMultiGPLayer false |> ignore




        0
//

