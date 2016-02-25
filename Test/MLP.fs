module MLP

open SymTensor


module NeuralLayer = 

    type Pars<'T> = {
        Weights:    ExprT<'T>;
        Bias:       ExprT<'T>;
    }

    let parsWithOut (nOut: int) (mc: MC) =
        {Weights = mc.Var "weights"     [nOut; "In"];
         Bias    = mc.Var "bias"        [nOut];}

    let pars (mc: MC) =
        {Weights = mc.Var "weights"     ["Out"; "In"];
         Bias    = mc.Var "bias"        ["Out"];}

    let output pars input =
        tanh (pars.Weights .* input + pars.Bias)



module AutoEncoder =

    type Pars<'T> = {
        InLayer:    NeuralLayer.Pars<'T>;
        OutLayer:   NeuralLayer.Pars<'T>;
    }

    let pars (mc: MC) nLatent tiedWeights =
        let p =
            {InLayer   = mc.Module "InLayer"   |> NeuralLayer.parsWithOut nLatent;
             OutLayer  = mc.Module "OutLayer"  |> NeuralLayer.pars;}
        if tiedWeights then
            {p with OutLayer = {p.OutLayer with Weights = p.InLayer.Weights.T}}
        else p

    let latent pars input =
        NeuralLayer.output pars.InLayer input

    let recons pars input =
        let hidden = latent pars input
        NeuralLayer.output pars.OutLayer hidden


//    {Weights = Expr.var "weights" [SizeSpec.symbol "NeuronsOut"; SizeSpec.symbol "NeuronsIn"];
//     Bias    = Expr.var "bias" [SizeSpec.symbol "NeuronsOut"]; }

// now this is a bit problematic:
// have to explicitly specify weights and bias
// biggest problem is specification of parameters
// caller would have to define a variable 

// part of number of weights could be derived automatically
// optionally we just set 


// problem: 
//  - variable usage/declaration is too verbose
//  - so now shape algebra has to figure out the neuron count
//  - but this is not completely possible, so user has to specifiy uninferable dimensions
//  - what about layer name?

//
//let buildNetwork input =
//    let layer1 = neuralLayer (neuralLayerPars "1") input
//    let layer2 = neuralLayer (neuralLayerPars "2") layer1
//    let layer3 = neuralLayer (neuralLayerPars "3") layer2
//    layer3
//
//
//let buildAutoencoder input =
//    let pars = neuralLayerPars "in"
//    let parsIn = {pars with Bias = Expr.zerosLike pars.Bias}
//    let hiddens = neuralLayer parsIn input
//    let parsRecon = {parsIn with Weights = pars.Weights.T}
//    let recon = neuralLayer parsRecon hiddens
//    recon, hiddens
//
//
//
//
