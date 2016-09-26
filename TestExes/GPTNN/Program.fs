// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace GPTransfer
open FSharp.Configuration
open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor

module ConfigTraining = 
    type TestConfig = YamlConfig<"Config.yaml">

    type NetworkT = GPTransferUnit| MLGPT | MLP

    type HyperparsT =   GPTPars of list<GPTransferUnit.HyperPars> 
                        | MLGPTPars of list<MLGPT.HyperPars> 
                        | MLPPars of list<MLP.HyperPars> 

    type Network = {
        Type:   NetworkT
        Name:   string
        Layers: int
        Hypers: HyperparsT
        }

    [<EntryPoint>]
    let main argv = 
        let config = TestConfig()
        let netName = config.Network.Name
        printfn "NetworkName = %s" netName
        0 // return an integer exit code
