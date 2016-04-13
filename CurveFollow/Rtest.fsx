#I "../packages/RProvider.1.1.15/lib/net40"
#I "../packages/R.NET.Community.1.6.4/lib/net40"
#I "../packages/R.NET.Community.FSharp.1.6.4/lib/net40"

#r "RDotNet.dll"
#r "RDotNet.FSharp.dll"
#r "RProvider.dll"
#r "RProvider.Runtime.dll"

open RProvider
open RProvider.graphics
open RProvider.grDevices

let x = seq { 0 .. 4 } |> Array.ofSeq
let y = seq { 0 .. 2 } |> Array.ofSeq
let matrix : float[,] = Array2D.zeroCreate 5 3
matrix.[1,2] <- 1.0
matrix.[3,2] <- 1.0

matrix.[1,0] <- 1.0
matrix.[2,0] <- 1.0
matrix.[3,0] <- 1.0


R.image (namedParams [
                    "x", box x
                    "y", box y
                    "z", box matrix
])

