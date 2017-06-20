namespace GaussianProcess

open Tensor
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Random


module HostTensor = 

    /// Converts a 1D Tensor to a Math.Net Vector.
    let toVector (a: Tensor<'T>) : Vector<'T> =
        match typeof<'T> with
        | t when t=typeof<single> ->
            a |> HostTensor.toArray |> unbox |> Single.DenseVector.OfArray |> unbox
        | t when t=typeof<double> ->
            a |> HostTensor.toArray |> unbox |> Double.DenseVector.OfArray |> unbox
        | t -> failwithf "unsupported type: %A" t

    /// Converts a Math.Net Vector to a 1D Tensor.
    let ofVector (a: Vector<'T>) : Tensor<'T> =
        a.ToArray() |> HostTensor.ofArray

    /// Converts a 2D Tensor to a Math.Net Matrix.
    let toMatrix (a: Tensor<'T>) : Matrix<'T> =
        match typeof<'T> with
        | t when t=typeof<single> ->
            a |> HostTensor.toArray2D |> unbox |> Single.DenseMatrix.OfArray |> unbox
        | t when t=typeof<double> ->
            a |> HostTensor.toArray2D |> unbox |> Double.DenseMatrix.OfArray |> unbox
        | t -> failwithf "unsupported type: %A" t

    /// Converts a Math.Net Matrix to a 2D Tensor.
    let ofMatrix (a: Matrix<'T>) : Tensor<'T> =
        a.ToArray() |> HostTensor.ofArray2D
        


/// Multivariate normal distribution
type MVN () =

    /// Gets multiple samples from the multivariate normal distribution.
    static member sampleN (rnd, mu: Tensor<double>, sigma: Tensor<double>, nSamples) =
        if mu.NDims <> 1 then invalidArg "mu" "mu must be a vector"
        match sigma.Shape with
        | [m; n] when m = mu.Shape.[0] && m = n -> ()
        | _ -> invalidArg "sigma" "sigma must be a square matrix with same size as mu"
        
        let muMat = mu.[*, NewAxis] |> Tensor.replicate 1 nSamples |> HostTensor.toMatrix
        let sigmaMat = sigma |> HostTensor.toMatrix
        let k = Double.DenseMatrix.CreateDiagonal(int nSamples, int nSamples, 1.0)
        let normal = MatrixNormal.Sample(rnd, muMat, sigmaMat, k)
        normal |> HostTensor.ofMatrix

    /// Gets one sample from the multivariate normal distribution.
    static member sample (rnd, mu, sigma) =
        let s = MVN.sampleN (rnd, mu, sigma, 1L)
        s.[*, 0L]




