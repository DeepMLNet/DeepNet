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
        

/// Normal distribution.        
type Normal () =

    /// Gets multiple samples from the normal distribution.
    static member sampleN (rnd: System.Random, mean: double, var: double, nSamples: int64) =
        Normal.Samples (rnd, mean, sqrt var)
        |> Seq.take (int nSamples)
        |> HostTensor.ofSeq

    /// Gets one sample from the normal distribution.
    static member sample (rnd, mean, var) =
        Normal.Sample (rnd, mean, sqrt var)


/// Multivariate normal distribution
type MVN () =

    /// Gets multiple samples from the multivariate normal distribution.
    static member sampleN (rnd, mu: Tensor<double>, sigma: Tensor<double>, nSamples) =
        if mu.NDims <> 1 then invalidArg "mu" "mu must be a vector"
        match sigma.Shape with
        | [m; n] when m = mu.Shape.[0] && m = n -> ()
        | _ -> invalidArg "sigma" "sigma must be a square matrix with same size as mu"
        
        let muMat = mu.[*, NewAxis] |> HostTensor.toMatrix
        let sigmaMat = sigma |> HostTensor.toMatrix
        let k = Double.DenseMatrix.CreateDiagonal(1, 1, 1.0)

        let normal = Tensor ([mu.Shape.[0]; nSamples], HostTensor.Dev)
        for smpl in 0L .. nSamples-1L do
            normal.[*, smpl..smpl] <- 
                MatrixNormal.Sample(rnd, muMat, sigmaMat, k) |> HostTensor.ofMatrix
        normal

    /// Gets one sample from the multivariate normal distribution.
    static member sample (rnd, mu, sigma) =
        let s = MVN.sampleN (rnd, mu, sigma, 1L)
        s.[*, 0L]


/// Uniform distribution.
type Uniform () =

    /// Gets multiple samples from the uniform distribution.
    static member sampleN (rnd: System.Random, low, high, nSamples) =
        HostTensor.init [nSamples] (fun _ ->
            low + (high - low) * rnd.NextDouble())

    /// Gets one sample from the uniform distribution.
    static member sample (rnd, low, high) =
        Uniform.sampleN (rnd, low, high, 1L)





