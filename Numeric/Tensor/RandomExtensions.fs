namespace Tensor

open Tensor.Utils
open System


/// Extensions to System.Random.
[<AutoOpen>]
module RandomExtensions = 

    type System.Random with

        /// Generates an infinite sequence of non-negative random integers.
        member this.Seq () =
            Seq.initInfinite (fun _ -> this.Next(maxValue))    

        /// Generates an infinite sequence of non-negative random integers that is less than the specified maximum.
        member this.Seq (maxValue) =
            Seq.initInfinite (fun _ -> this.Next(maxValue))    

        /// Generates an infinite sequence of random integers within the given range.
        member this.Seq (minValue, maxValue) =
            Seq.initInfinite (fun _ -> this.Next(minValue, maxValue))

        /// Generates a random floating-point number within the given range.
        member this.NextDouble (minValue, maxValue) =
            this.NextDouble() * (maxValue - minValue) + minValue

        /// Generates an infinite sequence of random numbers between 0.0 and 1.0.
        member this.SeqDouble () =
            Seq.initInfinite (fun _ -> this.NextDouble())

        /// Generates an infinite sequence of random numbers within the given range.
        member this.SeqDouble (minValue, maxValue) =
            Seq.initInfinite (fun _ -> this.NextDouble(minValue, maxValue))
        
        /// Generates a sample from a normal distribution with the given mean and variance.
        member this.NextNormal (mean, variance) =
            let rec notZeroRnd () =
                match this.NextDouble() with
                | x when x > 0.0 -> x
                | _ -> notZeroRnd()
            let u1, u2 = notZeroRnd(), this.NextDouble()
            let z0 = sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)
            mean + z0 * sqrt variance

        /// Generates an infinite sequence of samples from a normal distribution with the given mean and variance.
        member this.SeqNormal (mean, variance) =
            Seq.initInfinite (fun _ -> this.NextNormal(mean, variance))
        
