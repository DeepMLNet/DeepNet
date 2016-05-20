namespace Basics

open System


[<AutoOpen>]
module RandomExtensions = 

    type System.Random with
        /// Generates an infinite sequence of random numbers within the given range.
        member this.Seq (minValue, maxValue) =
            Seq.initInfinite (fun _ -> this.Next(minValue, maxValue))

        /// Generates an infinite sequence of random numbers between 0.0 and 1.0.
        member this.SeqDouble () =
            Seq.initInfinite (fun _ -> this.NextDouble())

        /// Generates an infinite sequence of random numbers within the given range.
        member this.SeqDouble (minValue, maxValue) =
            Seq.initInfinite (fun _ -> this.NextDouble() * (maxValue - minValue) + minValue)

        /// Generates an infinite sequence of random numbers between 0.0 and 1.0.
        member this.SeqSingle () =
            this.SeqDouble () |> Seq.map single

        /// Generates an infinite sequence of random numbers within the given range.
        member this.SeqSingle (minValue: single, maxValue: single) =
            this.SeqDouble (double minValue, double maxValue) |> Seq.map single
        


