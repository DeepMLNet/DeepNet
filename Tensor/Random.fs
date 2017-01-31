namespace ArrayNDNS

open Basics
open System
open MathNet.Numerics.Distributions

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
        
        /// Generates an infinite sequence of random doubles x ~ N(mean,variance)
        member this.NormalDouble mean variance =
            let normal = Normal.WithMeanVariance (mean, variance)
            Seq.initInfinite (fun _ -> normal.Sample())
        
        /// Generates an infinite sequence of random singles x ~ N(mean,variance)
        member this.NormalSingle mean variance =
            this.NormalDouble mean variance |> Seq.map single

        /// Generates an infinite sequence of random numbers within the given range.
        member this.SeqSingle (minValue: single, maxValue: single) =
            this.SeqDouble (double minValue, double maxValue) |> Seq.map single

        /// Samples each element of an ArrayND of shape shp from a discrete uniform distribution
        /// between minValue and maxValue.      
        member this.IntArrayND (minValue, maxValue) shp =
            this.Seq (minValue, maxValue)
            |> ArrayNDNS.ArrayNDHost.ofSeqWithShape shp

        /// Samples each element of an ArrayND of shape shp from a uniform distribution
        /// between minValue and maxValue.
        member this.UniformArrayND (minValue: 'T, maxValue: 'T) shp =
            let minValue, maxValue = conv<float> minValue, conv<float> maxValue
            this.SeqDouble() 
            |> Seq.map (fun x -> x * (maxValue - minValue) + minValue |> conv<'T>)
            |> ArrayNDNS.ArrayNDHost.ofSeqWithShape shp
            
        /// Samples each element of an ArrayND of shape shp from a uniform distribution
        /// between minValue and maxValue.
        member this.SortedUniformArrayND (minValue: 'T, maxValue: 'T) shp =
            let nElems = shp |> List.fold (*) 1L
            let minValue, maxValue = conv<float> minValue, conv<float> maxValue
            this.SeqDouble() 
            |> Seq.map (fun x -> x * (maxValue - minValue) + minValue |> conv<'T>)
            |> Seq.take (int32 nElems)
            |> Seq.toList
            |> List.sort
            |> ArrayNDNS.ArrayNDHost.ofList
            |> ArrayNDNS.ArrayND.reshape shp
        
        /// Generates an array of random elements x ~ N(mean,variance)
        member this.NormalArrayND (mean: 'T, variance: 'T) shp  =
            let mean, variance = conv<float> mean, conv<float> variance
            this.NormalDouble mean variance |> Seq.map conv<'T>
            |> ArrayNDNS.ArrayNDHost.ofSeqWithShape shp

