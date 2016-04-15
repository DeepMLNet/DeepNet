/// PID Controller
module PID

open System.Diagnostics
open System.Collections.Generic


/// PID Controller configuration
type Cfg = {
    /// factor for proportional component
    PFactor:        float
    /// factor for integral component
    IFactor:        float
    /// factor for differential component
    DFactor:        float
    /// integration time for integral component
    ITime:          float
    /// smoothing time for differential component
    DTime:          float
}

type private State = {
    Time:           float
    Target:         float
    Value:          float
} with 
    member this.Diff = this.Target - this.Value

/// PID Controller
type Controller (cfg: Cfg) =        
    let history = Queue<State> ()
    let timer = Stopwatch.StartNew()

    let dequeueUntil t =
        while let {State.Time=it} = history.Peek () in it < t do
            history.Dequeue() |> ignore

    /// Computes the PID control for given target and current value.
    member this.Control target value =
        let t = float timer.ElapsedMilliseconds / 1000.
        let state = {Time=t; Target=target; Value=value}           
        history.Enqueue state
        dequeueUntil (t - max cfg.ITime cfg.DTime)

        let P = target - value

        let I =
            history
            |> Seq.filter (fun {State.Time=ht} -> t - ht <= cfg.ITime)
            |> Seq.pairwise
            |> Seq.fold (fun i (s1, s2) -> i + s1.Diff * abs (s1.Time - s2.Time)) 0.

        let dState = 
            history |> Seq.find (fun {State.Time=ht} -> t - ht <= cfg.DTime)
        let D =
            if dState.Time < t then (state.Target - dState.Target) / (state.Time - dState.Time)
            else 0.                         

        //printfn "P=%.3f    I=%.3f    D=%.3f" P I D
        cfg.PFactor * P + cfg.IFactor * I + cfg.DFactor * D
