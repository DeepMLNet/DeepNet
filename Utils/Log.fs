namespace DeepNet.Utils


/// Log level.
[<RequireQualifiedAccess>]
type LogLevel =
    | Trace = 10
    | Debug = 20
    | Info = 30
    | Warn = 40
    | Err = 50
    | None = 1000


/// Logs messages from the specified source.
/// Supports settings different log levels for each source.
type Log (src: string) =

    static let mutable defaultLevel = LogLevel.None
    static let mutable srcLevels: Map<string, LogLevel> = Map.empty

    /// Sets the log level for log sources that have not explicitly been specified.
    static member setLevel level =
        defaultLevel <- level

    /// Sets the log level for the specified source.
    static member setLevel (src, level) =
        srcLevels <- srcLevels |> Map.add src level

    /// Logs the specified string.
    member private this.LogStr (msgLevel: LogLevel) (msg: string) =
        let srcLevel =
            match srcLevels |> Map.tryFind src with
            | Some level -> level
            | None -> defaultLevel
        if msgLevel >= srcLevel then
            eprintf "%s" msg
        
    /// Logs the specified printf-style message using the specified log level.
    member this.Log (msgLevel: LogLevel) format =
        Printf.kprintf (this.LogStr msgLevel) format

    member this.Trace format = this.Log LogLevel.Trace format
    member this.Debug format = this.Log LogLevel.Debug format
    member this.Info format = this.Log LogLevel.Info format
    member this.Warn format = this.Log LogLevel.Warn format
    member this.Err format = this.Log LogLevel.Err format

