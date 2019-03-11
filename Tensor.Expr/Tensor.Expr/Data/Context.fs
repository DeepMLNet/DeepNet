namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Backend


/// A path for a named resource within an expression (variable, data, random variable).
[<StructuredFormatDisplay("{Pretty}")>]
type ContextPath = ContextPath of string list with
    
    /// The parts of the context path.
    member this.Parts =
        let (ContextPath parts) = this
        parts

    /// The parts of the context.
    static member parts (vp: ContextPath) = vp.Parts

    /// Split off last part of context path.
    static member splitLast (vp: ContextPath) = 
        match vp.Parts with
        | [] -> failwith "Cannot split root context path."
        | parts ->
            ContextPath parts.[0 .. parts.Length-2], List.last parts

    /// Sub-context with specified name.
    static member (/) (vp: ContextPath, name: string) =
        ContextPath (vp.Parts @ [name])
    
    /// Appends one context path to another.
    static member (/) (a: ContextPath, b: ContextPath) =
        ContextPath (a.Parts @ b.Parts)

    /// A string that is unique for each context path.
    member this.Str =
        this.Parts 
        |> List.map (fun part -> "[" + part + "]")
        |> String.concat "/"  
        
    /// A string that is unique for each context path.
    static member str (cp: ContextPath) = cp.Str

    /// Pretty string (not necessarily unique).
    member this.Pretty =
        this.Parts |> String.concat "/"

    /// Root (empty) context path.
    static member root = ContextPath []

    /// A resource in the root context of the specified name.
    static member from (name: string) = ContextPath.root / name

    /// True if `path` begins with `start` and is therefore a sub-path of `start`.
    static member startsWith (start: ContextPath) (path: ContextPath) =
        if path.Parts.Length >= start.Parts.Length then
            path.Parts.[0 .. start.Parts.Length-1] = start.Parts
        else
            false



/// A context for named resource creation.
[<StructuredFormatDisplay("{Pretty}")>]
type Context = {
    /// Context path.
    Path: ContextPath
    /// Device.
    Dev: ITensorDevice
} with 

    /// Path of context.
    static member path (ctx: Context) = ctx.Path
    
    /// Device of context.
    static member dev (ctx: Context) = ctx.Dev

    /// Root context on specified device.
    static member root dev = 
        {Path=ContextPath.root; Dev=dev}

    /// Append name to context path.
    static member (/) (ctx: Context, name: string) =
        {ctx with Path=ctx.Path / name}

    /// Append another path to context path.
    static member (/) (ctx: Context, path: ContextPath) =
        {ctx with Path=ctx.Path / path}

    /// Pretty string.
    member this.Pretty =
        sprintf "%A@%A" this.Path this.Dev


