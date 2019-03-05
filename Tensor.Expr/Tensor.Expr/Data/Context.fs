namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Backend


/// A path for a variable.
[<StructuredFormatDisplay("{Pretty}")>]
type VarPath = VarPath of string list with
    
    /// The parts of the context.
    member this.Parts =
        let (VarPath parts) = this
        parts

    /// The parts of the context.
    static member parts (vp: VarPath) = vp.Parts

    /// Root (empty) context.
    static member root = VarPath []

    /// Split off last part of context.
    static member splitLast (vp: VarPath) = 
        match vp.Parts with
        | [] -> failwith "Cannot split root context."
        | parts ->
            VarPath parts.[0 .. parts.Length-2], List.last parts

    /// Sub-context with specified name.
    static member (/) (vp: VarPath, name: string) =
        VarPath (vp.Parts @ [name])
    
    /// Appends one context to another.
    static member (/) (a: VarPath, b: VarPath) =
        VarPath (a.Parts @ b.Parts)

    /// Pretty string.
    member this.Pretty =
        this.Parts |> String.concat "/"



/// A context for variable creation.
[<StructuredFormatDisplay("{Pretty}")>]
type Context = {
    /// Path.
    Path: VarPath
    /// Device.
    Dev: ITensorDevice
} with 

    /// Path of context.
    static member path (ctx: Context) = ctx.Path
    
    /// Device of context.
    static member dev (ctx: Context) = ctx.Dev

    /// Root context on specified device.
    static member root dev = 
        {Path=VarPath.root; Dev=dev}

    /// Append name to context path.
    static member (/) (ctx: Context, name: string) =
        {ctx with Path=ctx.Path / name}

    /// Append another path to context path.
    static member (/) (ctx: Context, path: VarPath) =
        {ctx with Path=ctx.Path / path}

    /// Pretty string.
    member this.Pretty =
        sprintf "%A@%A" this.Path this.Dev


