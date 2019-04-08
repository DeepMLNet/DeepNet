namespace Tensor.Expr.Opt

open System
open System.Reflection

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops


/// Declares that this type is an optimizer.
[<AttributeUsage(AttributeTargets.Class)>]
type OptimizerAttribute () =
    inherit System.Attribute()


/// An optimizer.
type IOptimizer =
    /// Position in optimizer execution order.
    abstract Order: int

type IUExprOptimizer =
    /// Perform optimization of the specified expression.
    abstract Optimize: subOpt:(BaseExpr -> BaseExpr) -> expr:UExpr -> UExpr

type IMultiChannelOptimizer =
    /// Perform optimization of the specified expression.
    abstract Optimize: subOpt:(BaseExpr -> BaseExpr) -> expr:MultiChannelExpr -> MultiChannelExpr


/// Optimizer access functions.
module Optimizer = 

    /// Registered optimiers.
    let mutable private optimizers: Map<string, IOptimizer> = Map.empty

    /// Register the specified optimizer type.
    let private registerOpt (optType: Type) =
        let opt = Activator.CreateInstance (optType) :?> IOptimizer
        let name = optType.FullName
        if optimizers.ContainsKey name then
            failwithf "Optimizer %s has already been registered." name
        optimizers <- optimizers |> Map.add name opt

    /// Registers all optimizers from the specified assembly.
    let register (asm: Assembly) =
        let optTypes =
            asm.GetTypes()
            |> Seq.filter (fun typ ->
                typ.CustomAttributes 
                |> Seq.exists (fun a -> a.AttributeType = typeof<OptimizerAttribute>))

        for optType in optTypes do
            registerOpt optType

    do register (Assembly.GetExecutingAssembly())

    /// Gets the optimizers with the specified names.
    /// If no names have been specified, all regisitered optimizers are returned.
    let private getOptimizers (optNames: string list option) =
        match optNames with
        | Some optNames ->
            optNames
            |> List.map (fun optName ->
                match optimizers |> Map.tryFind optName with
                | Some opt -> opt
                | None -> failwithf "The optimizer %s was not found." optName)
        | None -> 
            optimizers
            |> Map.toList
            |> List.map snd
        |> List.sortBy (fun opt -> opt.Order)
        
    /// Apply optimizer to expression once.
    let private applyOnce (optRec: BaseExpr -> BaseExpr) (opt: IOptimizer) (baseExpr: BaseExpr) =
        match opt, baseExpr with
        | :? IUExprOptimizer as opt, ExprChs.Single uExpr -> 
            opt.Optimize optRec uExpr |> UExpr.baseExpr
        | :? IMultiChannelOptimizer as opt, ExprChs.Multi mcExpr -> 
            opt.Optimize optRec mcExpr |> MultiChannelExpr.baseExpr
        | _ -> baseExpr

    /// Apply optimizers to expression until no more optimizations are performed.
    let rec applyIterated (optRec: BaseExpr -> BaseExpr) (opts: IOptimizer list) (baseExpr: BaseExpr) =       
        let rec applyLoop optQueue expr =
            match optQueue with
            | opt :: rOptQueue ->
                let optExpr = applyOnce optRec opt expr
                if optExpr = baseExpr then 
                    applyLoop rOptQueue optExpr
                else
                    applyLoop opts optExpr
            | [] -> expr
        applyLoop opts baseExpr

    /// Optimizes the specified expression tree.
    let optimize (optNames: string list option) (baseExpr: BaseExpr) =
        let opts = getOptimizers optNames

        let optimized = Dictionary<BaseExpr, BaseExpr> ()
        let rec optRec expr =            
            optimized.GetOrAdd expr (fun _ ->
                expr
                |> BaseExpr.mapArgs (BaseExprCh.map optRec)
                |> applyIterated optRec opts)
        optRec baseExpr



