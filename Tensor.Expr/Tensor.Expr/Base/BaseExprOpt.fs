namespace Tensor.Expr.Base

open System
open System.Reflection

open DeepNet.Utils



/// Declares that this type is an optimizer and should be
/// automatically added to the optimizer registry.
[<AttributeUsage(AttributeTargets.Class)>]
type OptimizerAttribute () =
    inherit System.Attribute()


/// An optimizer.
type IOptimizer =
    /// Position in optimizer execution order.
    abstract Order: int
    /// Perform optimization of the specified expression.
    abstract Optimize: data:OptimizerData -> expr:BaseExpr -> BaseExpr


/// Information about an expression that may help an optimizer to perform its
/// optimizations.
and OptimizerData = {
    /// Apply the optimizers with the current settings to the expression tree.
    SubOptimize:   BaseExpr -> BaseExpr 
    /// Information about the expression tree being optimized.
    ExprGroup:     BaseExprGroup
    ///  Active optimizer.
    Optimizers:    IOptimizer list
}


/// Optimizer access functions.
module BaseExprOpt = 

    /// Registered optimiers.
    let mutable private regOpts: Map<string, IOptimizer> = Map.empty

    /// Register the specified optimizer type.
    /// It is instantiated using the default constructor.
    let register (optType: Type) =
        lock regOpts (fun () -> 
            let opt = Activator.CreateInstance (optType) :?> IOptimizer
            let name = optType.FullName
            regOpts <- regOpts |> Map.add name opt
        )

    /// Registers all optimizers from the specified assembly.
    /// Optimizers are identified by the OptimizerAttribute.
    let registerAssembly (asm: Assembly) =
        let optTypes =
            asm.GetTypes()
            |> Seq.filter (fun typ ->
                typ.CustomAttributes 
                |> Seq.exists (fun a -> a.AttributeType = typeof<OptimizerAttribute>))

        for optType in optTypes do
            register optType

    do registerAssembly (Assembly.GetExecutingAssembly())

    /// Gets the optimizers with the specified names.
    let getOptimizers (optNames: string list) =
        optNames
        |> List.map (fun optName ->
            match regOpts |> Map.tryFind optName with
            | Some opt -> opt
            | None -> failwithf "The optimizer %s was not found." optName)
        |> List.sortBy (fun opt -> opt.Order)
        
    /// Gets all registered optimizers.
    let allOptimizers () =
        regOpts
        |> Map.toList
        |> List.map snd
        |> List.sortBy (fun opt -> opt.Order)

    ///// Apply optimizer to expression once.
    //let private applyOnce (data: OptimizerData) (opt: IOptimizer) (baseExpr: BaseExpr) =
    //    match opt, baseExpr with
    //    | :? IUExprOptimizer as opt, ExprChs.Single uExpr -> 
    //        opt.Optimize data uExpr |> UExpr.baseExpr
    //    | :? IMultiChannelOptimizer as opt, ExprChs.Multi mcExpr -> 
    //        opt.Optimize data mcExpr |> MultiChannelExpr.baseExpr
    //    | _ -> baseExpr

    /// Apply optimizers to expression until no more optimizations are performed.
    let rec applyIterated (data: OptimizerData) (opts: IOptimizer list) (baseExpr: BaseExpr) =       
        let rec applyLoop (optQueue: IOptimizer list) expr =
            match optQueue with
            | opt :: rOptQueue ->
                let optExpr = opt.Optimize data expr
                if optExpr = baseExpr then 
                    applyLoop rOptQueue optExpr
                else
                    applyLoop opts optExpr
            | [] -> expr
        applyLoop opts baseExpr

    /// Optimizes the expression tree using the specified optimizers.
    let rec optimize (opts: IOptimizer list) (baseExpr: BaseExpr) =
        let data = {
            SubOptimize = optimize opts
            ExprGroup = BaseExprGroup [baseExpr]
            Optimizers = opts
        }        

        let optimized = Dictionary<BaseExpr, BaseExpr> ()
        let rec optRec expr =      
            optimized.IGetOrAdd expr (fun _ ->
                expr
                |> BaseExpr.mapArgs (BaseExprCh.map optRec)
                |> applyIterated data opts)
        optRec baseExpr



