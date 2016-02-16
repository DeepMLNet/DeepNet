namespace ExprNS

open System

open Util
open VarSpec
open SizeSymbolTypes
open ArrayNDNS
open ArrayNDNS.ArrayND


[<AutoOpen>]
module VarEnvTypes = 
    /// variable environment
    type VarEnvT = Map<IVarSpec, IHasLayout>

module VarEnv = 
    /// add variable value to environment
    let add (var: Expr.ExprT<'T>) (value: ArrayNDT<'T>) (varEnv: VarEnvT) =
        let vs = Expr.extractVar var
        Map.add (vs :> IVarSpec) (value :> IHasLayout) varEnv

    let getVarSpecT (vs: VarSpecT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        varEnv.[vs] :?> ArrayNDT<'T>

    /// get variable value from environment
    let get (var: Expr.ExprT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        getVarSpecT (Expr.extractVar var) varEnv

    /// empty variable environment
    let (empty: VarEnvT) =
        Map.empty

    /// joins two variable environments
    let join (a: VarEnvT) (b: VarEnvT) =
        Map.join a b
   
    /// enhances an existing size symbol environment from the variables occuring in the expression  
    let enhanceSizeSymbolEnvUsingExprs sizeSymbolEnv exprs (varEnv: VarEnvT) =
        let varSymShapes = 
            exprs
            |> Seq.map Expr.extractVars
            //|> Seq.map (fun vss -> Set.map (fun vs -> vs :> IVarSpec) vss)
            |> Set.unionMany
            |> Set.toSeq 
            |> Seq.map (fun vs -> VarSpec.name vs, VarSpec.shape vs)
            |> Map.ofSeq
        let varValShapes = 
            varEnv 
            |> Map.toSeq 
            |> Seq.map (fun (vs, ary) -> VarSpec.name vs, ArrayND.shape ary) 
            |> Map.ofSeq
        ShapeSpec.enhanceSizeSymbolEnv sizeSymbolEnv varValShapes varSymShapes

    /// builds a size symbol environment from the variables occuring in the expression
    let inferSizeSymbolEnvUsingExprs exprs (varEnv: VarEnvT) =
        enhanceSizeSymbolEnvUsingExprs SizeSymbolEnv.empty exprs varEnv


[<AutoOpen>]
module EvalEnvTypes =
    /// Information neccessary to evaluate an expression.
    /// Contains numeric values for variables and size symbols.
    type EvalEnvT = {VarEnv: VarEnvT; SizeSymbolEnv: SizeSymbolEnvT}


module EvalEnv = 
    open VarEnv

    /// empty EvalEnvT
    let empty = 
        {VarEnv = Map.empty; SizeSymbolEnv = Map.empty}

    /// Create an EvalEnvT using the specified VarEnvT and infers the size symbol values
    /// from the given expressions.
    let create varEnv exprs = 
        {VarEnv = varEnv; SizeSymbolEnv = inferSizeSymbolEnvUsingExprs exprs varEnv;}

    /// Enhances an existing EvalEnvT using the specified VarEnvT and infers the size symbol values
    /// from the given expressions.
    let enhance varEnv exprs evalEnv =
        let joinedVarEnv = VarEnv.join evalEnv.VarEnv varEnv
        {VarEnv = joinedVarEnv;
         SizeSymbolEnv = enhanceSizeSymbolEnvUsingExprs evalEnv.SizeSymbolEnv exprs joinedVarEnv;}

    //let fromVarEnv varEnv =
    //    {empty with VarEnv = varEnv;}

    //let addSizeSymbolsFromExpr expr evalEnv =
    //    create evalEnv.VarEnv expr

