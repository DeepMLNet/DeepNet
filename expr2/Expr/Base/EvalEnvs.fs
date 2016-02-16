namespace ExprNS

open System

open Util
open VarSpec
open SizeSymbolTypes
open ArrayNDNS
open ArrayNDNS.ArrayND


module VarEnv = 

    /// variable environment
    type VarEnvT = Map<IVarSpec, IHasLayout>

    /// add variable value to environment
    let add (var: Expr.ExprT<'T>) (value: ArrayNDT<'T>) (varEnv: VarEnvT) =
        let vs = Expr.extractVar var
        Map.add (vs :> IVarSpec) (value :> IHasLayout) varEnv

    /// get variable value from environment
    let get (var: Expr.ExprT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        let vs = Expr.extractVar var
        varEnv.[vs] :?> ArrayNDT<'T>

    /// empty variable environment
    let (empty: VarEnvT) =
        Map.empty

    /// builds a size symbol environment from the variables occuring in the expression
    let inferSizeSymbolEnvUsingExprs exprs (varEnv: VarEnvT) =
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
        ShapeSpec.inferSizeSymbolEnv varValShapes varSymShapes


module EvalEnv = 
    open VarEnv

    /// Information neccessary to evaluate an expression.
    /// Contains numeric values for variables and size symbols.
    type EvalEnvT = {VarEnv: VarEnvT; SizeSymbolEnv: SizeSymbolEnvT}

    /// empty EvalEnvT
    let empty = 
        {VarEnv = Map.empty; SizeSymbolEnv = Map.empty}

    /// create an EvalEnvT using the specified VarEnvT and infers the size symbol values
    /// from the given expression.
    let create varEnv exprs = 
        {VarEnv = varEnv; SizeSymbolEnv = inferSizeSymbolEnvUsingExprs exprs varEnv;}

    //let fromVarEnv varEnv =
    //    {empty with VarEnv = varEnv;}

    //let addSizeSymbolsFromExpr expr evalEnv =
    //    create evalEnv.VarEnv expr

