namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Backend



type ParamaterVar = {
    Placeholder: Var
    Init: ITensor -> unit
} 


type ParameterInstance = {
    ParameterVar: ParamaterVar
    Expr: UExpr
    Data: ITensor
}
    

type ParameterGroup = private {
    StoreBasePath: ContextPath
    ParameterVars: ParamaterVar list
    ContainedVars: Set<VarName>
} with
    static member create (storePath: ContextPath): ParameterGroup =
        {
            StoreBasePath = storePath
            ParameterVars = List.empty
            ContainedVars = Set.empty
        }

    static member noInit (_: #ITensor) = ()

    static member addUntyped (var: Var) (init: ITensor -> unit) (pg: ParameterGroup) =
        if pg.ContainedVars |> Set.contains var.Name then
            failwithf "ParameterGroup %A already contains variable %A." pg.StoreBasePath var
        let pv = {
            Placeholder = var
            Init = init
        }
        {pg with 
            ParameterVars = pv :: pg.ParameterVars
            ContainedVars = pg.ContainedVars |> Set.add var.Name
        }

    static member add (var: Var<'T>) (init: Tensor<'T> -> unit) (pg: ParameterGroup) =
        let initUntyped (t: ITensor) = init (t :?> Tensor<'T>)
        ParameterGroup.addUntyped var.Untyped initUntyped pg

    static member inst (sizeEnv: SymSizeEnv) (pg: ParameterGroup) =
        // substitute symbolic sizes into all variables 
        let pvs =
            pg.ParameterVars
            |> List.map (fun pv -> {pv with Placeholder = pv.Placeholder |> Var.substSymSizes sizeEnv})

        // check that shapes of all variables can be evaluated
        for pv in pvs do
            if not (ShapeSpec.canEval pv.Placeholder.Shape) then
                failwithf "Cannot evaluate shape of variable %A." pv.Placeholder

        // group variables by data type and storage device
        let pvGroups =
            pg.ParameterVars
            |> List.groupBy (fun pv -> pv.Placeholder.TypeName, pv.Placeholder.Dev)
            |> Map.ofList
        
        // for each group perform instantiation
        let pvGroupStoragesAndInsts =
            pvGroups
            |> Map.map (fun (typeName, dev) pvs ->
                // create symbolic group storage variable
                let size = pvs |> List.sumBy (fun pv -> ShapeSpec.nElem pv.Placeholder.Shape)
                let name = pg.StoreBasePath / sprintf "<%A@%A>" typeName dev
                let var = Var.make (VarName name, typeName.Type, dev, [size])
                let varExpr = UExpr var
                
                // create group storage tensor
                let data = ITensor.zeros typeName.Type dev [SizeSpec.eval size]

                // slice group storage variable/tensor to obtain expressions/subtensors for each parameter
                let pvInsts =
                    (SizeSpec.zero, pvs)
                    ||> List.mapFold (fun pos pv ->
                        let size = ShapeSpec.nElem pv.Placeholder.Shape
                        let varSlice = varExpr.[pos .. pos + size - 1L] |> UExpr.reshape pv.Placeholder.Shape
                        let nPos, nSize = SizeSpec.eval pos, SizeSpec.eval size
                        let nShape = ShapeSpec.eval pv.Placeholder.Shape
                        let dataSlice = data.[nPos .. nPos + nSize - 1L] |> ITensor.reshape nShape
                        (pv.Placeholder.Name, {ParameterVar=pv; Expr=varSlice; Data=dataSlice}), pos + size)
                    |> fst
                    |> Map.ofList

                data, pvInsts)

        // extract storage for each group
        let pvGroupStorages = pvGroupStoragesAndInsts |> Map.map (fun _ (storage, insts) -> storage)

        // merge instantiations from all groups 
        let pvInsts = 
            pvGroupStoragesAndInsts 
            |> Map.toSeq
            |> Seq.map (fun (_, (storage, insts)) -> insts)
            |> Map.joinMany

        // build and initalize
        let pgi = {
            ParameterGroup = pg
            Storages = pvGroupStorages
            Insts = pvInsts
        }
        ParameterGroupInstance.init pgi
        pgi

    // now, we need a way to substitute these values into the expression



and ParameterGroupInstance = {
    ParameterGroup: ParameterGroup
    Storages: Map<TypeName * ITensorDevice, ITensor>
    Insts: Map<VarName, ParameterInstance>
} with 

    static member init (pgi: ParameterGroupInstance) =
        for KeyValue (varName, pi) in pgi.Insts do
            pi.ParameterVar.Init pi.Data

    //member this.Use (expr: UExpr) =
        
