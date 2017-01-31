namespace SymTensor

open System.Threading
open Microsoft.Msagl
open Microsoft.Msagl.Drawing
open Microsoft.Msagl.GraphViewerGdi

open Basics



type UExprVisualizer (rootExprs: UExprT list) =

    let graph = Graph ("UExpr")

    let mutable ids = 0
    let newId () =
        ids <- ids + 1
        sprintf "%d" ids

    let newSubgraph (parent: Subgraph) =
        let subgraph = Subgraph(newId())
        parent.AddSubgraph subgraph
        subgraph

    let newNode (subgraph: Subgraph) =
        let node = Node(newId())
        graph.AddNode node
        subgraph.AddNode node
        node

    let newEdge (src: Node) (trgt: Node) label =
        graph.AddEdge (src.Id, label, trgt.Id)

    let getOpText op =
        match op with
        | ULeafOp (Expr.Var vs) ->
            sprintf "%s %A" vs.Name vs.NShape, Color.Yellow
        | UUnaryOp (Expr.StoreToVar vs) ->
            sprintf "%s %A" vs.Name vs.NShape, Color.Red

        | ULeafOp op -> sprintf "%A" op, Color.Transparent
        | UUnaryOp op -> sprintf "%A" op, Color.Transparent
        | UBinaryOp op -> sprintf "%A" op, Color.Transparent
        | UNaryOp op -> sprintf "%A" op, Color.Transparent
        | UExtraOp op -> sprintf "%A" op, Color.Transparent

    let getElemOpText op =
        match op with
        | UElemExpr.ULeafOp (ElemExpr.ArgElement ((ElemExpr.Arg pos, idx), _)) ->
            let idxStr = (sprintf "%A" idx).Replace("\"", "")
            sprintf "%d %s" pos idxStr, Color.Yellow
        | UElemExpr.ULeafOp (ElemExpr.Const cs) ->
            sprintf "%A" cs.Value, Color.Transparent
        | UElemExpr.ULeafOp (ElemExpr.SizeValue (ss, _)) ->
            sprintf "%d" (SizeSpec.eval ss), Color.Transparent

        | UElemExpr.UUnaryOp ElemExpr.Negate -> "-", Color.Transparent

        | UElemExpr.UBinaryOp ElemExpr.Add -> "+", Color.Transparent
        | UElemExpr.UBinaryOp ElemExpr.Substract -> "-", Color.Transparent
        | UElemExpr.UBinaryOp ElemExpr.Multiply -> "*", Color.Transparent
        | UElemExpr.UBinaryOp ElemExpr.Divide -> "/", Color.Transparent
        | UElemExpr.UBinaryOp ElemExpr.Modulo -> "%", Color.Transparent
        | UElemExpr.UBinaryOp ElemExpr.Power -> "***", Color.Transparent

        //| UElemExpr.ULeafOp op -> sprintf "%A" op, Color.Transparent
        | UElemExpr.UUnaryOp op -> sprintf "%A" op, Color.Transparent
        | UElemExpr.UBinaryOp op -> sprintf "%A" op, Color.Transparent


    let subgraphForElemExpr (parent: Subgraph) elements =
        match elements with 
        | Elements (shp, {Expr=elemExpr}) ->
            let subgraph = newSubgraph parent
            subgraph.LabelText <- sprintf "Elements %A" shp

            let nodeForElemExpr = Dictionary<UElemExpr.UElemExprT, Node> (HashIdentity.Reference)
            let rec build elemExpr =
                match nodeForElemExpr.TryFind elemExpr with
                | Some node -> node
                | None ->
                    match elemExpr with
                    | UElemExpr.UElemExpr (op, args, tn) ->
                        let opNode = newNode subgraph
                        nodeForElemExpr.[elemExpr] <- opNode
                        let txt, color = getElemOpText op
                        opNode.LabelText <- txt
                        opNode.Attr.FillColor <- color
                        match op with
                        | UElemExpr.ULeafOp (ElemExpr.ArgElement _) -> 
                            opNode.Attr.Shape <- Shape.Diamond
                            //graph.LayerConstraints.PinNodesToMaxLayer opNode
                        | _ -> ()
                        opNode.UserData <- elemExpr
                        for i, arg in List.indexed args do
                            let argNode = build arg
                            let lbl = if args.Length > 1 then sprintf "%d" i else ""
                            newEdge argNode opNode lbl |> ignore
                        opNode
            build elemExpr |> ignore
            subgraph
        | _ -> failwith "not an elements expression"

    let rec subgraphForLoop (parent: Subgraph) (loopSpec: ULoopSpecT) =
        let subgraph = newSubgraph parent
        subgraph.LabelText <- sprintf "Loop %A" loopSpec.Length
            
        let channels = loopSpec.Channels |> Map.toList
        let chExprs = channels |> List.map (fun (ch, lv) -> lv.UExpr)
        let chResNodes, nodeForExpr = nodesForExprs subgraph chExprs

        let chResLabelNodes = 
            List.zip channels chResNodes
            |> List.map (fun ((ch, lv), chResNode) ->
                // add channel nodes
                let chResLabelNode = newNode subgraph
                chResLabelNode.LabelText <- sprintf "%s (SliceDim=%d)" ch lv.SliceDim
                chResLabelNode.Attr.FillColor <- Color.Blue
                chResLabelNode.Label.FontColor <- Color.White
                newEdge chResNode chResLabelNode "" |> ignore                
                ch, chResLabelNode)
            |> Map.ofList

        // annotate loop variables
        for KeyValue(expr, node) in nodeForExpr do
            match expr with
            | UExpr (ULeafOp (Expr.Var vs), _, _) ->
                let li = loopSpec.Vars.[vs]
                match li with
                | Expr.ConstArg argIdx ->
                    node.LabelText <- sprintf "Arg %d\n%s" argIdx node.LabelText
                    node.Attr.FillColor <- Color.Yellow
                    node.Attr.Shape <- Shape.Diamond
                | Expr.SequenceArgSlice sas ->
                    node.LabelText <- 
                        sprintf "Arg %d (SliceDim=%d)\n%s" sas.ArgIdx sas.SliceDim node.LabelText
                    node.Attr.FillColor <- Color.Yellow
                    node.Attr.Shape <- Shape.Diamond
                | Expr.PreviousChannel pc ->
                    // add loop to channel
                    let ln = chResLabelNodes.[pc.Channel]
                    newEdge ln node (sprintf "delay %A" pc.Delay) |> ignore
                    // add initial
                    let initialNode = newNode subgraph
                    initialNode.LabelText <- sprintf "Initial %d" pc.InitialArg
                    initialNode.Attr.FillColor <- Color.Yellow
                    initialNode.Attr.Shape <- Shape.Diamond
                    node.Attr.FillColor <- Color.Transparent
                    newEdge initialNode node "" |> ignore
                | _ -> ()
            | _ -> ()                            

        subgraph

    and nodesForExprs (subgraph: Subgraph) exprs : Node list * Map<UExprT, Node> =
        let nodeForExpr = Dictionary<UExprT, Node> (HashIdentity.Reference)
        let rec build expr =
            match nodeForExpr.TryFind expr with
            | Some node -> node
            | None ->
                match expr with
                | UExpr (op, args, metadata) ->
                    let opNode = 
                        match op with
                        | UExtraOp (Elements (shp, elemFunc)) ->
                            subgraphForElemExpr subgraph (Elements (shp, elemFunc))
                            :> Node
                        | UExtraOp (Loop loopSpec) ->
                            subgraphForLoop subgraph loopSpec
                            :> Node
                        | _ ->
                            let node = newNode subgraph
                            let txt, color = getOpText op
                            node.LabelText <- txt
                            node.Attr.FillColor <- color                            
                            node
                    opNode.UserData <- expr
                    nodeForExpr.[expr] <- opNode

                    for i, arg in List.indexed args do
                        match arg with
                        | UExpr (UExtraOp (Channel ch), [chSrc], _) ->
                            let argNode = build chSrc
                            let lbl = if args.Length > 1 then sprintf "%d=%s" i ch else ch
                            newEdge argNode opNode lbl |> ignore
                        | _ -> 
                            let argNode = build arg
                            let lbl = if args.Length > 1 then sprintf "%d" i else ""
                            newEdge argNode opNode lbl |> ignore
                    opNode

        exprs |> List.map build, Map.ofDictionary nodeForExpr

    // build main graph
    do
        let resNodes, _ = nodesForExprs graph.RootSubgraph rootExprs
        for i, resNode in List.indexed resNodes do
            let resLabelNode = newNode graph.RootSubgraph
            resLabelNode.LabelText <- sprintf "Result %d" i
            resLabelNode.Attr.FillColor <- Color.Blue
            resLabelNode.Label.FontColor <- Color.White
            newEdge resNode resLabelNode "" |> ignore

    member this.Show () = 
        let form = new System.Windows.Forms.Form(Text=graph.Label.Text, 
                                                 Width=1000, Height=600)
        form.WindowState <- System.Windows.Forms.FormWindowState.Maximized
        let viewer = new GViewer(Graph=graph, Dock=System.Windows.Forms.DockStyle.Fill)
        viewer.PanButtonPressed <- true
        viewer.ToolBarIsVisible <- false
        form.Controls.Add viewer
        //form.Show()
        form.ShowDialog () |> ignore


[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module UExprVisualizer =

    let active = new ThreadLocal<UExprVisualizer option> ()   

    let build rootExprs = 
        if Debug.VisualizeUExpr then
            let v = UExprVisualizer (rootExprs)
            active.Value <- Some v

    let getActive () =
        match active.Value with
        | Some v -> v
        | None -> failwith "no visualization is active"

    let show () =
        if Debug.VisualizeUExpr then
            getActive().Show()

    let finish () =
        active.Value <- None


