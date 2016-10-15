namespace SymTensor

open Microsoft.Msagl
open Microsoft.Msagl.Drawing
open Microsoft.Msagl.GraphViewerGdi

open Basics


module UExprVisualize =

    let buildGraph () =
    
        let graph = Graph ("TestGraph")

        let node1 = Node ("MyNodeId")
        node1.LabelText <- "blabla"
        //node1.Label <- Label ("my node text")
        //let label = Label("mt")
        //label.FontColor <- Color.Black
        //node1.La
        graph.AddNode node1

        let node2 = Node ("MyNodeId2")
        //node2.Label <- Label ("my node other text")
        node2.Attr.FillColor <- Color.Magenta
        graph.AddNode node2

        graph.AddPrecalculatedEdge (Edge (node1, node2, ConnectionToGraph.Connected))

        graph


    let fromUExpr rootExprs =
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
                            //match op with
                            //| UElemExpr.ULeafOp (ElemExpr.ArgElement _) -> 
                            //    graph.LayerConstraints.PinNodesToMaxLayer opNode
                            //| _ -> ()
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
            let chResNodes = nodesForExprs subgraph chExprs

            for (ch, lv), chResNode in List.zip channels chResNodes do
                let chResLabelNode = newNode subgraph
                chResLabelNode.LabelText <- sprintf "%s" ch
                chResLabelNode.Attr.FillColor <- Color.Blue
                chResLabelNode.Label.FontColor <- Color.White
                newEdge chResNode chResLabelNode "" |> ignore                
            subgraph

        and nodesForExprs (subgraph: Subgraph) exprs =
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
                            let argNode = build arg
                            let lbl = if args.Length > 1 then sprintf "%d" i else ""
                            newEdge argNode opNode lbl |> ignore
                        opNode

            exprs |> List.map build

        // build main graph
        let resNodes = nodesForExprs graph.RootSubgraph rootExprs
        for i, resNode in List.indexed resNodes do
            let resLabelNode = newNode graph.RootSubgraph
            resLabelNode.LabelText <- sprintf "Result %d" i
            resLabelNode.Attr.FillColor <- Color.Blue
            resLabelNode.Label.FontColor <- Color.White
            newEdge resNode resLabelNode "" |> ignore

        graph


    let showGraph (graph: Graph) = 
        let form = new System.Windows.Forms.Form(Text=graph.Label.Text, 
                                                 Width=1000, Height=600)
        form.WindowState <- System.Windows.Forms.FormWindowState.Maximized
        let viewer = new GViewer(Graph=graph, Dock=System.Windows.Forms.DockStyle.Fill)
        viewer.PanButtonPressed <- true
        viewer.ToolBarIsVisible <- false
        form.Controls.Add viewer
        //form.Show()
        form.ShowDialog () |> ignore

    let show exprs =
        let graph = fromUExpr exprs
        showGraph graph 


    let demo () =
        let graph = buildGraph ()
        showGraph graph 
