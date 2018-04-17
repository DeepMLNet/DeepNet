namespace SymTensor

open System.Threading
open System.Windows.Forms

open Microsoft.Msagl
open Microsoft.Msagl.Drawing
open Microsoft.Msagl.GraphViewerGdi

open Tensor.Utils
open SymTensor.Compiler


module DiagnosticsVisualizer =

    type VisMemManikin =
        {Manikin: MemManikinT}
        member this.Kind =
            match this.Manikin with
            | MemZero _ -> "Zero"
            | MemAlloc _ -> "Alloc"
            | MemExternal _ -> "External"
            | MemConst _ -> "Const"
        member this.AllocId =
            match this.Manikin with
            | MemAlloc a -> a.Id
            | _ -> 0
        member this.SizeInKB =
            match this.Manikin with
            | MemAlloc a -> a.ByteSize / 1024L
            | _ -> 0L

    type VisManikinUExprRelation =
        | Target of string
        | Src of int
        | Extra
        override this.ToString () =
            match this with
            | Target ch -> sprintf "Target %s" ch
            | Src src -> sprintf "Src %d" src
            | Extra -> "Extra"

    type VisArrayNDManikin =
        {Manikin:  ArrayNDManikinT option
         Storage:  MemManikinT option
         EuId:     int
         UExpr:    UExprT
         Relation: VisManikinUExprRelation}        
        member this.Type = 
            match this.Manikin with
            | Some m -> sprintf "%A" m.TypeName
            | None -> ""
        member this.Shape = 
            match this.Manikin with
            | Some m -> sprintf "%A" m.Shape
            | None -> ""
        member this.MemManikin = 
            match this.Manikin, this.Storage with
            | Some m, None -> {VisMemManikin.Manikin = m.Storage}
            | None, Some s -> {VisMemManikin.Manikin = s}
            | _, _ -> failwith "invalid combination of manikin/storage"


    type Visualizer (diagnostics: CompileDiagnosticsT) =

        let arrayNDManikins = ResizeArray<VisArrayNDManikin> ()
        let srcEdges = Dictionary<UExprT * int, Edge> ()

        let graph = Graph ("UExpr")
        let viewer = new GViewer(Dock=DockStyle.Fill, 
                                 PanButtonPressed=true, ToolBarIsVisible=false)

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
            for KeyValue(expr, node : Node) in nodeForExpr do
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

        and nodesForExprs (subgraph: Subgraph) exprs : Node list * IReadOnlyDictionary<_,_> =
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
                            let edge = 
                                match arg with
                                | UExpr (UExtraOp (Channel ch), [chSrc], _) ->
                                    let argNode = build chSrc
                                    let lbl = if args.Length > 1 then sprintf "%d=%s" i ch else ch
                                    newEdge argNode opNode lbl 
                                | _ -> 
                                    let argNode = build arg
                                    let lbl = if args.Length > 1 then sprintf "%d" i else ""
                                    newEdge argNode opNode lbl 
                            srcEdges.[(expr, i)] <- edge
                        opNode

            exprs |> List.map build, nodeForExpr :> IReadOnlyDictionary<_,_>

        // blinking
        let mutable blinkMarked = false
        let mutable blinkEdges : List<Edge> = []
        let mutable blinkOriginalEdgeColor : List<Edge * Color> = []
        let mutable blinkNodes : List<Node> = []
        let mutable blinkOriginalNodeColor : List<Node * Color> = []
        let doBlinkUpdate () =
            if not blinkMarked then
                blinkOriginalEdgeColor <- blinkEdges
                                          |> List.map (fun e -> 
                                                let origColor = e.Attr.Color
                                                e.Attr.Color <- Color.Blue
                                                e, origColor)
                blinkOriginalNodeColor <- blinkNodes
                                          |> List.map (fun n -> 
                                                let origColor = n.Attr.FillColor
                                                n.Attr.FillColor <- Color.Blue
                                                n, origColor)
            else
                for e, origColor in blinkOriginalEdgeColor do
                    e.Attr.Color <- origColor
                for n, origColor in blinkOriginalNodeColor do
                    n.Attr.FillColor <- origColor
            blinkMarked <- not blinkMarked
            viewer.Refresh()

        /// nodes for execution units
        let nodeForEuId = Dictionary<int, Node> ()

        // build graph
        let resNodes, nodeForExpr = nodesForExprs graph.RootSubgraph [diagnostics.UExpr]

        do
            // add result nodes
            for i, resNode in List.indexed resNodes do
                let resLabelNode = newNode graph.RootSubgraph
                resLabelNode.LabelText <- sprintf "Result %d" i
                resLabelNode.Attr.FillColor <- Color.Blue
                resLabelNode.Label.FontColor <- Color.White
                newEdge resNode resLabelNode "" |> ignore

            // add information from ExecUnits
            for eu in diagnostics.ExecUnits do

                match nodeForExpr.TryFindReadOnly eu.Expr with
                | Some euNode -> 
                    nodeForEuId.[eu.Id] <- euNode

                    let text = 
                        eu.Items
                        |> List.indexed
                        |> List.map (fun (eiIdx, ei) -> 
                            sprintf "#%d: %s" eiIdx ei.VisualizationText)
                        |> String.concat "\n"
                    if Debug.VisualizeExecItems && text.Length > 0 then
                        let eiNode = newNode graph.RootSubgraph
                        eiNode.LabelText <- text
                        eiNode.Attr.FillColor <- Color.Black
                        eiNode.Label.FontColor <- Color.White
                        newEdge euNode eiNode "" |> ignore
                | None -> ()

                for KeyValue(ch, (manikin, shrd)) in eu.Channels do
                    arrayNDManikins.Add {Manikin=Some manikin; Storage=None; EuId=eu.Id; UExpr=eu.Expr; Relation=Target ch} 
                for src, manikin in List.indexed eu.Srcs do
                    arrayNDManikins.Add {Manikin=Some manikin; Storage=None; EuId=eu.Id; UExpr=eu.Expr; Relation=Src src}
                for extraMem in eu.ExtraMem do
                    arrayNDManikins.Add {Manikin=None; Storage=Some extraMem; EuId=eu.Id; UExpr=eu.Expr; Relation=Extra}             
                
        /// shows the visualization
        member this.Show () = 
            // add execution unit id to nodes
            for KeyValue(euId, node) in Map.ofDictionary nodeForEuId do
                node.LabelText <- sprintf "%d: %s" euId node.LabelText

            // form
            let form = new System.Windows.Forms.Form(Text=graph.Label.Text, 
                                                     Width=1000, Height=600)
            form.SuspendLayout()
            form.WindowState <- System.Windows.Forms.FormWindowState.Maximized

            // graph viewer
            viewer.Graph <- graph
            //viewer.tool
            form.Controls.Add viewer

            // splitter ArrayNDManikins / MemManikins
            let manikinSplitter = new SplitContainer(Orientation=Orientation.Horizontal, 
                                                     Dock=DockStyle.Right,
                                                     Width=350)            
            form.Controls.Add manikinSplitter

            // MemManikins data binding
            let visMemManikins = 
                arrayNDManikins 
                |> Seq.map (fun a -> a.MemManikin) 
                |> Set.ofSeq
                |> Set.toList
                |> List.sortByDescending (fun m -> match m.Manikin with
                                                   | MemAlloc a -> a.ByteSize
                                                   | _ -> 0L)
            let memManikinBinding = new BindingSource()
            for visMemManikin in visMemManikins do
                memManikinBinding.Add visMemManikin |> ignore

            // MemManikins viewer
            let memManikinView = new DataGridView(Dock=DockStyle.Fill, 
                                                  DataSource=memManikinBinding,
                                                  AutoGenerateColumns=false,
                                                  AutoSizeColumnsMode=DataGridViewAutoSizeColumnsMode.AllCells,
                                                  AllowUserToResizeRows=false,
                                                  SelectionMode=DataGridViewSelectionMode.FullRowSelect,
                                                  MultiSelect=false, RowHeadersVisible=false)
            memManikinView.Columns.Add (new DataGridViewTextBoxColumn(Name="Kind", 
                                                                      DataPropertyName="Kind",
                                                                      SortMode=DataGridViewColumnSortMode.Automatic)) |> ignore
            memManikinView.Columns.Add (new DataGridViewTextBoxColumn(Name="Id", 
                                                                      DataPropertyName="AllocId",
                                                                      SortMode=DataGridViewColumnSortMode.Automatic)) |> ignore
            memManikinView.Columns.Add (new DataGridViewTextBoxColumn(Name="Size (KB)", 
                                                                      DataPropertyName="SizeInKB",
                                                                      SortMode=DataGridViewColumnSortMode.Automatic)) |> ignore
            //memManikinView.Sort(memManikinView.Columns.[2], System.ComponentModel.ListSortDirection.Descending)
            memManikinView.AutoResizeColumns()
            manikinSplitter.Panel1.Controls.Add memManikinView

            // MemManikins info text
            let memManikinInfo = new TextBox(ReadOnly=true, Dock=DockStyle.Bottom, Height=40)
            manikinSplitter.Panel1.Controls.Add memManikinInfo
            let manikinsTotalSize = visMemManikins |> List.sumBy (fun v -> v.SizeInKB)
            memManikinInfo.Text <- sprintf "Total allocated memory: %.3f MB" (float manikinsTotalSize / 1024.0)

            // ArrayNDManikins viewer
            let arrayNDManikinBinding = new BindingSource()
            let arrayNDManikinView = new DataGridView(Dock=DockStyle.Fill,
                                                      DataSource=arrayNDManikinBinding,
                                                      AutoGenerateColumns=false,
                                                      AutoSizeColumnsMode=DataGridViewAutoSizeColumnsMode.AllCells,
                                                      AllowUserToResizeRows=false,
                                                      SelectionMode=DataGridViewSelectionMode.FullRowSelect,
                                                      MultiSelect=true, RowHeadersVisible=false)
            arrayNDManikinView.Columns.Add (new DataGridViewTextBoxColumn(Name="ExecUnit", 
                                                                          DataPropertyName="EuId")) |> ignore
            arrayNDManikinView.Columns.Add (new DataGridViewTextBoxColumn(Name="Relation", 
                                                                          DataPropertyName="Relation")) |> ignore
            arrayNDManikinView.Columns.Add (new DataGridViewTextBoxColumn(Name="Type", 
                                                                          DataPropertyName="Type")) |> ignore
            arrayNDManikinView.Columns.Add (new DataGridViewTextBoxColumn(Name="Shape", 
                                                                          DataPropertyName="Shape")) |> ignore
            manikinSplitter.Panel2.Controls.Add arrayNDManikinView

            let updateArrayNDManikinView visMemManikin = 
                let visArrayNDManikins =
                    arrayNDManikins
                    |> Seq.filter (fun a -> a.MemManikin = visMemManikin)
                    |> Seq.sortBy (fun a -> a.EuId)
                arrayNDManikinBinding.Clear()
                for visArrayNDManikin in visArrayNDManikins do
                    arrayNDManikinBinding.Add visArrayNDManikin |> ignore
                arrayNDManikinView.AutoResizeColumns()
                arrayNDManikinView.SelectAll()

            memManikinView.SelectionChanged.Add (fun _ -> 
                match memManikinView.SelectedRows.Count with
                | 0 -> arrayNDManikinBinding.Clear()
                | 1 -> updateArrayNDManikinView (memManikinView.SelectedRows.[0].DataBoundItem :?> VisMemManikin)
                | _ -> ())

            arrayNDManikinView.SelectionChanged.Add (fun _ ->
                let vis = 
                    arrayNDManikinView.SelectedRows
                    |> Seq.cast<DataGridViewRow>
                    |> Seq.choose (fun row -> if row.DataBoundItem <> null then 
                                                  Some (row.DataBoundItem :?> VisArrayNDManikin)
                                              else None)
                    |> Seq.toList
                let edges = 
                    vis
                    |> List.choose (function | {UExpr=uExpr; Relation=Src src} -> Some (uExpr, src)
                                             | _ -> None)
                    |> List.choose (fun (uexpr, src) -> srcEdges.TryFind (uexpr, src))
                let nodes =
                    vis
                    |> List.choose (function | {EuId=euId; Relation=Extra} -> nodeForEuId.TryFind euId
                                             | _ -> None)
                blinkEdges <- edges
                blinkNodes <- nodes)

            // blink timer
            let blinkTimer = new Timer(Interval=500)
            blinkTimer.Tick.Add (fun _ -> doBlinkUpdate())
            blinkTimer.Start()

            // show
            form.ResumeLayout()
            form.ShowDialog () |> ignore


    /// Visualizes compilation information.
    let visualize (diagnostics: CompileDiagnosticsT) =
        let vis = Visualizer diagnostics
        vis.Show()


            

