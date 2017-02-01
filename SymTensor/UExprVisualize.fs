namespace SymTensor

open System.Threading
open System.Windows.Forms

open Microsoft.Msagl
open Microsoft.Msagl.Drawing
open Microsoft.Msagl.GraphViewerGdi

open Basics
open SymTensor.Compiler


module UExprVisualizer =

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
        override this.ToString () =
            match this with
            | Target ch -> sprintf "Target %s" ch
            | Src src -> sprintf "Src %d" src

    type VisArrayNDManikin =
        {Manikin:  ArrayNDManikinT
         EuId:     int
         UExpr:    UExprT
         Relation: VisManikinUExprRelation}        
        member this.Type = sprintf "%A" this.Manikin.TypeName
        member this.Shape = sprintf "%A" this.Manikin.Shape
        member this.MemManikin = {VisMemManikin.Manikin = this.Manikin.Storage}


    type Visualizer (rootExprs: UExprT list) =

        let arrayNDManikins = ResizeArray<VisArrayNDManikin> ()

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

        /// nodes for storages
        //let nodeForStorage = Dictionary<string, Node> ()

        // build graph
        let resNodes, nodeForExpr = nodesForExprs graph.RootSubgraph rootExprs

        do
            // add result nodes
            for i, resNode in List.indexed resNodes do
                let resLabelNode = newNode graph.RootSubgraph
                resLabelNode.LabelText <- sprintf "Result %d" i
                resLabelNode.Attr.FillColor <- Color.Blue
                resLabelNode.Label.FontColor <- Color.White
                newEdge resNode resLabelNode "" |> ignore

        member this.AddManikins (euId: int) (uExpr: UExprT) 
                                (trgtManikins: Map<string, ArrayNDManikinT>) 
                                (srcManikins: ArrayNDManikinT list) =

            for KeyValue(ch, manikin) in trgtManikins do
                arrayNDManikins.Add {Manikin=manikin; EuId=euId; UExpr=uExpr; Relation=Target ch} 
            for src, manikin in List.indexed srcManikins do
                arrayNDManikins.Add {Manikin=manikin; EuId=euId; UExpr=uExpr; Relation=Src src}

//            match nodeForExpr.TryFind uExpr with
//            | Some exprNode ->
//                // add manikin info node
//                let manikinNode = newNode graph.RootSubgraph // TODO: fix subgraphs
//                manikinNode.LabelText <- manikins 
//                                         |> List.map (fun (_, manikinStr, _) -> manikinStr)
//                                         |> String.concat "\n"
//                manikinNode.Attr.FillColor <- Color.Beige
//                manikinNode.Label.FontColor <- Color.Black
//                let manikinEdge = newEdge manikinNode exprNode ""
//                manikinEdge.Attr.ArrowheadAtTarget <- ArrowStyle.None
//
//                // add storage nodes
//                for label, _, storageStr in manikins do
//                    let storageNode =
//                        match nodeForStorage.TryFind storageStr with
//                        | Some storageNode -> storageNode
//                        | None -> 
//                            let n = newNode graph.RootSubgraph // TODO: fix subgraphs
//                            n.LabelText <- storageStr
//                            n.Attr.FillColor <- Color.Bisque
//                            n.Label.FontColor <- Color.Black
//                            nodeForStorage.[storageStr] <- n
//                            n
//                    newEdge storageNode manikinNode label |> ignore                       
//
//            | None -> () // ignore manikin information for unknown nodes

        /// shows the graph
        member this.Show () = 
            // form
            let form = new System.Windows.Forms.Form(Text=graph.Label.Text, 
                                                     Width=1000, Height=600)
            form.SuspendLayout()
            form.WindowState <- System.Windows.Forms.FormWindowState.Maximized

            // graph viewer
            let viewer = new GViewer(Graph=graph, Dock=DockStyle.Fill)
            viewer.PanButtonPressed <- true
            viewer.ToolBarIsVisible <- false
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
                                                      MultiSelect=false, RowHeadersVisible=false)
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
            memManikinView.SelectionChanged.Add (fun _ -> 
                match memManikinView.SelectedRows.Count with
                | 0 -> arrayNDManikinBinding.Clear()
                | 1 -> updateArrayNDManikinView (memManikinView.SelectedRows.[0].DataBoundItem :?> VisMemManikin)
                | _ -> ())

            // show
            form.ResumeLayout()
            form.ShowDialog () |> ignore


    /// active visualizer
    let active = new ThreadLocal<Visualizer option> ()   

    let build rootExprs = 
        if Debug.VisualizeUExpr then
            let v = Visualizer (rootExprs)
            active.Value <- Some v

    let getActive () =
        match active.Value with
        | Some v -> v
        | None -> failwith "no UExprVisualizer is active on the current thread"

    let show () =
        if Debug.VisualizeUExpr then
            getActive().Show()

    let finish () =
        active.Value <- None

    let addManikins euId uExpr trgtManikins srcManikins =
        if Debug.VisualizeUExpr then
            getActive().AddManikins euId uExpr trgtManikins srcManikins

            

