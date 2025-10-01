import os
import json
import networkx as nx
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, callback_context
import dash_cytoscape as cyto
import plotly.colors as pc

# Load extra layouts for Cytoscape
cyto.load_extra_layouts()


def get_plotly_color(colormap, data_values, threshold: float) -> str:
    """
    Return a hex color from a Plotly colorscale.

    - data_min/data_max: your data range (include threshold via min/max).
    - threshold: pivot point where the diverging scale should center.
    - value: the actual data value to map.

    Returns a '#RRGGBB' string.
    """

    data_max = max(data_values)

    if threshold in [float("inf"), None]:
        zmax = data_max
        threshold = data_max / 2 if data_max > 0 else 0
    else:
        zmax = max(data_max, threshold)

    t = []

    for i in data_values:
        if i is None:
            t.append(0)  # Map None values to 0
        elif i < threshold:
            t.append((i / threshold) * 0.5) if threshold > 0 else t.append(0)
        else:
            if (zmax - threshold) > 0:
                t.append(((i - threshold) / (zmax - threshold)) * 0.5 + 0.5)
            else:
                t.append(0.5)

    # continuous: list of [pos, color] pairs
    return pc.sample_colorscale(colormap, t)


def build_file_hierarchy(root_dir: str) -> dict:
    """
    Walk through root_dir and build a nested dict representing folder/file hierarchy.
    NDJSON files are treated as leaf nodes.
    """
    hierarchy = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)
        node = hierarchy if rel == "." else _get_node(hierarchy, rel.split(os.sep))
        
        for fname in sorted(f for f in filenames if f.endswith(".ndjson")):
            node[fname] = None
    return hierarchy


def _get_node(tree: dict, path_parts: list) -> dict:
    """Navigate or create nested dicts per path."""
    node = tree
    for part in path_parts:
        node = node.setdefault(part, {})
    return node


def get_options(tree: dict, path: list) -> list:
    """Return sorted keys at a nested path in the tree."""
    node = tree
    for key in path:
        node = node.get(key, {})
    return sorted(node.keys()) if isinstance(node, dict) else []


def load_ndjson_graphs(filepath: str) -> list:
    """Load all graphs from an NDJSON file."""
    graphs = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                G = nx.cytoscape_graph(data)
                graphs.append(G)
    return graphs


def _make_dropdown(
    level: int, path: list, value=None, file_hierarchy=None
) -> dcc.Dropdown:
    """Create a dropdown for given level based on current path."""
    return dcc.Dropdown(
        id={"type": "dynamic-dropdown", "level": level},
        options=[{"label": o, "value": o} for o in get_options(file_hierarchy, path)],
        value=value,
        placeholder="Select...",
        style={"marginBottom": "10px"},
    )


def _info_panel() -> html.Div:
    """Return the element-info panel layout."""
    return html.Div(
        [
            html.H3("Element Information"),
            html.Div(
                html.P("Click on a node or edge to see its attributes"),
                id="element-info",
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "minHeight": "550px",
                    "overflowY": "auto",
                },
            ),
        ]
    )


def _container_style(side: str) -> dict:
    """Return styling for left or right container."""
    base = {
        "border": "1px solid #ddd",
        "borderRadius": "5px",
        "padding": "10px",
        "width": "70%" if side == "left" else "27%",
        "float": side,
    }
    return base


def _default_graph() -> html.Div:
    """Return placeholder graph div when no file is selected."""
    return html.Div(
        [
            html.P(
                "Select an NDJSON graph file to visualize",
                style={"padding": "20px", "textAlign": "center"},
            ),
            cyto.Cytoscape(
                id="cytoscape-graph",
                layout={"name": "preset"},
                style={"width": "100%", "height": "300px", "display": "none"},
                elements=[
                    {
                        "data": {"id": "one", "label": "Node 1"},
                        "position": {"x": 75, "y": 75},
                    },
                    {
                        "data": {"id": "two", "label": "Node 2"},
                        "position": {"x": 200, "y": 200},
                    },
                    {"data": {"source": "one", "target": "two"}},
                ],
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "background-color": "#07ABA0",
                        },
                    },
                    {
                        "selector": "edge",
                        "style": {
                            "curve-style": "bezier",
                            "target-arrow-shape": "triangle",
                        },
                    },
                ],
            ),
        ]
    )


def _edge_style() -> dict:
    return {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#ccc",
            "target-arrow-color": "#ccc",
            "width": 1,
        },
    }


def _nx_to_cyto(G: nx.Graph) -> list:
    """Convert NetworkX graph to Cytoscape element list with styled nodes."""
    # Compute normalization values, handling None values
    as_values = []
    for _, d in G.nodes(data=True):
        val = d.get("node_as", 0)
        # Handle None values by converting to 0
        if val is None:
            as_values.append(0.0)
        else:
            try:
                as_values.append(float(val))
            except (TypeError, ValueError):
                as_values.append(0.0)

    elements = []
    

    
    # Get threshold from graph attributes, default to inf if not present
    threshold = G.graph.get("threshold", float("inf"))
    
    colors = get_plotly_color(pc.diverging.balance, as_values, threshold)
    
    for i, (node, data) in enumerate(G.nodes(data=True)):
        
        data["color"] = colors[i] if i < len(colors) else "#808080"

        elements.append(
            {"data": {**{k: str(v) for k, v in data.items()}, "id": str(node)}}
        )
        
        
       

    for u, v, data in G.edges(data=True):
        edge = {
            "data": {
                **{k: str(v) for k, v in data.items()},
                "source": str(u),
                "target": str(v),
            }
        }
        elements.append(edge)

    return elements


def create_graph_vis_app(root="test_data/AfterImageGraph/graphs"):

    # Initialize Dash app
    app = Dash(__name__, suppress_callback_exceptions=True)
    server = app.server  # for deployment

    # Build hierarchy once
    file_hierarchy = build_file_hierarchy(root)

    # Store for loaded graphs
    app.layout = html.Div(
        [
            html.H2("Interactive Graph Explorer"),
            # Store for current graphs
            dcc.Store(id="graphs-store"),
            
            # Controls: dropdowns + load button
            html.Div(
                [
                    html.Div(
                        id="dropdown-container",
                        children=[_make_dropdown(0, [], file_hierarchy=file_hierarchy)],
                        style={"marginBottom": "10px"},
                    ),
                    html.Button("Load NDJSON File", id="load-file-btn", n_clicks=0),
                ],
                style={"marginBottom": "20px"},
            ),
            
            # Graph slider controls (initially hidden)
            html.Div(
                id="slider-container",
                children=[
                    html.Div(
                        [
                            html.Label("Graph Index:", style={"marginRight": "10px"}),
                            dcc.Input(
                                id="graph-index-input",
                                type="number",
                                min=0,
                                step=1,
                                style={"width": "80px", "marginRight": "20px"},
                                debounce=True
                            ),
                            html.Span(id="graph-count-label", style={"marginLeft": "10px"}),
                        ],
                        style={"marginBottom": "10px", "display": "flex", "alignItems": "center"},
                    ),
                    dcc.Slider(
                        id="graph-slider",
                        min=0,
                        step=1,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                style={"marginBottom": "20px", "display": "none"},
            ),
            
            # Content
            html.Div(
                [
                    html.Div(id="graph-container", style=_container_style("left")),
                    html.Div(_info_panel(), style=_container_style("right")),
                ],
                style={"overflow": "hidden"},
            ),
        ]
    )

    @app.callback(
        Output("dropdown-container", "children"),
        Input({"type": "dynamic-dropdown", "level": ALL}, "value"),
    )
    def update_dropdowns(selected):
        """Dynamically add or remove dropdowns based on selections."""
        vals = [v for v in selected if v]
        dropdowns = []
        for i, val in enumerate(vals):
            dropdowns.append(
                _make_dropdown(i, vals[:i], value=val, file_hierarchy=file_hierarchy)
            )
        # Determine next options
        node = file_hierarchy
        for val in vals:
            node = node.get(val, {})
            if node is None:
                return html.Div(dropdowns)
        if isinstance(node, dict) and node:
            dropdowns.append(
                _make_dropdown(len(vals), vals, file_hierarchy=file_hierarchy)
            )
        return html.Div(dropdowns)

    @app.callback(
        [Output("graphs-store", "data"),
         Output("slider-container", "style"),
         Output("graph-slider", "max"),
         Output("graph-slider", "value"),
         Output("graph-index-input", "max"),
         Output("graph-index-input", "value"),
         Output("graph-count-label", "children")],
        Input("load-file-btn", "n_clicks"),
        State({"type": "dynamic-dropdown", "level": ALL}, "value"),
    )
    def load_ndjson_file(n_clicks, selected):
        """Load NDJSON file and setup slider controls."""
        if n_clicks is None or n_clicks == 0:
            return None, {"display": "none"}, 0, 0, 0, 0, ""

        vals = [v for v in selected if v]
        if not vals or not vals[-1].endswith(".ndjson"):
            return None, {"display": "none"}, 0, 0, 0, 0, ""

        path = os.path.join(root, *vals)
        
        try:
            graphs = load_ndjson_graphs(path)
            if not graphs:
                return None, {"display": "none"}, 0, 0, 0, 0, ""
            
            # Serialize graphs for storage
            graphs_data = []
            for G in graphs:
                # Convert to cytoscape format for storage
                cyto_data = nx.cytoscape_data(G)
                graphs_data.append(cyto_data)
            
            max_idx = len(graphs) - 1
            return (
                graphs_data,
                {"marginBottom": "20px"},  # Show slider
                max_idx,
                0,  # Start at first graph
                max_idx,
                0,
                f"of {len(graphs)} graphs"
            )
        except Exception as e:
            print(f"Error loading NDJSON: {e}")
            return None, {"display": "none"}, 0, 0, 0, 0, ""

    @app.callback(
        Output("graph-slider", "value", allow_duplicate=True),
        Input("graph-index-input", "value"),
        prevent_initial_call=True
    )
    def sync_input_to_slider(input_value):
        """Sync input field value to slider."""
        if input_value is not None:
            return input_value
        return 0

    @app.callback(
        Output("graph-index-input", "value", allow_duplicate=True),
        Input("graph-slider", "value"),
        prevent_initial_call=True
    )
    def sync_slider_to_input(slider_value):
        """Sync slider value to input field."""
        if slider_value is not None:
            return slider_value
        return 0

    @app.callback(
        Output("graph-container", "children"),
        [Input("graph-slider", "value"),
         Input("graphs-store", "data")],
    )
    def display_graph(slider_value, graphs_data):
        """Display the selected graph from the NDJSON file."""
        if not graphs_data or slider_value is None:
            return _default_graph()
        
    
        # Get the selected graph
        cyto_data = graphs_data[slider_value]
        G = nx.cytoscape_graph(cyto_data)
    
        
        elements = _nx_to_cyto(G)
        
        # Get threshold for display
        threshold = G.graph.get('threshold', 'N/A')
        
        stylesheet = [
            {
                "selector": "node",
                "style": {
                    "background-color": "data(color)",
                    "label": "data(name)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "12px",
                },
            },
            _edge_style(),
        ]
        
        return html.Div(
            [
                html.H3(
                    f"Graph {slider_value + 1} - Threshold: {threshold}", 
                    style={"marginBottom": "15px"}
                ),
                cyto.Cytoscape(
                    id="cytoscape-graph",
                    layout={"name": "cose"},
                    style={"width": "100%", "height": "500px"},
                    elements=elements,
                    stylesheet=stylesheet,
                ),
            ]
        )


    @app.callback(
        Output("element-info", "children"),
        Input("cytoscape-graph", "tapNodeData"),
        Input("cytoscape-graph", "tapEdgeData"),
    )
    def display_element_info(node_data, edge_data):
        """Show attributes of clicked node or edge in a table."""
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        if node_data and trigger == "cytoscape-graph.tapNodeData":
            data = node_data
            title = f"Node {data.get('label', data.get('id', 'Unknown'))}"
        elif edge_data and trigger == "cytoscape-graph.tapEdgeData":
            data = edge_data
            title = f"Edge {data.get('source')} → {data.get('target')}"
        else:
            data = None

        if not data:
            return html.P("Click on a node or edge to see its attributes")

        rows = []
        for k, v in data.items():
            if k not in ["id", "source", "target", "timeStamp", "color"]:
                rows.append(html.Tr([html.Td(k), html.Td(str(v))]))

        return [
            html.H4(title),
            html.Table(
                [html.Tr([html.Th("Attribute"), html.Th("Value")])] + rows,
                style={
                    "width": "100%",
                    "tableLayout": "fixed",
                    "borderCollapse": "collapse",
                },
            ),
        ]

    return app


if __name__ == "__main__":
    app = create_graph_vis_app("../test_data/NetworkAccessGraphExtractor/graphs")
    app.run(debug=True)