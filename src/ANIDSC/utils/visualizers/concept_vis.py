from collections import defaultdict
import os
import json
import dash
from dash import State, dcc, html, Input, Output
from numpy import copy
import plotly.express as px
import pandas as pd 
import plotly.graph_objects as go
import random




def list_folders(root):
    """Return all subfolders (relative to root)."""
    folders = []
    for dirpath, _, filenames in os.walk(root):
        # Only include folder if it has ndjson files
        if any(f.endswith(".ndjson") for f in filenames):
            rel = os.path.relpath(dirpath, root)
            folders.append((rel, dirpath))
    return folders

def list_files(folder):
    """Return all .ndjson files in a folder (non-recursive)."""
    files = []
    for f in os.listdir(folder):
        if f.endswith(".ndjson"):
            full = os.path.join(folder, f)
            files.append((f, full))
    return files

def load_frames(file_path, prob=1):
    random.seed(42)
    with open(file_path) as f: 
        for line in f:
            if random.random() < prob:
                yield json.loads(line)

# ------------------ DASH APP ------------------
def create_concept_vis_app(root, mac_to_device_map={}, prob=1):
    app = dash.Dash(__name__)

    
    
    app.layout = html.Div([
        html.H2("Node Attribute Plot"),
        html.Div([
            html.Label("Choose file:"),
            
            dcc.Dropdown(
                id="folder-dropdown",
                options=[{"label": rel, "value": path} for rel, path in list_folders(root)],
                placeholder="Select folder"
            ),

            dcc.Dropdown(
                id="file-dropdown",
                placeholder="Select NDJSON file"
            ),

        ], style={"width": "40%", "display": "inline-block"}),

        html.Div([
            html.Label("Choose node id:"),
            dcc.Dropdown(id="node-dropdown", placeholder="Select a node")
        ], style={"width": "40%", "display": "inline-block", "marginLeft": "20px"}),

        dcc.Graph(id="node-plot"),
    ])

    # ------------------ CALLBACKS ------------------
    @app.callback(
        Output("file-dropdown", "options"),
        Input("folder-dropdown", "value")
    )
    def update_file_dropdown(folder):
        if not folder:
            return []
        return [{"label": name, "value": path} for name, path in list_files(folder)]

    @app.callback(
        Output("node-dropdown", "options"),
        Input("file-dropdown", "value")
    )
    def update_node_dropdown(file_path):
        if file_path is None:
            return []
        frames = load_frames(file_path, prob)
        # gather unique node ids across all frames
        node_ids = sorted({n["data"]["id"] for fr in frames for n in fr["elements"]["nodes"]})
        
        return [{"label": mac_to_device_map.get(h, h), "value": h} for h in node_ids]

    @app.callback(
        Output("node-plot", "figure"),
        [Input("file-dropdown", "value"),
         Input("node-dropdown", "value")]
    )
    def plot_node_attr(file_path, node_id):
        if file_path is None or node_id is None:
            empty_fig1 = px.scatter(title="Select a file and node")
            
            return empty_fig1

        
        data_dict=defaultdict(list)
        
        prev_concept=None
        concept_change_indices = []
        concept_idx_order=[]
        unique_concepts=set()        
        for i, fr in enumerate(load_frames(file_path, prob)):
            
            node = next((n for n in fr["elements"]["nodes"] if n["data"]["id"] == node_id), None)
            if node:
                for key, value in fr["data"]:
                    if key=="time_stamp":
                        time=pd.to_datetime(value, unit='s')
                        data_dict["times"].append(time)
                        
                    if key=="threshold":
                        data_dict["threshold"].append(value)
                
                concept_idx=str(node["data"].get("concept_idx"))

                if prev_concept is None:
                    prev_concept=concept_idx
                    concept_idx_order.append(concept_idx)
                    concept_change_indices.append(time)
                    unique_concepts.add(concept_idx)
                
                if prev_concept!=concept_idx:
                    concept_change_indices.append(time)
                    concept_idx_order.append(concept_idx)
                    prev_concept=concept_idx
                    unique_concepts.add(concept_idx)
                
                data_dict["node_as_vals"].append(node["data"].get("node_as"))
                data_dict["count_vals"].append(node["data"].get("count"))
                data_dict["size_vals"].append(node["data"].get("size"))
                
                data_dict["scaled_count_vals"].append(node["data"].get("scaled_count"))
                data_dict["scaled_size_vals"].append(node["data"].get("scaled_size"))
        
        concept_change_indices.append(time)

        # Get unique concepts for color mapping
        colors = px.colors.qualitative.Plotly[:len(unique_concepts)]
        color_map = {concept: colors[i % len(colors)] for i, concept in enumerate(unique_concepts)}
        
        # Plot 1: node_as_vals
        fig1 = go.Figure()
        
        # Add scatter plot for node_as_vals
        fig1.add_trace(go.Scatter(
            x=data_dict['times'],
            y=data_dict['node_as_vals'],
            mode='markers',
            name=f'Node Anomaly Score',
            showlegend=True
        ))
        
        fig1.add_trace(go.Scatter(
            x=data_dict['times'],
            y=data_dict['threshold'],
            mode='markers',
            name=f'Threshold (Graph)',
            showlegend=True
        ))
        

        
        fig1.add_trace(go.Scatter(
            x=data_dict['times'],
            y=data_dict['count_vals'],
            mode='markers',
            name=f'Count',
            marker=dict(color="red"),
            showlegend=True
        ))
        fig1.add_trace(go.Scatter(
            x=data_dict['times'],
            y=data_dict['size_vals'],
            mode='markers',
            name=f'Size',
            marker=dict(color="blue"),
            showlegend=True
        ))

        
        fig1.add_trace(go.Scatter(
            x=data_dict['times'],
            y=data_dict['scaled_count_vals'],
            mode='markers',
            name=f'Scaled Count',
            marker=dict(color="red"),
            showlegend=True
        ))
        fig1.add_trace(go.Scatter(
            x=data_dict['times'],
            y=data_dict['scaled_size_vals'],
            mode='markers',
            name=f'Scaled Size',
            marker=dict(color="blue"),
            showlegend=True
        ))

        # Add background colors for different concept sections
        for i, (prev_idx, cur_idx) in enumerate(zip(concept_change_indices[:-1], concept_change_indices[1:])):
            concept = concept_idx_order[i]
            fig1.add_vrect(
                x0=prev_idx,
                x1=cur_idx,
                fillcolor=color_map[concept],
                annotation_text=f"Concept {concept}",
                opacity=0.1,
                layer="below",
                line_width=0,
            )
            
           
        
        fig1.update_layout(
            title=f"Node {node_id}: node_as values over time",
            xaxis_title="Time",
            showlegend=True
        )
        
        
        return fig1
   
    return app