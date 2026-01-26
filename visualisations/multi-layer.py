from matplotlib import pyplot as plt
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
import numpy as np
from collections import defaultdict
import networkx as nx
import pickle
from bokeh.plotting import figure
from bokeh.models import (HoverTool, Circle, MultiLine, 
                          Label, ColumnDataSource, PointDrawTool)
from bokeh.palettes import Category10
import panel as pn

# Initialize Panel extension
pn.extension()


def update_connection(G, src, dst, payload_len):
    if G.has_edge(src, dst):
        G[src][dst]["size"] += payload_len
        G[src][dst]["count"] += 1
    else:
        G.add_edge(src, dst, size=payload_len, count=1)

def horizontal_band_layout(G):
    """
    For each layer, compute a force-directed layout on the subgraph containing
    only nodes of that layer, then place the resulting mini-layout onto a fixed
    horizontal band.
    """

    # Group nodes by layer
    groups = {}
    for node, data in G.nodes(data=True):
        key = data.get("layer")
        groups.setdefault(key, []).append(node)

    # Fixed y-positions per layer
    layer_y = {
        'Physical': 0,
        'Internet': 200,
        'Transport': 400
    }

    pos = {}

    for layer, nodes in groups.items():
        # Extract subgraph for this layer
        subG = G.subgraph(nodes)

        # Force-directed positions *within the layer*
        sub_pos = nx.kamada_kawai_layout(subG)   # deterministic layout

        # Normalize/shift x positions so nodes spread out nicely
        xs = [p[0] for p in sub_pos.values()]
        min_x, max_x = min(xs), max(xs)
        span = max(max_x - min_x, 1e-6)
        
        # Normalize/shift y positions so nodes spread out nicely
        ys = [p[1] for p in sub_pos.values()]
        min_y, max_y = min(ys), max(ys)
        span_y = max(max_y - min_y, 1e-6)

        
        for node in nodes:
            x = (sub_pos[node][0] - min_x) / span * 800   # scale to width ~800
            y = (sub_pos[node][1] - min_y) / span_y * 100   # scale to width ~100
            # vertically shift y
            y = y+layer_y[layer]
            
            pos[node] = (x, y)
    return pos

class MultilayerNetworkGraph:
    def __init__(self, pcap_file, max_packets=None):
        self.pcap_file = pcap_file
        self.max_packets = max_packets        
        
        # NetworkX graphs for each layer
        self.graph=nx.DiGraph()
    
    def parse_pcap(self):
        """Parse PCAP file and extract network information"""
        packets = scapy.rdpcap(self.pcap_file)
        
        # Limit number of packets if specified
        if self.max_packets:
            packets = packets[:self.max_packets]
            print(f"Processing {len(packets)} packets (limited by max_packets={self.max_packets})")
        else:
            print(f"Processing {len(packets)} packets")
        
        for packet in packets:
            # Extract MAC layer information
            if Ether in packet:
                src_mac = packet[Ether].src
                dst_mac = packet[Ether].dst
                
                # add nodes
                self.graph.add_node(src_mac, layer="Physical")
                self.graph.add_node(dst_mac, layer="Physical")
                
                ether_len=len(packet[Ether].payload)
                
                # Add horizontal connection, update if exists
                update_connection(self.graph, src_mac, dst_mac, ether_len)
            
            # Extract IP layer information
            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                # add nodes
                self.graph.add_node(src_ip, layer="Internet")
                self.graph.add_node(dst_ip, layer="Internet")
                
                ip_len=len(packet[IP].payload)
                
                # Add horizontal connection, update if exists
                update_connection(self.graph, src_ip, dst_ip, ether_len)
                
                # Add interlayer connection (MAC to IP)
                if Ether in packet:
                    update_connection(self.graph, src_ip,src_mac,ip_len)
                    
                    update_connection(self.graph, dst_mac,dst_ip,ether_len)
                    
                    
                
                # Extract transport layer information
                port_src, port_dst = None, None
                if TCP in packet:
                    port_src = packet[TCP].sport
                    port_dst = packet[TCP].dport
                    transport_len=len(packet[TCP].payload)
                elif UDP in packet:
                    port_src = packet[UDP].sport
                    port_dst = packet[UDP].dport
                    transport_len=len(packet[UDP].payload)
                
                if port_src and port_dst:
                    src_socket = f"{src_ip}:{port_src}"
                    dst_socket = f"{dst_ip}:{port_dst}"
                    
                    # add nodes
                    self.graph.add_node(src_socket, layer="Transport")
                    self.graph.add_node(dst_socket, layer="Transport")
                    
                    update_connection(self.graph, src_socket,dst_socket,transport_len)
                    
                    update_connection(self.graph, src_socket,src_ip,transport_len)
                    update_connection(self.graph, dst_ip,dst_socket,ip_len)
        

    def get_stats(self):
        """Get network statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'physical_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('layer') == 'Physical']),
            'internet_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('layer') == 'Internet']),
            'transport_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('layer') == 'Transport']),
        }
        return stats

    def visualize(self):
        """Create visualization of multilayer network using Bokeh"""
        
        # Layer colors
        layer_colors = {'Physical': '#FF8C42', 'Internet': '#4ECDC4', 'Transport': '#95E06C'}
        
        # Calculate node positions for each layer
        pos = horizontal_band_layout(self.graph)
        # pos=nx.multipartite_layout(self.graph, subset_key="layer")
        
        # Prepare node data
        node_x, node_y, node_colors, node_labels, node_layers = [], [], [], [], []
        node_indices = {}
        
        for idx, node in enumerate(self.graph.nodes()):
            layer = self.graph.nodes[node].get("layer", "Unknown")
            x, y = pos[node]
            
            node_x.append(x * 500)
            node_y.append(y * 500)
            node_colors.append(layer_colors.get(layer, '#CCCCCC'))
            node_labels.append(str(node))
            node_layers.append(layer)
            node_indices[node] = idx
        
        # Prepare edge data
        edge_xs, edge_ys, edge_colors, edge_widths, edge_dashes = [], [], [], [], []
        edge_sources, edge_targets = [], []
        
        for u, v in self.graph.edges():
            lu = self.graph.nodes[u].get("layer", None)
            lv = self.graph.nodes[v].get("layer", None)
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            edge_xs.append([x0 * 500, x1 * 500])
            edge_ys.append([y0 * 500, y1 * 500])
            edge_sources.append(node_indices[u])
            edge_targets.append(node_indices[v])
            
            if lu == lv:
                edge_colors.append('#666666')
                edge_dashes.append('solid')
            else:
                edge_colors.append('#999999')
                edge_dashes.append('dashed')
            edge_widths.append(2)
        
        # Create figure
        plot = figure(
            title="Multilayer Network Visualization (Drag nodes to move)",
            width=1000,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above"
        )
        
        # Create data sources
        edge_source = ColumnDataSource(data=dict(
            xs=edge_xs, ys=edge_ys, colors=edge_colors,
            widths=edge_widths, line_dash=edge_dashes,
            sources=edge_sources, targets=edge_targets
        ))
        
        node_source = ColumnDataSource(data=dict(
            x=node_x,
            y=node_y,
            colors=node_colors,
            labels=node_labels,
            layers=node_layers
        ))
        
        # Draw edges first (so they're behind nodes)
        plot.multi_line('xs', 'ys', source=edge_source,
                    line_width='widths', color='colors',
                    line_dash='line_dash', alpha=0.6)
        
        # Draw nodes - MUST use Circle renderer for PointDrawTool
        node_renderer = plot.circle('x', 'y', size=20, source=node_source,
                        fill_color='colors', line_color='black',
                        line_width=2, alpha=0.9)
        
        # Add labels to nodes
        labels = plot.text('x', 'y', text='labels', source=node_source,
                          text_align='center', text_baseline='top',
                          y_offset=-25, text_font_size='10pt')
        
        # Add hover tool
        hover = HoverTool(renderers=[node_renderer], tooltips=[
            ("Node", "@labels"),
            ("Layer", "@layers")
        ])
        plot.add_tools(hover)
        
        # Add PointDrawTool for dragging nodes
        draw_tool = PointDrawTool(renderers=[node_renderer], add=False)
        plot.add_tools(draw_tool)
        plot.toolbar.active_tap = draw_tool
        
        # Style the plot
        plot.axis.visible = False
        plot.grid.visible = False
        plot.outline_line_color = None        

        # JavaScript callback to update edges when nodes move
        from bokeh.models import CustomJS
        
        callback = CustomJS(args=dict(node_source=node_source, edge_source=edge_source), code="""
            const node_data = node_source.data;
            const edge_data = edge_source.data;
            
            // Update edge positions based on new node positions
            for (let i = 0; i < edge_data.sources.length; i++) {
                const src_idx = edge_data.sources[i];
                const tgt_idx = edge_data.targets[i];
                
                edge_data.xs[i] = [node_data.x[src_idx], node_data.x[tgt_idx]];
                edge_data.ys[i] = [node_data.y[src_idx], node_data.y[tgt_idx]];
            }
            
            edge_source.change.emit();
        """)
        
        # Attach callback to node source changes
        node_source.js_on_change('data', callback)
        
        return plot


def create_panel_app():
    """Create Panel dashboard application"""
    
    # Widgets for user input
    pcap_input = pn.widgets.TextInput(
        name='PCAP File Path', 
        value='../datasets/test_data/pcap/benign_lenovo_bulb.pcap',
        placeholder='Enter path to PCAP file...'
    )
    max_packets_input = pn.widgets.IntInput(
        name='Max Packets (0 for all)', 
        value=10, 
        start=0, 
        step=10
    )
    load_button = pn.widgets.Button(name='Load Network', button_type='primary')
    status_text = pn.pane.Markdown("Click 'Load Network' to visualize", sizing_mode='stretch_width')
    
    # Placeholder for plot
    plot_pane = pn.pane.Bokeh(figure(width=800, height=600, title="No data loaded"))
    
    def load_network(event):
        """Load and visualize network when button is clicked"""
        pcap_file = pcap_input.value
        max_packets = max_packets_input.value if max_packets_input.value > 0 else None
        
        try:
            status_text.object = f"Loading PCAP file: {pcap_file}..."
            print(f"Loading PCAP file: {pcap_file}")
            
            # Create and parse graph
            graph = MultilayerNetworkGraph(pcap_file, max_packets=max_packets)
            graph.parse_pcap()
            
            # Get statistics
            stats = graph.get_stats()
            print(f"Graph created with {stats['total_nodes']} nodes and {stats['total_edges']} edges")
            
            # Check if graph is empty
            if stats['total_nodes'] == 0:
                status_text.object = "⚠️ **Warning:** No nodes found in graph. Check PCAP file contents."
                return
            
            # Create visualization
            plot = graph.visualize()
            plot_pane.object = plot
            
            # Update status with statistics
            status_text.object = f"""
            ✅ **Network Loaded Successfully!**
            
            - Total Nodes: {stats['total_nodes']}
            - Total Edges: {stats['total_edges']}
            - Physical Layer: {stats['physical_nodes']} nodes
            - Internet Layer: {stats['internet_nodes']} nodes
            - Transport Layer: {stats['transport_nodes']} nodes
            """
            
        except FileNotFoundError:
            error_msg = f"❌ **Error:** File not found: {pcap_file}"
            print(error_msg)
            status_text.object = error_msg
            
        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            print(error_msg)
            status_text.object = error_msg
    
    # Connect button to callback
    load_button.on_click(load_network)
    
    # Load network on startup
    load_network(None)
    
    # Create info panel
    info_text = """
    ## Multilayer Network Analyzer
    
    ### Instructions
    1. Enter PCAP file path above
    2. Set max packets (0 for all)
    3. Click **Load Network**
    4. Click the **tap tool** (finger icon) in toolbar
    5. **Drag nodes** to reposition them
    
    ### Layer Colors
    - 🟠 **Physical Layer** (MAC addresses)
    - 🟢 **Internet Layer** (IP addresses)  
    - 🟣 **Transport Layer** (Sockets)
    
    ### Edge Types
    - **Solid lines:** Same-layer connections
    - **Dashed lines:** Cross-layer connections
    """
    
    # Create Panel components
    sidebar = pn.Column(
        pn.pane.Markdown("# Network Visualizer", sizing_mode='stretch_width'),
        pcap_input,
        max_packets_input,
        load_button,
        pn.layout.Divider(),
        status_text,
        pn.layout.Divider(),
        pn.pane.Markdown(info_text, sizing_mode='stretch_width'),
        width=350,
        sizing_mode='stretch_height'
    )
    
    main_panel = pn.Column(
        plot_pane,
        sizing_mode='stretch_both'
    )
    
    # Create template
    template = pn.template.MaterialTemplate(
        site="Network Analysis",
        title="Multilayer Network Visualization",
        sidebar=[sidebar],
        main=[main_panel],
        header_background='#2196F3',
    )
    template.servable()
    return template


# Create and serve the app
if __name__.startswith("bokeh") or __name__ == "__main__":
    create_panel_app()
    