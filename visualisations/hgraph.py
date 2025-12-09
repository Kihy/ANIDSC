#!/usr/bin/env python3
"""
PCAP Hierarchical Graph Visualizer with Communication Flows
Parses PCAP files and creates a hierarchical graph: MAC -> IP -> Port
Including peer-to-peer communication at the lowest available layer
"""

from scapy.all import rdpcap, Ether, IP, TCP, UDP
import networkx as nx
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import (Circle, MultiLine, HoverTool, BoxZoomTool, ResetTool, 
                          PanTool, WheelZoomTool, TapTool, SaveTool, 
                          NodesAndLinkedEdges, GraphRenderer, StaticLayoutProvider)
from bokeh.palettes import Spectral4
from bokeh.layouts import column
from bokeh.models import Div, ColumnDataSource
from collections import defaultdict
import sys

class NetworkHierarchy:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.stats = {
            'macs': defaultdict(lambda: {'packets': 0, 'bytes': 0}),
            'ips': defaultdict(lambda: {'packets': 0, 'bytes': 0}),
            'ports': defaultdict(lambda: {'packets': 0, 'bytes': 0, 'protocol': None})
        }
        
    def process_pcap(self, pcap_file, max_packets=None):
        """Parse PCAP file and build hierarchical graph with communication flows"""
        print(f"Reading PCAP file: {pcap_file}")
        
        try:
            packets = rdpcap(pcap_file)
        except Exception as e:
            print(f"Error reading PCAP: {e}")
            return
        
        total_packets = len(packets)
        if max_packets:
            packets = packets[:max_packets]
            
        print(f"Processing {len(packets)} packets (out of {total_packets} total)")
        
        for pkt in packets:
            if not pkt.haslayer(Ether):
                continue
                
            # Extract MAC addresses
            src_mac = pkt[Ether].src
            dst_mac = pkt[Ether].dst
            pkt_len = len(pkt)
            
            # Add MAC nodes
            self._add_mac_node(src_mac, pkt_len)
            self._add_mac_node(dst_mac, pkt_len)
            
            # Add MAC-to-MAC communication (will be filtered later if IP layer exists)
            self._add_edge(src_mac, dst_mac, pkt_len, 'mac_communication')
            
            # Process IP layer
            if pkt.haslayer(IP):
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                
                # Add IP nodes
                self._add_ip_node(src_ip, pkt_len)
                self._add_ip_node(dst_ip, pkt_len)
                
                # Add hierarchy edges (MAC -> IP)
                self._add_edge(src_mac, src_ip, pkt_len, 'mac_to_ip')
                self._add_edge(dst_mac, dst_ip, pkt_len, 'mac_to_ip')
                
                # Add IP-to-IP communication (will be filtered later if transport layer exists)
                self._add_edge(src_ip, dst_ip, pkt_len, 'ip_communication')
                
                # Process Transport layer (TCP/UDP)
                if pkt.haslayer(TCP):
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                    protocol = 'TCP'
                    self._process_ports(src_ip, dst_ip, src_port, dst_port, protocol, pkt_len)
                    
                elif pkt.haslayer(UDP):
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport
                    protocol = 'UDP'
                    self._process_ports(src_ip, dst_ip, src_port, dst_port, protocol, pkt_len)
        
        print(f"\nGraph Statistics:")
        print(f"  Total nodes: {self.graph.number_of_nodes()}")
        print(f"  Total edges: {self.graph.number_of_edges()}")
        
        # Count edges by type
        edge_types = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            edge_types[data['type']] += 1
        
        print(f"\nEdge Types (before filtering):")
        print(f"  MAC-to-MAC communications: {edge_types['mac_communication']}")
        print(f"  MAC-to-IP mappings: {edge_types['mac_to_ip']}")
        print(f"  IP-to-IP communications: {edge_types['ip_communication']}")
        print(f"  IP-to-Port mappings: {edge_types['ip_to_port']}")
        print(f"  Port-to-Port communications: {edge_types['port_communication']}")
        
        print(f"\nNode Counts:")
        print(f"  MAC addresses: {len([n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'physical'])}")
        print(f"  IP addresses: {len([n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'internet'])}")
        print(f"  Ports: {len([n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'transport'])}")
        
        # Filter communication edges to show only at lowest layer
        self._filter_communication_edges()
        
    def _add_mac_node(self, mac, pkt_len):
        """Add MAC address node"""
        if not self.graph.has_node(mac):
            self.graph.add_node(mac, layer='physical', type='MAC', label=f"{mac[:8]}...")
        self.stats['macs'][mac]['packets'] += 1
        self.stats['macs'][mac]['bytes'] += pkt_len
        
    def _add_ip_node(self, ip, pkt_len):
        """Add IP address node"""
        if not self.graph.has_node(ip):
            self.graph.add_node(ip, layer='internet', type='IP', label=ip)
        self.stats['ips'][ip]['packets'] += 1
        self.stats['ips'][ip]['bytes'] += pkt_len
        
    def _add_port_node(self, ip, port, protocol, pkt_len):
        """Add port node"""
        node_id = f"{ip}:{port}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, layer='transport', type='Port', 
                              ip=ip, port=port, protocol=protocol,
                              label=f"{port}\n({protocol})")
        self.stats['ports'][node_id]['packets'] += 1
        self.stats['ports'][node_id]['bytes'] += pkt_len
        self.stats['ports'][node_id]['protocol'] = protocol
        
    def _add_edge(self, src, dst, pkt_len, edge_type):
        """Add or update edge between nodes"""
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]['packets'] += 1
            self.graph[src][dst]['bytes'] += pkt_len
        else:
            self.graph.add_edge(src, dst, packets=1, bytes=pkt_len, type=edge_type)
            
    def _process_ports(self, src_ip, dst_ip, src_port, dst_port, protocol, pkt_len):
        """Process transport layer ports"""
        src_endpoint = f"{src_ip}:{src_port}"
        dst_endpoint = f"{dst_ip}:{dst_port}"
        
        # Add port nodes
        self._add_port_node(src_ip, src_port, protocol, pkt_len)
        self._add_port_node(dst_ip, dst_port, protocol, pkt_len)
        
        # Add hierarchy edges (IP -> Port)
        self._add_edge(src_ip, src_endpoint, pkt_len, 'ip_to_port')
        self._add_edge(dst_ip, dst_endpoint, pkt_len, 'ip_to_port')
        
        # Add port-to-port communication (Transport layer)
        self._add_edge(src_endpoint, dst_endpoint, pkt_len, 'port_communication')
    
    def _filter_communication_edges(self):
        """
        Filter communication edges to only show them at the lowest available layer.
        - If a node has transport layer, only show port-to-port communication
        - If a node has IP layer but no transport, show IP-to-IP communication
        - If a node has only MAC layer, show MAC-to-MAC communication
        
        IMPORTANT: Only filter MAC-to-MAC edges between MACs that BOTH have IP layers.
        If either MAC has no IP layer (non-IP protocols), keep the MAC-to-MAC edge.
        """
        edges_to_remove = []
        
        # Build a set of MACs that have IP layers
        macs_with_ip = set()
        for mac in [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'physical']:
            has_ip_layer = any(self.graph.nodes[neighbor]['layer'] == 'internet' 
                              for neighbor in self.graph.neighbors(mac) 
                              if neighbor in self.graph.nodes() and 
                              self.graph.nodes[neighbor]['layer'] == 'internet')
            if has_ip_layer:
                macs_with_ip.add(mac)
        
        # Remove MAC-to-MAC edges only if BOTH MACs have IP layers
        for mac in [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'physical']:
            if mac in macs_with_ip:
                for neighbor in list(self.graph.neighbors(mac)):
                    if (neighbor in self.graph.nodes() and 
                        self.graph.nodes[neighbor]['layer'] == 'physical' and
                        neighbor in macs_with_ip):  # Both have IP layers
                        edges_to_remove.append((mac, neighbor))
        
        # Build a set of IPs that have port layers
        ips_with_ports = set()
        for ip in [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'internet']:
            has_port_layer = any(self.graph.nodes[neighbor]['layer'] == 'transport' 
                                for neighbor in self.graph.neighbors(ip) 
                                if neighbor in self.graph.nodes() and 
                                self.graph.nodes[neighbor]['layer'] == 'transport')
            if has_port_layer:
                ips_with_ports.add(ip)
        
        # Remove IP-to-IP edges only if BOTH IPs have port layers
        for ip in [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'internet']:
            if ip in ips_with_ports:
                for neighbor in list(self.graph.neighbors(ip)):
                    if (neighbor in self.graph.nodes() and 
                        self.graph.nodes[neighbor]['layer'] == 'internet' and
                        neighbor in ips_with_ports):  # Both have port layers
                        edges_to_remove.append((ip, neighbor))
        
        # Remove filtered edges
        self.graph.remove_edges_from(edges_to_remove)
        
        print(f"\nFiltered {len(edges_to_remove)} communication edges (showing only at lowest layer)")
        print(f"Kept MAC-to-MAC edges for non-IP protocols and IP-to-IP for non-transport protocols")
        
        # Remove isolated nodes (nodes with no edges at all)
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            print(f"\nRemoving {len(isolated_nodes)} isolated nodes with no connections:")
            for node in isolated_nodes[:10]:  # Show first 10
                layer = self.graph.nodes[node]['layer']
                print(f"  - {node} ({layer})")
            if len(isolated_nodes) > 10:
                print(f"  ... and {len(isolated_nodes) - 10} more")
            self.graph.remove_nodes_from(isolated_nodes)
    
    def visualize(self, output_filename='network_hierarchy.html', layout='hierarchical'):
        """Visualize the hierarchical graph with interactive Bokeh"""
        if self.graph.number_of_nodes() == 0:
            print("No data to visualize")
            return
        
        output_file(output_filename)
        
        # Separate nodes by layer
        physical_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'physical']
        internet_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'internet']
        transport_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'transport']
        
        # Create hierarchical layout
        pos = {}
        layer_spacing = 4.0
        
        # Position physical layer (top)
        for i, node in enumerate(physical_nodes):
            pos[node] = (i * 3.0, layer_spacing * 2)
            
        # Position internet layer (middle)
        for i, node in enumerate(internet_nodes):
            pos[node] = (i * 3.0, layer_spacing)
            
        # Position transport layer (bottom)
        for i, node in enumerate(transport_nodes):
            pos[node] = (i * 3.0, 0)
        
        # Set positions as node attributes for NetworkX
        for node, position in pos.items():
            self.graph.nodes[node]['pos'] = position
        
        # Create Bokeh plot
        plot = figure(title="Network Traffic Hierarchical Graph (Interactive)",
                     x_range=(-5, max(len(physical_nodes), len(internet_nodes), len(transport_nodes)) * 3 + 5),
                     y_range=(-2, layer_spacing * 2 + 2),
                     width=1400, height=900,
                     tools=[PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool(), TapTool()],
                     toolbar_location="above")
        
        plot.title.text_font_size = "16pt"
        plot.title.align = "center"
        
        # Add node attributes for visualization
        node_colors = []
        node_sizes = []
        node_labels = []
        node_stats = []
        
        for node in self.graph.nodes():
            layer = self.graph.nodes[node]['layer']
            
            # Set colors by layer
            if layer == 'physical':
                node_colors.append('#9b59b6')
                stats = self.stats['macs'][node]
            elif layer == 'internet':
                node_colors.append('#3498db')
                stats = self.stats['ips'][node]
            else:  # transport
                node_colors.append('#2ecc71')
                stats = self.stats['ports'][node]
            
            # Set size based on traffic volume (scale down for radius)
            size = min(0.3 + (stats['packets'] / 100), 50)
            node_sizes.append(size)
            
            # Create label and hover info
            label = self.graph.nodes[node].get('label', str(node))
            node_labels.append(label)
            
            hover_info = f"{node} | {stats['packets']} pkts | {stats['bytes']/1024:.1f} KB"
            node_stats.append(hover_info)
        
        # Add attributes to graph
        nx.set_node_attributes(self.graph, dict(zip(self.graph.nodes(), node_colors)), 'node_color')
        nx.set_node_attributes(self.graph, dict(zip(self.graph.nodes(), node_sizes)), 'node_size')
        nx.set_node_attributes(self.graph, dict(zip(self.graph.nodes(), node_labels)), 'node_label')
        nx.set_node_attributes(self.graph, dict(zip(self.graph.nodes(), node_stats)), 'node_stats')
        
        # Add edge attributes for visualization
        edge_colors = []
        edge_widths = []
        edge_alphas = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_type = data['type']
            
            if edge_type in ['mac_to_ip', 'ip_to_port']:
                # Hierarchy edges
                edge_colors.append('#95a5a6')
                edge_widths.append(1)
                edge_alphas.append(0.3)
            elif edge_type == 'mac_communication':
                # MAC communication
                edge_colors.append('#9b59b6')
                edge_widths.append(3)
                edge_alphas.append(0.6)
            elif edge_type == 'ip_communication':
                # IP communication
                edge_colors.append('#3498db')
                edge_widths.append(3)
                edge_alphas.append(0.6)
            elif edge_type == 'port_communication':
                # Port communication
                edge_colors.append('#2ecc71')
                edge_widths.append(3)
                edge_alphas.append(0.7)
            else:
                edge_colors.append('#cccccc')
                edge_widths.append(1)
                edge_alphas.append(0.5)
        
        nx.set_edge_attributes(self.graph, dict(zip(self.graph.edges(), edge_colors)), 'edge_color')
        nx.set_edge_attributes(self.graph, dict(zip(self.graph.edges(), edge_widths)), 'edge_width')
        nx.set_edge_attributes(self.graph, dict(zip(self.graph.edges(), edge_alphas)), 'edge_alpha')
        
        # Create graph renderer manually
        graph_renderer = GraphRenderer()
        
        # Prepare node data
        node_indices = list(self.graph.nodes())
        node_index_map = {node: i for i, node in enumerate(node_indices)}
        
        # Create node data source
        node_data = {
            'index': node_indices,
            'x': [pos[node][0] for node in node_indices],
            'y': [pos[node][1] for node in node_indices],
            'node_color': node_colors,
            'node_size': node_sizes,
            'node_label': node_labels,
            'node_stats': node_stats
        }
        node_source = ColumnDataSource(node_data)
        
        # Create edge data source
        edge_start = []
        edge_end = []
        for u, v in self.graph.edges():
            edge_start.append(node_index_map[u])
            edge_end.append(node_index_map[v])
        
        edge_data = {
            'start': edge_start,
            'end': edge_end,
            'edge_color': edge_colors,
            'edge_width': edge_widths,
            'edge_alpha': edge_alphas
        }
        edge_source = ColumnDataSource(edge_data)
        
        # Set up layout provider
        graph_layout = {node: pos[node] for node in node_indices}
        layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        
        # Configure graph renderer
        graph_renderer.node_renderer.data_source = node_source
        graph_renderer.edge_renderer.data_source = edge_source
        graph_renderer.layout_provider = layout_provider
        
        # Configure node appearance
        graph_renderer.node_renderer.glyph = Circle(radius='node_size', fill_color='node_color', 
                                                     line_color='black', line_width=2)
        graph_renderer.node_renderer.hover_glyph = Circle(radius='node_size', fill_color='node_color', 
                                                           line_color='yellow', line_width=4)
        graph_renderer.node_renderer.selection_glyph = Circle(radius='node_size', fill_color='node_color', 
                                                               line_color='red', line_width=4)
        
        # Configure edge appearance
        graph_renderer.edge_renderer.glyph = MultiLine(line_color='edge_color', 
                                                        line_width='edge_width',
                                                        line_alpha='edge_alpha')
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color='yellow', line_width=5, line_alpha=0.8)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='red', line_width=5, line_alpha=0.9)
        
        # Enable hover and selection policies
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesAndLinkedEdges()
        
        # Add graph to plot
        plot.renderers.append(graph_renderer)
        
        # Add hover tool
        node_hover = HoverTool(tooltips=[("Node", "@node_stats")], renderers=[graph_renderer.node_renderer])
        plot.add_tools(node_hover)
        
        # Add layer labels
        from bokeh.models import Label
        plot.add_layout(Label(x=-3, y=layer_spacing * 2, text="Physical Layer (MAC)", 
                             text_font_size="12pt", text_font_style="bold", text_color="#9b59b6"))
        plot.add_layout(Label(x=-3, y=layer_spacing, text="Internet Layer (IP)", 
                             text_font_size="12pt", text_font_style="bold", text_color="#3498db"))
        plot.add_layout(Label(x=-3, y=0, text="Transport Layer (Port)", 
                             text_font_size="12pt", text_font_style="bold", text_color="#2ecc71"))
        
        # Create legend
        legend_html = """
        <div style="padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px;">
            <h3>Legend</h3>
            <p><span style="color: #9b59b6;">●</span> <strong>Physical Layer (MAC)</strong></p>
            <p><span style="color: #3498db;">●</span> <strong>Internet Layer (IP)</strong></p>
            <p><span style="color: #2ecc71;">●</span> <strong>Transport Layer (Port)</strong></p>
            <hr>
            <p><span style="color: #95a5a6;">━━━</span> Hierarchy (MAC→IP→Port) [dashed]</p>
            <p><span style="color: #9b59b6;">━━━</span> MAC Communication (non-IP)</p>
            <p><span style="color: #3498db;">━━━</span> IP Communication (no ports)</p>
            <p><span style="color: #2ecc71;">━━━</span> Port Communication</p>
            <hr>
            <p><strong>Interactions:</strong></p>
            <ul>
                <li>Hover over nodes/edges for details</li>
                <li>Click to select nodes</li>
                <li>Pan and zoom to explore</li>
            </ul>
        </div>
        """
        
        legend_div = Div(text=legend_html, width=300)
        
        # Create layout
        layout = column(plot, legend_div)
        
        print(f"\nSaving interactive visualization to: {output_filename}")
        save(layout)
        print(f"Interactive visualization saved successfully!")
        print(f"Open {output_filename} in your web browser to explore the graph")
        
        return layout
    
    def print_hierarchy(self, max_items=5):
        """Print hierarchical structure with communications at lowest layer"""
        print("\n" + "="*80)
        print("NETWORK HIERARCHY (Communications at Lowest Available Layer)")
        print("="*80)
        
        physical_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'physical']
        
        for mac in physical_nodes[:max_items]:
            mac_stats = self.stats['macs'][mac]
            print(f"\n📟 MAC: {mac}")
            print(f"   └─ {mac_stats['packets']} packets, {mac_stats['bytes']/1024:.1f} KB")
            
            # Check if MAC has IP layer
            ip_neighbors = [n for n in self.graph.neighbors(mac) 
                          if self.graph.nodes[n]['layer'] == 'internet']
            
            if not ip_neighbors:
                # No IP layer - show MAC communications
                mac_peers = [n for n in self.graph.neighbors(mac) 
                            if self.graph.nodes[n]['layer'] == 'physical']
                if mac_peers:
                    print(f"   │  Communicates with {len(mac_peers)} other MAC(s):")
                    for peer in mac_peers[:3]:
                        edge_data = self.graph[mac][peer]
                        print(f"   │  ├─ ↔ {peer} ({edge_data['packets']} pkts, {edge_data['bytes']/1024:.1f} KB)")
            
            for ip in ip_neighbors[:max_items]:
                ip_stats = self.stats['ips'][ip]
                print(f"   │")
                print(f"   ├─ 🌐 IP: {ip}")
                print(f"   │  └─ {ip_stats['packets']} packets, {ip_stats['bytes']/1024:.1f} KB")
                
                # Check if IP has transport layer
                port_neighbors = [n for n in self.graph.neighbors(ip) 
                                if self.graph.nodes[n]['layer'] == 'transport']
                
                if not port_neighbors:
                    # No transport layer - show IP communications
                    ip_peers = [n for n in self.graph.neighbors(ip) 
                               if self.graph.nodes[n]['layer'] == 'internet']
                    if ip_peers:
                        print(f"   │     Communicates with {len(ip_peers)} other IP(s):")
                        for peer in ip_peers[:3]:
                            edge_data = self.graph[ip][peer]
                            print(f"   │     ├─ ↔ {peer} ({edge_data['packets']} pkts, {edge_data['bytes']/1024:.1f} KB)")
                
                for port in port_neighbors[:max_items]:
                    port_stats = self.stats['ports'][port]
                    protocol = port_stats['protocol']
                    print(f"   │  │")
                    print(f"   │  ├─ 🔌 Port: {port} ({protocol})")
                    print(f"   │  │  └─ {port_stats['packets']} packets, {port_stats['bytes']/1024:.1f} KB")
                    
                    # Show port communications (at transport layer)
                    port_peers = [n for n in self.graph.neighbors(port) 
                                 if self.graph.nodes[n]['layer'] == 'transport']
                    if port_peers:
                        print(f"   │  │     Communicates with {len(port_peers)} other port(s):")
                        for peer in port_peers[:2]:
                            edge_data = self.graph[port][peer]
                            print(f"   │  │     ├─ ↔ {peer} ({edge_data['packets']} pkts)")
                
                if len(port_neighbors) > max_items:
                    print(f"   │  └─ ... and {len(port_neighbors) - max_items} more ports")
            
            if len(ip_neighbors) > max_items:
                print(f"   └─ ... and {len(ip_neighbors) - max_items} more IPs")
        
        if len(physical_nodes) > max_items:
            print(f"\n... and {len(physical_nodes) - max_items} more MAC addresses")
        
        print("\n" + "="*80)
    
    def get_communication_summary(self):
        """Get summary of communication patterns at lowest available layer"""
        print("\n" + "="*80)
        print("COMMUNICATION SUMMARY (Lowest Available Layer)")
        print("="*80)
        
        # Collect traffic at each layer
        mac_traffic = {}
        ip_traffic = {}
        port_traffic = {}
        
        # MAC layer communications
        for mac in [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'physical']:
            outgoing = sum(self.graph[mac][n]['bytes'] for n in self.graph.neighbors(mac) 
                          if self.graph.nodes[n]['layer'] == 'physical')
            if outgoing > 0:
                mac_traffic[mac] = outgoing
        
        # IP layer communications
        for ip in [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'internet']:
            outgoing = sum(self.graph[ip][n]['bytes'] for n in self.graph.neighbors(ip) 
                          if self.graph.nodes[n]['layer'] == 'internet')
            if outgoing > 0:
                ip_traffic[ip] = outgoing
        
        # Port layer communications
        for port in [n for n in self.graph.nodes() if self.graph.nodes[n]['layer'] == 'transport']:
            outgoing = sum(self.graph[port][n]['bytes'] for n in self.graph.neighbors(port) 
                          if self.graph.nodes[n]['layer'] == 'transport')
            if outgoing > 0:
                port_traffic[port] = outgoing
        
        # Display top communicators
        print("\n🏆 Top Communicators:")
        
        if port_traffic:
            top_ports = sorted(port_traffic.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n  Transport Layer (Port-to-Port):")
            for port, bytes_sent in top_ports:
                port_data = self.stats['ports'][port]
                protocol = port_data['protocol']
                print(f"    {port} ({protocol}): {bytes_sent/1024:.1f} KB")
        
        if ip_traffic:
            top_ips = sorted(ip_traffic.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n  Internet Layer (IP-to-IP, no transport below):")
            for ip, bytes_sent in top_ips:
                print(f"    {ip}: {bytes_sent/1024:.1f} KB")
        
        if mac_traffic:
            top_macs = sorted(mac_traffic.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n  Physical Layer (MAC-to-MAC, no IP below):")
            for mac, bytes_sent in top_macs:
                print(f"    {mac}: {bytes_sent/1024:.1f} KB")
        
        if not (port_traffic or ip_traffic or mac_traffic):
            print("\n  No peer communications found")
        
        print("\n" + "="*80)
    
    def export_graph(self, output_file='network_graph.gexf'):
        """Export graph to file format (GEXF for Gephi)"""
        nx.write_gexf(self.graph, output_file)
        print(f"\nGraph exported to: {output_file}")
        print("You can open this file in Gephi or other graph analysis tools")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pcap_hierarchy.py <pcap_file> [max_packets]")
        print("Example: python pcap_hierarchy.py capture.pcap 1000")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    max_packets = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Create hierarchy analyzer
    hierarchy = NetworkHierarchy()
    
    # Process PCAP file
    hierarchy.process_pcap(pcap_file, max_packets)
    
    # Print text hierarchy
    hierarchy.print_hierarchy(max_items=5)
    
    # Print communication summary
    hierarchy.get_communication_summary()
    
    # Visualize graph
    hierarchy.visualize(output_filename='visualisations/network_hierarchy.html')
    


if __name__ == '__main__':
    main()