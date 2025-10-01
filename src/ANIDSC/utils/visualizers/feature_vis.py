import os
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import random 
import plotly.colors as pc


# 5. Define model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def compute_mmd(X, Y, gamma=1.0):
    X = np.asarray(X)
    Y = np.asarray(Y)
    m, n = len(X), len(Y)

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    sum_K_xx = (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1))
    sum_K_yy = (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1))
    sum_K_xy = np.sum(K_xy) / (m * n)

    return sum_K_xx + sum_K_yy - 2 * sum_K_xy

def mmd_permutation_test(X, Y, gamma=1.0, num_permutations=1000, seed=None):
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    m, n = len(X), len(Y)

    observed_mmd = compute_mmd(X, Y, gamma)

    combined = np.vstack([X, Y])
    permuted_mmds = []

    for _ in range(num_permutations):
        indices = rng.permutation(m + n)
        X_perm = combined[indices[:m]]
        Y_perm = combined[indices[m:]]
        mmd_perm = compute_mmd(X_perm, Y_perm, gamma)
        permuted_mmds.append(mmd_perm)

    permuted_mmds = np.array(permuted_mmds)
    p_value = np.mean(permuted_mmds >= observed_mmd)

    return observed_mmd, p_value

def list_csv_files(root):
    return [f for f in os.listdir(root) if f.endswith(".csv")]




def signed_log(x, linthresh=1):
    """Emulate symlog: linear near 0, log farther out."""
    return np.sign(x) * np.log10(1 + np.abs(x / linthresh))


def ks_test(X, Y, p=0.05):
    
    p_values=[]
    for i in range(X.shape[1]):
        _, p=ks_2samp(X[:,i],Y[:,i])
        p_values.append(p)
    
    return np.any(np.array(p_values)<p)

def mmd_test(X, Y, p=0.05):
    observed_mmd, p_value=mmd_permutation_test(X,Y)
    return p_value < p

def nn_test(X,Y):
    
    # 2. Stack and label
    x = np.vstack([X, Y]).astype(np.float32)
    y = np.concatenate([np.zeros(len(X)), np.ones(len(Y))]).astype(np.float32)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    # 4. Convert to PyTorch tensors
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train).unsqueeze(1)
    y_test = torch.tensor(y_test).unsqueeze(1)
    
    model = BinaryClassifier(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # 6. Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 7. Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        predicted_labels = (preds >= 0.5).float()
        f1=f1_score(predicted_labels, y_test)

    
    return f1>0.6

def simple_split(df, time_window=50):

    i=time_window
    
    drift_idx=[]
    while i<len(df):
        prev_chunk=df[i-time_window: i]
        next_chunk=df[i: i+time_window]
        
        color="#"
        
        X = prev_chunk[['count', 'size']].to_numpy()
        Y = next_chunk[['count', 'size']].to_numpy()
        if ks_test(X, Y):
            color+="FF"
        else:
            color+="00"
            
        if nn_test(X, Y):
            color+="FF"
        else:
            color+="00"
        
        color+="00"
        if color != "#000000":
            drift_idx.append((df.iloc[i]["time"], color))
            
        i+=time_window
        
    return drift_idx


def z_score_test(distribution, v):
    # Returns true is there is a drift, false otherwise
    
    mean_distribution = np.mean(distribution,axis=0)
    std_dev_distribution = np.std(distribution,axis=0)
    


    if np.any(std_dev_distribution<1e-6): # if std is small, just check if v is different from mean
        return np.any(np.abs(mean_distribution-v)>1e-6)
            
    
    # Calculate the z-score of the single value
    z = (v - mean_distribution) / std_dev_distribution
    
    p_value=[norm.sf(i) for i in z]
    return np.any(np.array(p_value)<0.05)

def tukey_fence_test(distribution, y):
    q1 = np.percentile(distribution, 25, axis=0)
    q3 = np.percentile(distribution, 75, axis=0)
    
    iqr = q3 - q1
    
    # Calculate Tukey's Fences
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
        
    return np.any(y>upper_fence) or np.any(y<lower_fence)

def mahalanobis_test(distribution, v):
    mean_data = np.mean(distribution, axis=0)
    cov_matrix = np.cov(distribution, rowvar=False) # rowvar=False because columns are features

    # Calculate the inverse of the covariance matrix, use pinv so all inverses are defined

    inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # Calculate the Mahalanobis distance
    # scipy.spatial.distance.mahalanobis expects two 1-D arrays and the inverse covariance matrix
    # Here, we calculate the distance between the point_1d and the mean of the data_2d
    mahal_dist = mahalanobis(v, mean_data, inv_cov_matrix)
    
    squared_mahalanobis = mahal_dist**2
    p_value = 1 - chi2.cdf(squared_mahalanobis, df=distribution.shape[1])
    
    return p_value<0.05

class ReservoirSampler:
    """
    Implements the Reservoir Sampling algorithm to sample k items from a stream.
    """

    def __init__(self, k):
        """
        Initializes the ReservoirSampler.

        Args:
            k (int): The desired size of the sample (the reservoir).
        """

        self.k = k
        self.reservoir = []
        self.count = 0  # Number of items processed so far

    def add_item(self, item):
        """
        Adds an item from the stream to the sampling process.

        Args:
            item: The item to be considered for inclusion in the reservoir.
        """
        self.count += 1

        if len(self.reservoir) < self.k:
            # Fill the reservoir with the first k items
            self.reservoir.append(item)
        else:
            # For subsequent items, decide whether to replace an existing item
            # The probability of replacement is k / self.count
            j = random.randrange(self.count)  # Random index from 0 to count-1
            if j < self.k:
                self.reservoir[j] = item
    
    def add_batch(self, batch):
        
        for i in np.vsplit(batch, batch.shape[0]):
            self.add_item(i)

    def get_sample(self):
        """
        Returns the current sample (the reservoir).

        Returns:
            list: A list containing the k sampled items.
        """
        return self.reservoir
    


def adaptive_split(df, min_time_window=60, point_func=z_score_test, dis_func=ks_test):

    j=min_time_window
    drift_idx=[(df.iloc[0]["time"], "red", 0)]
    
    max_concept_store=360
    concepts={0: ReservoirSampler(max_concept_store)}
    
    #initialize current concept
    concepts[0].add_batch(df[0:min_time_window][['count',"size"]].to_numpy())
    
    concept_count=1
    current_concept_index=0
    while j< len(df):

        current_concept=np.vstack(concepts[current_concept_index].get_sample()).astype(float)
        # there is a potential drift point
        Y = df[j:j+min_time_window][['count', 'size']].to_numpy().astype(float)
        
        
        # check if there is a drift
        if dis_func(current_concept, Y):
            
            possible_concepts=[]
            # check if same concept
            for idx, value in concepts.items():
                
                # same distribution
                if not dis_func(Y, np.vstack(value.get_sample()).astype(float)):
                    possible_concepts.append(idx)
            
            if len(possible_concepts)==1:
                current_concept_index=possible_concepts[0]
                concepts[current_concept_index].add_batch(Y)
            elif len(possible_concepts)==0:
                concepts[concept_count]=ReservoirSampler(max_concept_store)
                concepts[concept_count].add_batch(Y)
                current_concept_index=concept_count
                concept_count+=1
            else:
                # possibly merge these concepts?
                
                current_concept_index=possible_concepts[0]
                concepts[current_concept_index].add_batch(Y)
                    
            drift_idx.append((df.iloc[j]["time"], "red", current_concept_index))            
            j=j+min_time_window

        elif not dis_func(current_concept, Y, p=0.95):
            concepts[current_concept_index].add_batch(Y)                    
            j=j+min_time_window

        
        else:
            concepts[current_concept_index].add_item(df.iloc[j][['count','size']].to_numpy())        
            j+=1
        
    return drift_idx
    
def create_feature_vis_app(
    root="test_data/NetworkAccessExtractor/aggregated/", mac_map={}
):
    # Directory containing your CSV files

    CHUNKSIZE = 100_000  # Adjust based on memory limits

    app = dash.Dash(__name__)
    app.title = "Large CSV Host Visualizer"

    app.layout = html.Div(
        [
            html.H2("Host Traffic Visualizer - Large CSV Support"),
            html.Label("Select CSV File"),
            dcc.Dropdown(
                id="file-dropdown",
                options=[{"label": f, "value": f} for f in list_csv_files(root)],
                placeholder="Select a file...",
            ),
            html.Label("Select Host"),
            dcc.Dropdown(id="host-dropdown", placeholder="Select a host..."),
            html.Button("Toggle Y Scale", id="scale-toggle-btn", n_clicks=0),
            dcc.Store(id="scale-mode", data="linear"),
            html.Div([dcc.Graph(id="count-graph"), dcc.Graph(id="size-graph")]),
        ]
    )

    @app.callback(
        Output("scale-mode", "data"),
        Input("scale-toggle-btn", "n_clicks"),
        State("scale-mode", "data"),
    )
    def toggle_scale(n_clicks, current_mode):
        return "symlog" if current_mode == "linear" else "linear"

    # Callback 1: Extract host list
    @app.callback(Output("host-dropdown", "options"), Input("file-dropdown", "value"))
    def update_hosts(file_name):
        if not file_name:
            return []
        path = os.path.join(root, file_name)
        hosts = set(["All"])
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, usecols=["host"]):
            hosts.update(chunk["host"].unique())

        return [{"label": mac_map.get(h, h), "value": h} for h in sorted(hosts)]

    # Callback 2: Generate two separate plots
    @app.callback(
        Output("count-graph", "figure"),
        Output("size-graph", "figure"),
        Input("host-dropdown", "value"),
        Input("scale-mode", "data"),
        State("file-dropdown", "value"),
    )
    def update_graphs(selected_host, scale_mode, file_name):
        if not file_name or not selected_host:
            return go.Figure(), go.Figure()

        path = os.path.join(root, file_name)
        data = []

        # Filter relevant rows
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE):
            if selected_host == "All":
                filtered = chunk
            else:
                filtered = chunk[chunk["host"] == selected_host]
            if not filtered.empty:
                data.append(filtered)

        if not data:
            return go.Figure(), go.Figure()

        df = pd.concat(data)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        
        # drift_idx=simple_split(df, time_window=100)
        drift_idx=adaptive_split(df, min_time_window=60, point_func=z_score_test, dis_func=ks_test)
        
        
        if scale_mode == 'symlog':
            y_count = signed_log(df['count'])
            y_size = signed_log(df['size'])
            y_label_suffix = " (Signed Log Scale)"
        else:
            y_count = df['count']
            y_size = df['size']
            y_label_suffix = ""

        count_fig = go.Figure(
            [go.Scattergl(x=df["time"], y=y_count, mode="markers", name="Count")]
        )
        count_fig.update_layout(
            title="Count over Time",
            xaxis_title="Time",
            yaxis_title=f"Count{y_label_suffix}",
            template="plotly_white",
            
        )
        
        band_colors = pc.qualitative.Plotly
        
        
        x0, color, prev_concept_idx=drift_idx[0]
        for x1, color, current_concept_index in drift_idx[1:]:
            count_fig.add_vline(x=x1, line_width=2, line_dash="dash", line_color=color)
            
            count_fig.add_shape(
                type='rect',
                xref='x',
                x0=x0, x1=x1,
                yref='paper', y0=0, y1=1,
                fillcolor=band_colors[prev_concept_idx% len(band_colors)],
                opacity=0.2,
                line_width=0,
                layer='below'
            )
            # Compute center of band for annotation
            center = pd.to_datetime(x0) + (pd.to_datetime(x1) - pd.to_datetime(x0)) / 2
            
            # Add index label at the top
            count_fig.add_annotation(
                x=center,
                y=1.02,  # Just above the plot
                xref='x',
                yref='paper',
                text=f"{prev_concept_idx}",
                showarrow=False,
                font=dict(size=12, color='black'),
                align='center'
            )
            
            x0=x1
            prev_concept_idx=current_concept_index
            
            

        size_fig = go.Figure(
            [go.Scattergl(x=df["time"], y=y_size, mode="markers", name="Size")]
        )
        size_fig.update_layout(
            title="Size over Time",
            xaxis_title="Time",
            yaxis_title=f"Size{y_label_suffix}",
            template="plotly_white",
            
        )
        for i in drift_idx:
            size_fig.add_vline(x=i, line_width=2, line_dash="dash", line_color="red")
        
        return count_fig, size_fig

    return app


if __name__ == "__main__":
    app = create_feature_vis_app()

    app.run(debug=True)
