import os

st.title("Fake News Detection using GCNs & SNA")

# Define base directory for data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
train_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="train")
test_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="test")

datasets = train_data + test_data  # Combine train and test data

# Extract edges and labels
edges = []
node_labels = {}

for data in datasets:
    edge_index = data.edge_index.numpy()
    label = data.y.item()

    for src, tgt in zip(edge_index[0], edge_index[1]):
        edges.append((src, tgt, label))
        node_labels[src] = label
        node_labels[tgt] = label

df_edges = pd.DataFrame(edges, columns=["source", "target", "label"])
df_edges["weight"] = df_edges.groupby(["source", "target"])["source"].transform("count")
df_edges = df_edges.drop_duplicates().reset_index(drop=True)

# Create a directed graph
G = nx.from_pandas_edgelist(df_edges, source="source", target="target", create_using=nx.DiGraph())

# Get node positions
pos = nx.spring_layout(G, k=1)

# Convert positions to Plotly format
node_x, node_y = [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

# Create edge traces
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Create Plotly figure
fig = go.Figure()

# Add edges (grey)
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="grey"), hoverinfo="none", mode="lines"))

# Add nodes
fig.add_trace(go.Scatter(
    x=node_x, y=node_y, mode="markers",
    marker=dict(size=10, showscale=True),
    text=[f"Node {n}" for n in G.nodes()],
    hoverinfo="text"
))

fig.update_layout(title="UPFD Master Graph (Real & Fake News)", showlegend=False, margin=dict(l=0, r=0, t=40, b=0))

st.plotly_chart(fig)

# Button for model execution
st.header("Run Model on the Entire Master Graph")

if st.button("Run Model on Master Graph"):
    with st.spinner("Running model on the entire master graph..."):
        results = predict()
    st.success("Model has finished processing the entire graph!")

    # Display summary
    fake_count = sum(results)
    real_count = len(results) - fake_count
    st.write(f"**Fake News Count:** {fake_count}")
    st.write(f"**Real News Count:** {real_count}")

    # Subgraph visualization
    st.header("Rumor & Non-Rumor Subgraphs")

    rumor_nodes = [node for node, label in node_labels.items() if label == 1]
    non_rumor_nodes = [node for node, label in node_labels.items() if label == 0]

    G_rumor = G.subgraph(rumor_nodes)
    G_non_rumor = G.subgraph(non_rumor_nodes)
    
    def plot_subgraph(G_sub, title, color):
        pos_sub = nx.spring_layout(G_sub, k=1)

        sub_edge_x, sub_edge_y = [], []
        for edge in G_sub.edges():
            x0, y0 = pos_sub[edge[0]]
            x1, y1 = pos_sub[edge[1]]
            sub_edge_x.extend([x0, x1, None])
            sub_edge_y.extend([y0, y1, None])

        sub_node_x, sub_node_y = [], []
        for node in G_sub.nodes():
            x, y = pos_sub[node]
            sub_node_x.append(x)
            sub_node_y.append(y)

        sub_fig = go.Figure()

    # Add edges (always grey)
        sub_fig.add_trace(go.Scatter(
            x=sub_edge_x, y=sub_edge_y,
            line=dict(width=0.5, color="grey"),
            hoverinfo="none",
            mode="lines"
        ))

    # Add nodes
        sub_fig.add_trace(go.Scatter(
            x=sub_node_x, y=sub_node_y,
            mode="markers",
            marker=dict(size=10, color=color, showscale=False),
            text=[f"Node {n}" for n in G_sub.nodes()],
            hoverinfo="text"
        ))

        sub_fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=500,  
        height=400
    )

        return sub_fig


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rumor Subgraph (Fake News)")
        st.plotly_chart(plot_subgraph(G_rumor, "Rumor Subgraph (Fake News)", "red"))

    with col2:
        st.subheader("Non-Rumor Subgraph (Real News)")
        st.plotly_chart(plot_subgraph(G_non_rumor, "Non-Rumor Subgraph (Real News)", "blue"))

st.header("Community Detection for Rumor Subgraph")
if st.button("Run Community Detection on Rumor Subgraph"):
    with st.spinner("Performing community detection on rumor subgraph..."):
        train_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="train")
        test_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="test")
        datasets = train_data + test_data  # Combine train and test data
        edges = []
        node_labels = {}
        
        for data in datasets:
            edge_index = data.edge_index.numpy()
            label = data.y.item()
            for src, tgt in zip(edge_index[0], edge_index[1]):
                edges.append((src, tgt, label))
                node_labels[src] = label
                node_labels[tgt] = label
                
        df_edges = pd.DataFrame(edges, columns=["source", "target", "label"])
        df_edges["weight"] = df_edges.groupby(["source", "target"])["source"].transform("count")
        df_edges = df_edges.drop_duplicates().reset_index(drop=True)
        
        # Create a directed graph
        G = nx.from_pandas_edgelist(df_edges, source="source", target="target", create_using=nx.DiGraph())
        rumor_nodes = [node for node, label in node_labels.items() if label == 1]
        non_rumor_nodes = [node for node, label in node_labels.items() if label == 0]
        G_rumor = G.subgraph(rumor_nodes)
        all_nodes_rumour = list(G_rumor.nodes())
        
        # Step 2: Centrality measures for all nodes
        bc = nx.betweenness_centrality(G_rumor, normalized=True, weight='weight')

        # Convert to undirected graph for eigenvector centrality
        G_undirected = G_rumor.to_undirected()
        ec = nx.eigenvector_centrality_numpy(G_undirected)

        # Step 3: IC simulation and metrics per node
        results = []

        def run_IC(G, seed, p=1.0):
            active_nodes = set([seed])
            newly_active = set([seed])
            all_active_t = [set(active_nodes)]
            while newly_active:
                next_newly_active = set()
                for node in newly_active:
                    for neighbor in G_rumor.successors(node):
                        if neighbor not in active_nodes:
                            if random.random() < p:
                                next_newly_active.add(neighbor)
                active_nodes.update(next_newly_active)
                newly_active = next_newly_active
                all_active_t.append(set(active_nodes))
            return active_nodes, all_active_t

        for node in tqdm(all_nodes_rumour):
            active_nodes_ic, all_active_t_ic = run_IC(G, node)
            final_spread_size = len(active_nodes_ic)
            spread_proportion = final_spread_size / len(G)
            time_to_saturation = len(all_active_t_ic) - 1
            spread_efficiency = final_spread_size / time_to_saturation if time_to_saturation != 0 else 0

            # Subgraph of activated nodes
            subG = G_rumor.subgraph(active_nodes_ic)

            # Average path length (from seed to others in subgraph)
            try:
                lengths = nx.single_source_shortest_path_length(subG, node)
                avg_path_length = np.mean(list(lengths.values()))
            except:
                avg_path_length = 0

            # Avg BC & EC in activated subgraph
            bc_vals = [bc.get(n, 0) for n in subG.nodes()]
            ec_vals = [ec.get(n, 0) for n in subG.nodes()]
            avg_bc = np.mean(bc_vals)
            avg_ec = np.mean(ec_vals)

            results.append({
                'Node': node,
                'Spread Proportion': spread_proportion,
                'Spread Efficiency': spread_efficiency,
                'Avg Path Length': avg_path_length,
                'Avg Betweenness Centrality': avg_bc,
                'Avg Eigenvector Centrality': avg_ec
            })

        df_metrics = pd.DataFrame(results)

        # Step 4: Normalize metrics
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(df_metrics.iloc[:, 1:])
        df_normalized = pd.DataFrame(normalized, columns=df_metrics.columns[1:])
        df_normalized['Node'] = df_metrics['Node']

        # Step 5: Compute CIS score
        df_normalized['CIS Score'] = df_normalized.drop(columns='Node').sum(axis=1)

        # Final CIS dataframe
        df_cis = df_normalized[['Node', 'CIS Score']].sort_values(by='CIS Score', ascending=False).reset_index(drop=True)

        G_rumor_undirected = G_rumor.to_undirected()
        partition = community_louvain.best_partition(G_rumor_undirected, random_state=42)

        # Number of communities
        num_communities = len(set(partition.values()))
        st.write(f"Number of detected communities: {num_communities}")

        # Generate layout positions for visualization
        pos = nx.spring_layout(G_rumor,k=1)

        # Extract edges from the rumor subgraph
        edge_x = []
        edge_y = []
        for edge in G_rumor.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # None creates breaks in line segments
            edge_y.extend([y0, y1, None])

        # Create an edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color="rgba(169, 169, 169, 0.5)"),  # Light grey edges
            hoverinfo="none",
            mode="lines"
        )

        # Normalize CIS Scores for visualization
        if 'CIS Score' in df_cis.columns:
            min_cis, max_cis = df_cis['CIS Score'].min(), df_cis['CIS Score'].max()
            df_cis['Normalized CIS Score'] = (df_cis['CIS Score'] - min_cis) / (max_cis - min_cis) * 20 + 5  # Scale size
        else:
            raise KeyError("CIS Score column not found in df_cis")

        # Create node scatter plot with community colors
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        for node in G_rumor.nodes():
            if node in df_cis.index:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_color.append(partition[node])  # Assign color based on community
                node_size.append(df_cis.loc[node, 'Normalized CIS Score'])  # Use CIS Score for size
            else:
                print(f"Warning: Node {node} not found in df_cis")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                showscale=True,
                colorscale="Viridis",  # Color scheme for communities
                size=node_size,  # Set size based on CIS Score
                color=node_color,
                colorbar=dict(
                    title="Community",
                    title_font=dict(color='white'),
                    tickfont=dict(color='white')
                ),
            ),
            text=[f"Node {node}, Community {partition[node]}<br>CIS Score: {df_cis.loc[node, 'CIS Score']:.4f}" for node in df_cis.index if node in G_rumor.nodes()],
            hoverinfo="text"
        )

        # Create interactive plot
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Community Detection in Rumor Subgraph (Louvain) - {num_communities} Communities",
            title_x=0.5,
            title_font=dict(color='white'),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=40),
            paper_bgcolor="black",
            plot_bgcolor="black",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(169, 169, 169, 0.2)",
                zeroline=False,
                showticklabels=False,
                linecolor="rgba(169, 169, 169, 0.5)"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(169, 169, 169, 0.2)",
                zeroline=False,
                showticklabels=False,
                linecolor="rgba(169, 169, 169, 0.5)"
            )
        )

        st.plotly_chart(fig)
        st.success("Community detection completed!")


@st.cache_data
def load_data():
    train_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="train")
    test_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="test")
    val_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="val")
   
    st.write(f"Train Samples: {len(train_data)}")
    st.write(f"Test Samples: {len(test_data)}")
    st.write(f"Val Samples: {len(val_data)}")
   
    return train_data, test_data, val_data

def prepare_graph(datasets):
    edges = []
    node_labels = {}

    for data in datasets:
        edge_index = data.edge_index.numpy()
        label = data.y.item()
        for src, tgt in zip(edge_index[0], edge_index[1]):
            edges.append((src, tgt, label))
            node_labels[src] = label
            node_labels[tgt] = label

    df_edges = pd.DataFrame(edges, columns=["source", "target", "label"])
   
    # Only keep rumor edges (label=1)
    df_edges = df_edges[df_edges['label'] == 1]
   
    # Count occurrences and create weighted edges
    df_edges['weight'] = df_edges.groupby(['source', 'target'])['source'].transform('count')
    df_edges = df_edges.drop_duplicates().reset_index(drop=True)

    # Create graph
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target",
                               edge_attr='weight', create_using=nx.DiGraph())
   
    # Get rumor nodes
    rumor_nodes = [node for node, label in node_labels.items() if label == 1]
    G_rumor = G.subgraph(rumor_nodes)
   
    return G_rumor, node_labels

def run_IC(G, seed, p=1.0):
    active_nodes = set([seed])
    newly_active = set([seed])
    all_active_t = [set(active_nodes)]
   
    while newly_active:
        next_newly_active = set()
        for node in newly_active:
            for neighbor in G.successors(node):
                if neighbor not in active_nodes:
                    if random.random() < p:
                        next_newly_active.add(neighbor)
        active_nodes.update(next_newly_active)
        newly_active = next_newly_active
        all_active_t.append(set(active_nodes))
   
    return active_nodes, all_active_t

def calculate_cis_scores(G):
    all_nodes = list(G.nodes())
   
    # Calculate centrality measures
    bc = nx.betweenness_centrality(G, normalized=True, weight='weight')
    G_undirected = G.to_undirected()
    ec = nx.eigenvector_centrality_numpy(G_undirected)
   
    # Run IC model simulations and gather metrics
    results = []
    progress_bar = st.progress(0)
   
    for i, node in enumerate(all_nodes):
        active_nodes_ic, all_active_t_ic = run_IC(G, node)
        final_spread_size = len(active_nodes_ic)
        spread_proportion = final_spread_size / len(G)
        time_to_saturation = len(all_active_t_ic) - 1
        spread_efficiency = final_spread_size / time_to_saturation if time_to_saturation else 0

        # Create subgraph of activated nodes
        subG = G.subgraph(active_nodes_ic)

        # Calculate average path length
        try:
            lengths = nx.single_source_shortest_path_length(subG, node)
            avg_path_length = np.mean(list(lengths.values()))
        except:
            avg_path_length = 0

        # Calculate average centrality metrics
        bc_vals = [bc.get(n, 0) for n in subG.nodes()]
        ec_vals = [ec.get(n, 0) for n in subG.nodes()]
        avg_bc = np.mean(bc_vals)
        avg_ec = np.mean(ec_vals)

        results.append({
            'Node': node,
            'Spread Proportion': spread_proportion,
            'Spread Efficiency': spread_efficiency,
            'Avg Path Length': avg_path_length,
            'Avg Betweenness Centrality': avg_bc,
            'Avg Eigenvector Centrality': avg_ec
        })
       
        # Update progress bar
        progress_bar.progress((i + 1) / len(all_nodes))

    df_metrics = pd.DataFrame(results)

    # Normalize and compute CIS scores
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df_metrics.iloc[:, 1:])
    df_normalized = pd.DataFrame(normalized, columns=df_metrics.columns[1:])
    df_normalized['Node'] = df_metrics['Node']
    df_normalized['CIS Score'] = df_normalized.drop(columns='Node').sum(axis=1)
   
    # Final CIS dataframe
    df_cis = df_normalized[['Node', 'CIS Score']].sort_values(by='CIS Score', ascending=False).reset_index(drop=True)
   
    return df_cis, bc, ec

def detect_communities(G):
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
   
    # Run Louvain community detection
    partition = community.best_partition(G_undirected)
   
    # Number of communities
    num_communities = len(set(partition.values()))
    st.write(f"Number of detected communities: {num_communities}")
   
    return partition

def plot_ic_with_slider(G, active_nodes_time_steps, df_cis, central_node, model_name="IC"):
    pos = nx.spring_layout(G)
   
    # Normalize CIS Scores for visualization
    min_cis, max_cis = df_cis["CIS Score"].min(), df_cis["CIS Score"].max()
    df_cis["Normalized CIS Score"] = 5 + 20 * (df_cis["CIS Score"] - min_cis) / (max_cis - min_cis + 1e-6)
   
    # Map node sizes based on CIS Score
    node_sizes = {row["Node"]: row["Normalized CIS Score"] for _, row in df_cis.iterrows()}
   
    # Extract edges for visualization

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        mode='lines',
        hoverinfo='none'
    )

    frames = []
    cumulative_activated = set()

    for t, active_nodes in enumerate(active_nodes_time_steps):
        node_x, node_y, node_color, node_size, hover_text = [], [], [], [], []
       
        current_active = set(active_nodes)
        new_activations = current_active - cumulative_activated
       
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
           
            if t == 0 and node == central_node:
                color = "red"
                status = "Initial Active Node"
            elif node in new_activations:
                color = "yellow"
                status = "Newly Activated"
            elif node in cumulative_activated:
                color = "red"
                status = "Previously Activated"
            else:
                color = "lightgray"
                status = "Inactive"
               
            node_color.append(color)
            node_size.append(node_sizes.get(node, 10))
           
            # Get CIS score for hover text
            score = df_cis[df_cis['Node'] == node]['CIS Score'].values[0] if node in df_cis['Node'].values else 0
            hover_text.append(f"Node {node}<br>CIS Score: {score:.4f}<br>Status: {status}")
           
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color='black')),
            text=hover_text,
            hoverinfo='text',
        )
       
        frames.append(go.Frame(
            data=[edge_trace, node_trace],
            name=str(t),
            layout=go.Layout(title_text=f"{model_name} Model - Time Step {t}")
        ))
       
        cumulative_activated.update(current_active)
   
    # Create figure with frames
    fig = go.Figure(
        data=[edge_trace, frames[0].data[1]],
        layout=go.Layout(
            title=f"{model_name} Model - Time Step 0",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',

            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1150}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "x": 0.1,
                "y": -0.1,
                "showactive": True
            }],
            sliders=[{
                "active": 0,
                "y": -0.1,
                "x": 0.25,
                "len": 0.6,
                "pad": {"b": 10},
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}]
                    } for i in range(len(frames))
                ]
            }]
        ),
        frames=frames
    )
   
    st.plotly_chart(fig, use_container_width=True)

def backtrack_high_influence_path(G, selected_node, node_probabilities, origin_node=None):
    G_weighted = nx.DiGraph()
    for u, v in G.edges():
        prob = node_probabilities.get(u, 0.1)
        weight = 1 / (prob + 1e-6)
        G_weighted.add_edge(v, u, weight=weight)  # Note: reversed edges for backtracking

    if origin_node is None:
        origin_node = max(node_probabilities, key=node_probabilities.get)

    try:
        return nx.shortest_path(G_weighted, source=selected_node, target=origin_node, weight="weight")
    except nx.NetworkXNoPath:
        return []

def plot_backtracking_with_slider(G, selected_node, df_cis, node_probabilities):
    pos = nx.spring_layout(G)

    # Get node size from normalized CIS score
    def get_node_size(node):
        size = df_cis.loc[df_cis["Node"] == node, "Normalized CIS Score"]
        return size.values[0] if not size.empty else 10

    # Draw all edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        mode='lines',
        hoverinfo='none'
    )

    # Calculate backtracked path
    backtracked_path = backtrack_high_influence_path(G, selected_node, node_probabilities)
   
    if not backtracked_path:
        st.error(f"No path found from node {selected_node} to the source node.")
        return

    frames = []
    for step in range(1, len(backtracked_path) + 1):
        node_x, node_y, node_color, node_size, hover_text = [], [], [], [], []
        partial_path = backtracked_path[:step]

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            if node == selected_node:
                color = "yellow"
                status = "Selected Node"
            elif node in partial_path:
                color = "blue"
                status = "On Backtrack Path"
            else:
                color = "lightgray"
                status = "Inactive"

            score_series = df_cis.loc[df_cis["Node"] == node, "CIS Score"]
            score = score_series.values[0] if not score_series.empty else 0
            size = get_node_size(node)
            node_color.append(color)
            node_size.append(size)
            hover_text.append(f"Node {node}<br>CIS Score: {score:.4f}<br>Status: {status}")

        # Edges in path up to this step
        path_x, path_y = [], []
        for i in range(step - 1):
            u, v = backtracked_path[i], backtracked_path[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            path_x += [x0, x1, None]
            path_y += [y0, y1, None]

        backtrack_trace = go.Scatter(
            x=path_x, y=path_y,
            line=dict(width=3, color='blue'),
            mode='lines',
            hoverinfo='none'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color='black')),
            text=hover_text,
            hoverinfo='text',
        )

        frames.append(go.Frame(
            data=[edge_trace, node_trace, backtrack_trace],
            name=str(step - 1),
            layout=go.Layout(title_text=f"Backtracking Step {step - 1}")
        ))

    # Initial frame
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=f"Backtracking from Node {selected_node}",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "x": 0.1,
                "y": -0.1,
                "showactive": True
            }],
            sliders=[{
                "active": 0,
                "y": -0.1,
                "x": 0.25,
                "len": 0.6,
                "pad": {"b": 10},
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}]
                    } for i in range(len(frames))
                ]
            }]
        ),
        frames=frames
    )

    st.plotly_chart(fig, use_container_width=True)

def run_custom_simulation(G, df_cis):
    # Calculate influence probabilities based on CIS scores
    min_cis, max_cis = df_cis["CIS Score"].min(), df_cis["CIS Score"].max()
    df_cis["Influence Probability"] = 0.1 + 0.9 * (df_cis["CIS Score"] - min_cis) / (max_cis - min_cis + 1e-6)
    node_probabilities = dict(zip(df_cis["Node"], df_cis["Influence Probability"]))
   
    # Select the most influential node as central node
    central_node = df_cis.iloc[0]['Node']
   
    # Run IC model with dynamic probabilities
    G_ic = G.copy()
    active_nodes_ic = set([central_node])
    newly_active = set([central_node])
    all_active_t_ic = [set(active_nodes_ic)]

    while newly_active:
        next_newly_active = set()
        for node in newly_active:
            for neighbor in G_ic.successors(node):
                if neighbor not in active_nodes_ic:
                    p = node_probabilities.get(node, 0.1)
                    if random.random() < p:
                        next_newly_active.add(neighbor)
        active_nodes_ic.update(next_newly_active)
        newly_active = next_newly_active
        all_active_t_ic.append(set(active_nodes_ic))
   
    return central_node, all_active_t_ic, node_probabilities

## New
# Initialize session state variables
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'backtrack_run' not in st.session_state:
    st.session_state.backtrack_run = False

@st.cache_data
def load_data():
    train_data = UPFD(root=".", name="gossipcop", feature="content", split="train")
    test_data = UPFD(root=".", name="gossipcop", feature="content", split="test")
    val_data = UPFD(root=".", name="gossipcop", feature="content", split="val")
    
    st.write(f"Train Samples: {len(train_data)}")
    st.write(f"Test Samples: {len(test_data)}")
    st.write(f"Val Samples: {len(val_data)}")
    
    return train_data, test_data, val_data

def prepare_graph(datasets):
    edges = []
    node_labels = {}

    for data in datasets:
        edge_index = data.edge_index.numpy()
        label = data.y.item()
        for src, tgt in zip(edge_index[0], edge_index[1]):
            edges.append((src, tgt, label))
            node_labels[src] = label
            node_labels[tgt] = label

    df_edges = pd.DataFrame(edges, columns=["source", "target", "label"])
    
    # Only keep rumor edges (label=1)
    df_edges = df_edges[df_edges['label'] == 1]
    
    # Count occurrences and create weighted edges
    df_edges['weight'] = df_edges.groupby(['source', 'target'])['source'].transform('count')
    df_edges = df_edges.drop_duplicates().reset_index(drop=True)

    # Create graph
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target", 
                               edge_attr='weight', create_using=nx.DiGraph())
    
    # Get rumor nodes
    rumor_nodes = [node for node, label in node_labels.items() if label == 1]
    G_rumor = G.subgraph(rumor_nodes)
    
    return G_rumor, node_labels

def run_IC(G, seed, p=1.0):
    active_nodes = set([seed])
    newly_active = set([seed])
    all_active_t = [set(active_nodes)]
    
    while newly_active:
        next_newly_active = set()
        for node in newly_active:
            for neighbor in G.successors(node):
                if neighbor not in active_nodes:
                    if random.random() < p:
                        next_newly_active.add(neighbor)
        active_nodes.update(next_newly_active)
        newly_active = next_newly_active
        all_active_t.append(set(active_nodes))
    
    return active_nodes, all_active_t

def calculate_cis_scores(G):
    all_nodes = list(G.nodes())
    
    # Calculate centrality measures
    bc = nx.betweenness_centrality(G, normalized=True, weight='weight')
    G_undirected = G.to_undirected()
    ec = nx.eigenvector_centrality_numpy(G_undirected)
    
    # Run IC model simulations and gather metrics
    results = []
    progress_bar = st.progress(0)
    
    for i, node in enumerate(all_nodes):
        active_nodes_ic, all_active_t_ic = run_IC(G, node)
        final_spread_size = len(active_nodes_ic)
        spread_proportion = final_spread_size / len(G)
        time_to_saturation = len(all_active_t_ic) - 1
        spread_efficiency = final_spread_size / time_to_saturation if time_to_saturation else 0

        # Create subgraph of activated nodes
        subG = G.subgraph(active_nodes_ic)

        # Calculate average path length
        try:
            lengths = nx.single_source_shortest_path_length(subG, node)
            avg_path_length = np.mean(list(lengths.values()))
        except:
            avg_path_length = 0

        # Calculate average centrality metrics
        bc_vals = [bc.get(n, 0) for n in subG.nodes()]
        ec_vals = [ec.get(n, 0) for n in subG.nodes()]
        avg_bc = np.mean(bc_vals)
        avg_ec = np.mean(ec_vals)

        results.append({
            'Node': node,
            'Spread Proportion': spread_proportion,
            'Spread Efficiency': spread_efficiency,
            'Avg Path Length': avg_path_length,
            'Avg Betweenness Centrality': avg_bc,
            'Avg Eigenvector Centrality': avg_ec
        })
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(all_nodes))

    df_metrics = pd.DataFrame(results)

    # Normalize and compute CIS scores
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df_metrics.iloc[:, 1:])
    df_normalized = pd.DataFrame(normalized, columns=df_metrics.columns[1:])
    df_normalized['Node'] = df_metrics['Node']
    df_normalized['CIS Score'] = df_normalized.drop(columns='Node').sum(axis=1)
    
    # Final CIS dataframe
    df_cis = df_normalized[['Node', 'CIS Score']].sort_values(by='CIS Score', ascending=False).reset_index(drop=True)
    
    return df_cis, bc, ec

def detect_communities(G):
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Run Louvain community detection
    partition = community.best_partition(G_undirected)
    
    # Number of communities
    num_communities = len(set(partition.values()))
    st.write(f"Number of detected communities: {num_communities}")
    
    return partition

def plot_ic_with_slider(G, active_nodes_time_steps, df_cis, central_node, model_name="IC"):
    pos = nx.spring_layout(G, k=1)
    
    # Normalize CIS Scores for visualization
    min_cis, max_cis = df_cis["CIS Score"].min(), df_cis["CIS Score"].max()
    df_cis["Normalized CIS Score"] = 5 + 20 * (df_cis["CIS Score"] - min_cis) / (max_cis - min_cis + 1e-6)
    
    # Map node sizes based on CIS Score
    node_sizes = {row["Node"]: row["Normalized CIS Score"] for _, row in df_cis.iterrows()}
    
    # Extract edges for visualization
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        mode='lines',
        hoverinfo='none'
    )

    frames = []
    cumulative_activated = set()

    for t, active_nodes in enumerate(active_nodes_time_steps):
        node_x, node_y, node_color, node_size, hover_text = [], [], [], [], []
        
        current_active = set(active_nodes)
        new_activations = current_active - cumulative_activated
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if t == 0 and node == central_node:
                color = "red"
                status = "Initial Active Node"
            elif node in new_activations:
                color = "yellow"
                status = "Newly Activated"
            elif node in cumulative_activated:
                color = "red"
                status = "Previously Activated"
            else:
                color = "lightgray"
                status = "Inactive"
                
            node_color.append(color)
            node_size.append(node_sizes.get(node, 10))
            
            # Get CIS score for hover text
            score = df_cis[df_cis['Node'] == node]['CIS Score'].values[0] if node in df_cis['Node'].values else 0
            hover_text.append(f"Node {node}<br>CIS Score: {score:.4f}<br>Status: {status}")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color='black')),
            text=hover_text,
            hoverinfo='text',
        )
        
        frames.append(go.Frame(
            data=[edge_trace, node_trace],
            name=str(t),
            layout=go.Layout(title_text=f"{model_name} Model - Time Step {t}")
        ))
        
        cumulative_activated.update(current_active)
    
    # Create figure with frames
    fig = go.Figure(
        data=[edge_trace, frames[0].data[1]],
        layout=go.Layout(
            title=f"{model_name} Model - Time Step 0",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='gray', zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1150}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "x": 0.1,
                "y": -0.1,
                "showactive": True
            }],
            sliders=[{
                "active": 0,
                "y": -0.1,
                "x": 0.25,
                "len": 0.6,
                "pad": {"b": 10},
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}]
                    } for i in range(len(frames))
                ]
            }]
        ),
        frames=frames
    )
    
    st.plotly_chart(fig, use_container_width=True)

def backtrack_high_influence_path(G, selected_node, node_probabilities, origin_node=None):
    G_weighted = nx.DiGraph()
    for u, v in G.edges():
        prob = node_probabilities.get(u, 0.1)
        weight = 1 / (prob + 1e-6)
        G_weighted.add_edge(v, u, weight=weight)  # Note: reversed edges for backtracking

    if origin_node is None:
        origin_node = max(node_probabilities, key=node_probabilities.get)

    try:
        return nx.shortest_path(G_weighted, source=selected_node, target=origin_node, weight="weight")
    except nx.NetworkXNoPath:
        return []

def plot_backtracking_with_slider(G, selected_node, df_cis, node_probabilities):
    pos = nx.spring_layout(G,k=1)

    # Get node size from normalized CIS score
    def get_node_size(node):
        size = df_cis.loc[df_cis["Node"] == node, "Normalized CIS Score"]
        return size.values[0] if not size.empty else 10

    # Draw all edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        mode='lines',
        hoverinfo='none'
    )

    # Calculate backtracked path
    backtracked_path = backtrack_high_influence_path(G, selected_node, node_probabilities)
    
    if not backtracked_path:
        st.error(f"No path found from node {selected_node} to the source node.")
        return

    frames = []
    for step in range(1, len(backtracked_path) + 1):
        node_x, node_y, node_color, node_size, hover_text = [], [], [], [], []
        partial_path = backtracked_path[:step]

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            if node == selected_node:
                color = "yellow"
                status = "Selected Node"
            elif node in partial_path:
                color = "blue"
                status = "On Backtrack Path"
            else:
                color = "lightgray"
                status = "Inactive"

            score_series = df_cis.loc[df_cis["Node"] == node, "CIS Score"]
            score = score_series.values[0] if not score_series.empty else 0
            size = get_node_size(node)
            node_color.append(color)
            node_size.append(size)
            hover_text.append(f"Node {node}<br>CIS Score: {score:.4f}<br>Status: {status}")

        # Edges in path up to this step
        path_x, path_y = [], []
        for i in range(step - 1):
            u, v = backtracked_path[i], backtracked_path[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            path_x += [x0, x1, None]
            path_y += [y0, y1, None]

        backtrack_trace = go.Scatter(
            x=path_x, y=path_y,
            line=dict(width=3, color='blue'),
            mode='lines',
            hoverinfo='none'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color='black')),
            text=hover_text,
            hoverinfo='text',
        )

        frames.append(go.Frame(
            data=[edge_trace, node_trace, backtrack_trace],
            name=str(step - 1),
            layout=go.Layout(title_text=f"Backtracking Step {step - 1}")
        ))

    # Initial frame
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=f"Backtracking from Node {selected_node}",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='gray', zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "x": 0.1,
                "y": -0.1,
                "showactive": True
            }],
            sliders=[{
                "active": 0,
                "y": -0.1,
                "x": 0.25,
                "len": 0.6,
                "pad": {"b": 10},
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}]
                    } for i in range(len(frames))
                ]
            }]
        ),
        frames=frames
    )

    st.plotly_chart(fig, use_container_width=True)

def run_custom_simulation(G, df_cis):
    # Calculate influence probabilities based on CIS scores
    min_cis, max_cis = df_cis["CIS Score"].min(), df_cis["CIS Score"].max()
    df_cis["Influence Probability"] = 0.1 + 0.9 * (df_cis["CIS Score"] - min_cis) / (max_cis - min_cis + 1e-6)
    node_probabilities = dict(zip(df_cis["Node"], df_cis["Influence Probability"]))
    
    # Select the most influential node as central node
    central_node = df_cis.iloc[0]['Node']
    
    # Run IC model with dynamic probabilities
    G_ic = G.copy()
    pos = nx.spring_layout(G_ic,k=1)

    active_nodes_ic = set([central_node])
    newly_active = set([central_node])
    all_active_t_ic = [set(active_nodes_ic)]

    while newly_active:
        next_newly_active = set()
        for node in newly_active:
            for neighbor in G_ic.successors(node):
                if neighbor not in active_nodes_ic:
                    p = node_probabilities.get(node, 0.1)
                    if random.random() < p:
                        next_newly_active.add(neighbor)
        active_nodes_ic.update(next_newly_active)
        newly_active = next_newly_active
        all_active_t_ic.append(set(active_nodes_ic))
    
    return central_node, all_active_t_ic, node_probabilities

# Main app execution
def main():
    st.header("CIS-Based Influence Simulation")
    
    # Initialize session state variables if they don't exist
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = None
    
    if st.button("Run CIS-Based Influence Simulation", key="run_simulation_button"):
        with st.spinner("Loading data..."):
            train_data, test_data, val_data = load_data()
            datasets = train_data + test_data
        
        with st.spinner("Preparing graph..."):
            G_rumor, node_labels = prepare_graph(datasets)
            st.write(f"Graph contains {len(G_rumor.nodes())} nodes and {len(G_rumor.edges())} edges")
        
        with st.spinner("Calculating CIS scores..."):
            df_cis, bc, ec = calculate_cis_scores(G_rumor)
            st.dataframe(df_cis.head(10))
        
        with st.spinner("Detecting communities..."):
            partition = detect_communities(G_rumor)
        
        with st.spinner("Running influence spread simulation..."):
            central_node, all_active_t_ic, node_probabilities = run_custom_simulation(G_rumor, df_cis)
            st.write(f"Starting from most influential node: {central_node}")
            st.write(f"Final spread reached {len(all_active_t_ic[-1])} nodes in {len(all_active_t_ic)-1} time steps")
        
        # Store results in session state
        st.session_state.simulation_data = {
            'G_rumor': G_rumor,
            'df_cis': df_cis,
            'all_active_t_ic': all_active_t_ic,
            'central_node': central_node,
            'node_probabilities': node_probabilities
        }
        
        st.session_state.simulation_run = True
        
        with st.spinner("Generating visualization..."):
            st.subheader("Influence Spread Simulation")
            plot_ic_with_slider(G_rumor, all_active_t_ic, df_cis, central_node)
        
        st.success("Influence spread simulation completed!")
    
    # Only show backtracking section if simulation has been run
    if st.session_state.simulation_run:
        st.subheader("Influence Path Backtracking")
        
        # Get valid nodes for selection
        valid_nodes = list(st.session_state.simulation_data['G_rumor'].nodes())
        backtrack_node = st.selectbox(
            "Select a node to backtrack from:", 
            options=valid_nodes,
            index=0
        )
        
        if st.button("Run Backtracking", key="run_backtracking_button"):
            st.session_state.backtrack_run = True
            with st.spinner("Generating backtracking visualization..."):
                plot_backtracking_with_slider(
                    st.session_state.simulation_data['G_rumor'],
                    int(backtrack_node),
                    st.session_state.simulation_data['df_cis'],
                    st.session_state.simulation_data['node_probabilities']
                )

if __name__ == "__main__":
    main()


# Prediction Section for a Specific Sample
st.header("Test a Specific Sample")

# Select an index from the test set
index = st.number_input("Enter Test Sample Index:", min_value=0, step=1, value=0)

if st.button("Run Prediction"):
    pred, actual, data = get_test_sample(index)
    
    st.write(f"**Model Prediction:** {'📰 Fake' if pred == 1 else '✔️ Real'}")
    st.write(f"**Actual Label:** {'📰 Fake' if actual == 1 else '✔️ Real'}")
