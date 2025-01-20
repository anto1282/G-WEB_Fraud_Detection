import streamlit as st
import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch.optim.sgd
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import typer
from sklearn.metrics import f1_score
from model import GCN
from data import AMLtoGraph
import os


# API URL
api_url = "http://127.0.0.1:8000/predict/"

# Streamlit UI
st.title("G-WEB Fraud Detection Online")

# Show buttons for fraud detection and transaction visualization
option = st.selectbox("Choose an option", ["Visualize Transactions", "Start Fraud Detection"])

# Start Fraud Detection flow
if option == "Start Fraud Detection":
    st.header("Enter Transaction Data")

    # Input: Transaction ID
    transaction_id = st.text_input("Transaction ID")

    # Input: Node Features
    st.subheader("Node Features (List of floating point numbers)")
    node_features_input = st.text_area("Enter Node Features (e.g., [1.0, 2.0, 3.0, ...])", value="")
    node_features = np.array(eval(node_features_input), dtype=np.float32) if node_features_input else np.array([])

    # Input: Edge Indices
    st.subheader("Edge Index (List of pairs of integers)")
    edge_index_input = st.text_area("Enter Edge Index (e.g., [[0, 1], [1, 2], ...])", value="")
    edge_index = np.array(eval(edge_index_input), dtype=np.int64) if edge_index_input else np.array([])

    # Input: Edge Attributes
    st.subheader("Edge Attributes (List of floating point numbers)")
    edge_attr_input = st.text_area("Enter Edge Attributes (e.g., [0.1, 0.2, 0.3, ...])", value="")
    edge_attr = np.array(eval(edge_attr_input), dtype=np.float32) if edge_attr_input else np.array([])

    # Button to make prediction
    if st.button("Predict"):
        
        if node_features.size == 0 or edge_index.size == 0 or edge_attr.size == 0:
            st.error("Please enter valid node features, edge indices, and edge attributes.")
        else:
            # Prepare the data as a dictionary for POST request
            payload = {
                "node_features": node_features.tolist(),
                "edge_index": edge_index.tolist(),
                "edge_attr": edge_attr.tolist(),
                "batch_size": len(node_features),  # Assuming batch size is equal to number of nodes
                "transaction_id": transaction_id
            }

            try:
                # Send POST request to FastAPI
                response = requests.post(api_url, json=payload)

                # Check if the response is successful
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    st.success(f"Prediction: {prediction}")
                    st.write(f"Confidence: {confidence:.4f}")
                    st.write(f"Transaction ID: {result['transaction_id']}")
                else:
                    st.error(f"Error from API: {response.status_code}")
            except Exception as e:
                st.error(f"Error during prediction request: {e}")

# Visualize Transactions flow
elif option == "Visualize Transactions":
    st.header("Transaction Visualization")

    # Input: Transaction data (for visualization purposes)
    st.subheader("Information about Transaction Data")
    dataset = AMLtoGraph("/dtu/blackhole/0e/154958/data_small")  # Adjust path
    
    data = dataset[0]
    split = T.RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.1)
    data = split(data)
    test_loader = NeighborLoader(data,num_neighbors=[30] * 2,batch_size=1000,input_nodes=data.test_mask)
    
    
        
    for batch in test_loader:
        break  # We only need the first batch for visualization

    # Extract the batch's node indices (n_id) and the corresponding node features (x) and labels (y)
    node_ids = batch.n_id.numpy()  # Node indices for the current batch
    node_features = data.x[node_ids]  # Features for the nodes in the batch
    node_labels = data.y[node_ids]  # Labels for the nodes in the batch

    # Find fraud nodes (assuming 1 is fraud)
    fraud_node_ids = node_ids[node_labels == 1]
    non_fraud_node_ids = node_ids[node_labels == 0]

    # Create a subgraph that contains only fraud nodes
    # Extract the edges corresponding to the fraud nodes
    edge_index_batch = batch.edge_index.numpy()
    
    # Showing the fraud graph
    # Filter edges that connect to fraud nodes
    fraud_edges = []
    for edge in zip(edge_index_batch[0], edge_index_batch[1]):
        if edge[0] in fraud_node_ids and edge[1] in fraud_node_ids:
            fraud_edges.append(edge)

    # Create a subgraph using NetworkX
    G_batch_fraud = nx.Graph()

    # Add the fraud edges to the subgraph
    G_batch_fraud.add_edges_from(fraud_edges)
    G_batch_fraud.remove_edges_from([edge for edge in G_batch_fraud.edges() if edge[0] == edge[1]])

    # Create node positions for visualization (using the feature vectors as positions)
    node_positions_fraud = {node: (node_features[i, 0], node_features[i, 1]) 
                            for i, node in enumerate(node_ids) if node in fraud_node_ids}

    # Plot the fraud-only subgraph
    plt.figure(figsize=(8, 8))
    nx.draw(G_batch_fraud, pos=node_positions_fraud, with_labels=True, node_color='red', node_size=300, font_size=10)
    plt.title("Fraud Subgraph Visualization")
    st.pyplot(plt)
    # showing the non fraud graph 
    non_fraud_edges = []
    for edge in zip(edge_index_batch[0], edge_index_batch[1]):
        if edge[0] in non_fraud_node_ids and edge[1] in non_fraud_node_ids:
            non_fraud_edges.append(edge)

    # Create a subgraph using NetworkX
    G_batch_nonfraud = nx.Graph()

    # Add the fraud edges to the subgraph
    G_batch_nonfraud.add_edges_from(non_fraud_edges)
    G_batch_nonfraud.remove_edges_from([edge for edge in G_batch_nonfraud.edges() if edge[0] == edge[1]])

    # Create node positions for visualization (using the feature vectors as positions)
    node_positions_nonfraud = {node: (node_features[i, 0], node_features[i, 1]) 
                            for i, node in enumerate(node_ids) if node in non_fraud_node_ids}

    # Plot the fraud-only subgraph
    plt.figure(figsize=(8, 8))
    nx.draw(G_batch_nonfraud, pos=node_positions_nonfraud, with_labels=True, node_color='blue', node_size=300, font_size=10)
    plt.title("Normal transactions Subgraph Visualization")
    st.pyplot(plt)

    
        # Convert the fraud subgraph to a directed graph
    degreesfraud = [G_batch_fraud.degree(n) for n in G_batch_fraud.nodes()]
    degreesnonfraud = [G_batch_nonfraud.degree(n) for n in G_batch_nonfraud.nodes()]
    
    
    plt.figure(figsize=(8, 6))

# Plot the fraud degree distribution (in red)
    

# Plot the non-fraud degree distribution (in blue)
    plt.hist(degreesnonfraud, bins=10, alpha=0.7, color='blue', label='Non-Fraud')
    plt.hist(degreesfraud, bins=10, alpha=0.7, color='red', label='Fraud')

# Add labels and title
    plt.title("Degree Distribution for Fraud and Non-Fraud Nodes")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")

# Add a legend to differentiate between fraud and non-fraud
    plt.legend()
    plt.yscale('log')
    st.pyplot(plt)


    # Display the transaction data as a table
    st.write(f"Transaction Data: There were {len(degreesfraud)} fraud attempts and {len(degreesnonfraud) } normal transactions")
    

    

    # Show the plot in Streamlit
    #st.pyplot(plt)
