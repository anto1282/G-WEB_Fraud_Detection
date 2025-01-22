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
import time 
from evaluate_API import test
import seaborn as sns 
#from evaluate import test 


# API URL for prediction
api_url = "http://127.0.0.1:8000/predict/"

# Streamlit UI
st.title("G-WEB Fraud Detection Online")

# Show buttons for fraud detection and transaction visualization
option = st.selectbox("Choose an option", ["Start Fraud Detection", "Visualize Transactions"])

# Start Fraud Detection flow
if option == "Start Fraud Detection":
    st.header("Press button to start testing model on fraud data")
    # possible layout for training... 
    # should be changed to testing. 

    # Input: Transaction ID
    # Show the "Start Test" button to begin the input flow
    start_button = st.button("Start Test", help="Click to start fraud detection test")

    if start_button:
        # Once the button is clicked, show the inputs for transaction data
        with st.spinner('The model is testing on data...'):
            response = requests.post(api_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
                # Get the results from the response
            results = response.json()  # Should return a JSON with 'accuracy' and 'confusion_matrix'
                # Display the results in the Streamlit UI
            accuracy = results.get("accuracy", "Not available")
            cm = results.get("confusion_matrix", "Not available")


    # After evaluation, display a success message
        st.success("Model evaluation completed!")
        st.header(f"Accuracy of model was {accuracy:.3f}")
        class_names = ["Normal", "Fraud"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        

# Visualize Transactions flow
elif option == "Visualize Transactions":
    st.header("Transaction Visualization of batch from the testset")

    # Input: Transaction data (for visualization purposes)
    st.subheader("Information about Transaction Data")
    dataset = AMLtoGraph("/dtu/blackhole/0e/154958/data_small")  # Adjust path
    
    data = dataset[0]
    split = T.RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.1)
    data = split(data)
    test_loader = NeighborLoader(data,num_neighbors=[30] * 2,batch_size=600,input_nodes=data.test_mask)
    
    
        
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
    # showing entire graph
    
    
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
    node_positions_nonfraud = nx.circular_layout(G_batch_nonfraud)

    # Plot the fraud-only subgraph
    plt.figure(figsize=(8, 8))
    nx.draw(G_batch_nonfraud, pos=node_positions_nonfraud, with_labels=False, node_color='blue', node_size=100, font_size=10)
    plt.title(f"Normal transactions Subgraph Visualization")
    st.pyplot(plt)
    
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
    node_positions_fraud = nx.circular_layout(G_batch_fraud)

    # Plot the fraud-only subgraph
    plt.figure(figsize=(8, 8))
    nx.draw(G_batch_fraud, pos=node_positions_fraud, with_labels=False, node_color='red', node_size=100, font_size=10)
    plt.title("Fraud Subgraph Visualization")
    st.pyplot(plt)
    
    
    edges = []
    for edge in zip(edge_index_batch[0], edge_index_batch[1]):
        if (edge[0] in non_fraud_node_ids and edge[1] in non_fraud_node_ids) or (edge[0] in fraud_node_ids and edge[1] in fraud_node_ids) :
            edges.append(edge)

    # Create a graph using NetworkX
    G = nx.Graph()
    

    # Add the edges to the subgraph
    
    G.add_edges_from(edges)
    G.remove_edges_from([edge for edge in G.edges() if edge[0] == edge[1]])

    # Create node positions for visualization (using the feature vectors as positions)
    # Use NetworkX's circular layout to compute positions
    node_positions = nx.circular_layout(G)
    

    # Plot the fraud-only subgraph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=node_positions, with_labels=False, node_color='grey', node_size=100, font_size=10)
    plt.title("Graph Visualization")
    st.pyplot(plt)
    
    
   

    
        # Convert the fraud subgraph to a directed graph
    degreesfraud = [G_batch_fraud.degree(n) for n in G_batch_fraud.nodes()]
    degreesnonfraud = [G_batch_nonfraud.degree(n) for n in G_batch_nonfraud.nodes()]
    
    
    plt.figure(figsize=(8, 6))

    

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
