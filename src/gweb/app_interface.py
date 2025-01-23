import streamlit as st
import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# API URL for prediction and visualization
api_url_predict = "http://127.0.0.1:8000/predict/"
api_url_dataviz = "http://127.0.0.1:8000/dataviz/"
api_url_system_metrics = "http://127.0.0.1:8000/system-metrics/"  # New endpoint for system metrics

# Streamlit UI
st.title("G-WEB Fraud Detection Online")

# Sidebar for navigation
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an option", ["Start Fraud Detection", "Visualize Transactions"])

# System Metrics button in the top-left corner
if st.sidebar.button('Show System Metrics'):
    try:
        # Request system metrics from the FastAPI server
        response = requests.get(api_url_system_metrics)
        response.raise_for_status()  # Raise an exception for HTTP errors
        system_metrics = response.json()  # Get the system metrics in JSON format

        # Display system metrics
        st.subheader("System Metrics")
        st.write("CPU Usage: {}%".format(system_metrics['cpu_percent']))
        st.write("Memory Usage: {}% (Used: {} MB, Total: {} MB)".format(
            system_metrics['memory_percent'], 
            system_metrics['memory_used_MB'], 
            system_metrics['memory_total_MB']
        ))
        st.write("Disk Usage: {}% (Used: {} GB, Total: {} GB)".format(
            system_metrics['disk_percent'],
            system_metrics['disk_used_GB'],
            system_metrics['disk_total_GB']
        ))
        st.write("Total Requests: {}".format(system_metrics['total_requests']))
        st.write("Total Errors: {}".format(system_metrics['total_errors']))
    except Exception as e:
        st.error(f"Failed to retrieve system metrics: {str(e)}")

# Start Fraud Detection flow
if option == "Start Fraud Detection":
    st.header("Press button to start testing model on fraud data")
    start_button = st.button("Start Test", help="Click to start fraud detection test")

    if start_button:
        with st.spinner('The model is testing on data...'):
            response = requests.post(api_url_predict)
            response.raise_for_status()  # Raise an exception for HTTP errors
            results = response.json()  # Get the results from the response

        # Display the results
        accuracy = results.get("accuracy", "Not available")
        cm = results.get("confusion_matrix", "Not available")

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
    st.header("Transaction Visualization")
    with st.spinner('A Random Batch From The Test Set Is Loading...'):
        response = requests.post(api_url_dataviz)
        response.raise_for_status()  # Raise an exception for HTTP errors
        results = response.json()

    edges = results.get("edges", "Not available")
    fraud_edges = results.get("fraud_edges", "Not available")
    non_fraud_edges = results.get("non_fraud_edges", "Not available")

    st.subheader("Visualization of Random Batch From The Test Set")

    # Entire graph 
    G = nx.Graph()
    G.add_edges_from(edges)
    G.remove_edges_from([edge for edge in G.edges() if edge[0] == edge[1]])
    node_positions = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=node_positions, with_labels=False, node_color='grey', node_size=100, font_size=10)
    plt.title("Graph Visualization")
    st.pyplot(plt)

    # Non-fraud Graph
    G_batch_nonfraud = nx.Graph()
    G_batch_nonfraud.add_edges_from(non_fraud_edges)
    G_batch_nonfraud.remove_edges_from([edge for edge in G_batch_nonfraud.edges() if edge[0] == edge[1]])
    node_positions_nonfraud = nx.circular_layout(G_batch_nonfraud)
    plt.figure(figsize=(8, 8))
    nx.draw(G_batch_nonfraud, pos=node_positions_nonfraud, with_labels=False, node_color='blue', node_size=100, font_size=10)
    plt.title("Normal transactions Subgraph Visualization")
    st.pyplot(plt)

    # Fraud Graph
    G_batch_fraud = nx.Graph()
    G_batch_fraud.add_edges_from(fraud_edges)
    G_batch_fraud.remove_edges_from([edge for edge in G_batch_fraud.edges() if edge[0] == edge[1]])
    node_positions_fraud = nx.circular_layout(G_batch_fraud)
    plt.figure(figsize=(8, 8))
    nx.draw(G_batch_fraud, pos=node_positions_fraud, with_labels=False, node_color='red', node_size=100, font_size=10)
    plt.title("Fraud Subgraph Visualization")
    st.pyplot(plt)

    # Plot of degree distribution
    degreesfraud = [G_batch_fraud.degree(n) for n in G_batch_fraud.nodes()]
    degreesnonfraud = [G_batch_nonfraud.degree(n) for n in G_batch_nonfraud.nodes()]
    plt.figure(figsize=(8, 6))
    plt.hist(degreesnonfraud, bins=10, alpha=0.7, color='blue', label='Non-Fraud')
    plt.hist(degreesfraud, bins=10, alpha=0.7, color='red', label='Fraud')
    plt.title("Degree Distribution for Fraud and Non-Fraud Nodes")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")

    plt.legend()
    plt.yscale('log')
    st.pyplot(plt)

    # Display the transaction data as a table
    st.write(f"Transaction Data: There were {len(degreesfraud)} fraud attempts and {len(degreesnonfraud)} normal transactions")
