import streamlit as st
import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
    st.subheader("Enter Transaction Data for Visualization")

    # Example data (replace with actual data)
    transaction_data = {
        "Transaction ID": ["T001", "T002", "T003", "T004", "T005"],
        "Amount": [500, 1000, 1500, 700, 300],
        "Sender": ["S1", "S2", "S1", "S3", "S2"],
        "Receiver": ["R1", "R2", "R1", "R3", "R2"],
    }

    # Create a DataFrame
    df = pd.DataFrame(transaction_data)

    # Display the transaction data as a table
    st.write("Transaction Data:")
    st.dataframe(df)

    # Visualization using NetworkX (for example, visualize the transactions as a graph)
    G = nx.DiGraph()

    # Add nodes and edges (example)
    for index, row in df.iterrows():
        G.add_edge(row['Sender'], row['Receiver'], weight=row['Amount'], label=row['Transaction ID'])

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for nodes
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    
    # Add labels for transaction amounts
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show the plot in Streamlit
    st.pyplot(plt)
