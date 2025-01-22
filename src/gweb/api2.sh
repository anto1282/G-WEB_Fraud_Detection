curl -X 'POST' 'http://localhost:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
        "node_features": [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1922.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 330.1664, 0.0, 0.0]],
        "edge_index": [[0, 1]],
        "edge_attr": [[0.45632, 787200, 13.0, 787200, 13.0, 3.0]],
        "batch_size": 1,
        "transaction_id": "test-transaction"
    }'
