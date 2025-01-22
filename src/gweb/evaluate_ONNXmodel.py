import onnx
import onnxruntime as ort

def test(
    model_path: str = "/zhome/45/0/155089/G-WEB_Fraud_Detection/models/model_lr-2_83e-04_bs-1024_dropout-0_55_epochs-50.onnx",
    batchsize: int = 32,
    hdn_chnls: int = 16,
    atn_heads: int = 4,
    drop_out: float = 0.6,
) -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load dataset
    dataset = AMLtoGraph("/dtu/blackhole/0e/154958/data")
    data = dataset[0]

    # Model parameters
    hidden_channels = hdn_chnls
    heads = atn_heads
    dropout = drop_out

    # Load ONNX model
    onnx_model = onnx.load(model_path)
    ort_session = ort.InferenceSession(model_path)

    # Convert to torch tensor for processing
    def predict(data):
        ort_inputs = {
            ort_session.get_inputs()[0].name: data.x.cpu().numpy(),
            ort_session.get_inputs()[1].name: data.edge_index.cpu().numpy(),
        }
        ort_outs = ort_session.run(None, ort_inputs)
        return torch.tensor(ort_outs[0])

    # Split data
    split = T.RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
    data = split(data)

    test_loader = NeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=batchsize,
        input_nodes=data.test_mask,
    )

    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_data in test_loader:
            test_data.to(device)
            pred = predict(test_data)
            ground_truth = test_data.y
            predictions = (pred > 0.5).float()
            all_preds.extend(predictions.flatten().cpu().numpy())
            all_labels.extend(ground_truth.flatten().cpu().numpy())

            total_correct += (
                (predictions == ground_truth.unsqueeze(1)).sum().item()
            )
            total_samples += len(ground_truth)

    all_preds = [int(i) for i in all_preds]
    all_labels = [int(i) for i in all_labels]

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")

test()