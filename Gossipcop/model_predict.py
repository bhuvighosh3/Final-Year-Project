import os
import torch
import numpy as np
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from model import GNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define base directory for data and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
test_data = UPFD(root=BASE_DIR, name="gossipcop", feature="content", split="test")
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Load model
model = GNN(test_data.num_features, 128, 1).to(device)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "text_classification_model.pth"), map_location=device))
model.eval()

@torch.no_grad()
def predict():
    """Predicts labels for the entire test dataset."""
    all_preds = []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        all_preds.append(torch.round(out).cpu().numpy())

    return np.concatenate(all_preds).flatten()

@torch.no_grad()
def get_test_sample(index):
    """Fetches a single test sample, its prediction, actual label, and graph structure."""
    data = list(test_loader)[index].to(device)
    output = model(data.x, data.edge_index, data.batch)
    pred = torch.round(output).cpu().numpy().item()
    actual = data.y.cpu().numpy().item()
    edge_index = data.edge_index.cpu().numpy()
    return pred, actual, edge_index
