import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool as gmp

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)

        self.lin_news = nn.Linear(in_channels, hidden_channels)
        self.lin0 = nn.Linear(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()
        h = gmp(h, batch)

        h = self.lin0(h).relu()

        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        news = x[root]
        news = self.lin_news(news).relu()

        out = self.lin1(torch.cat([h, news], dim=-1))
        return torch.sigmoid(out)

# Load the model
def load_model(path, in_channels):
    model = GNN(in_channels, 128, 1)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model