import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, SAGEConv, global_mean_pool
)
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, use_conv3=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.use_conv3 = use_conv3  # 🔹 flag

        if self.use_conv3:
            self.conv3 = GCNConv(hidden_channels, out_channels)
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        activations = {}

        x = self.conv1(x, edge_index); activations['conv1'] = x.detach().clone()
        x = F.relu(x); activations['relu1'] = x.detach().clone()

        x = self.conv2(x, edge_index); activations['conv2'] = x.detach().clone()
        # x = F.relu(x); activations['relu2'] = x.detach().clone()

        if self.use_conv3:
            x = self.conv3(x, edge_index); activations['conv3'] = x.detach().clone()

        x = global_mean_pool(x, batch); activations['global_pool'] = x.detach().clone()
        x = self.fc(x); activations['fc'] = x.detach().clone()
        return x, activations


# ==============================
# 🔹 GIN
# ==============================
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, use_conv3=True):
        super(GIN, self).__init__()
        self.use_conv3 = use_conv3

        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)

        if self.use_conv3:
            nn3 = nn.Sequential(nn.Linear(hidden_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
            self.conv3 = GINConv(nn3)
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        acts = {}
        x = self.conv1(x, edge_index); acts['conv1'] = x.detach().clone()
        x = F.relu(x); acts['relu1'] = x.detach().clone()

        x = self.conv2(x, edge_index); acts['conv2'] = x.detach().clone()
        x = F.relu(x); acts['relu2'] = x.detach().clone()

        if self.use_conv3:
            x = self.conv3(x, edge_index); acts['conv3'] = x.detach().clone()

        x = global_mean_pool(x, batch); acts['global_pool'] = x.detach().clone()
        x = self.fc(x); acts['fc'] = x.detach().clone()
        return x, acts


# ==============================
# 🔹 GAT
# ==============================
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, heads=4, use_conv3=True):
        super(GAT, self).__init__()
        self.use_conv3 = use_conv3

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True)

        if self.use_conv3:
            self.conv3 = GATConv(hidden_channels, out_channels, heads=1, concat=True)
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        acts = {}
        x = self.conv1(x, edge_index); acts['conv1'] = x.detach().clone()
        x = F.elu(x); acts['relu1'] = x.detach().clone()

        x = self.conv2(x, edge_index); acts['conv2'] = x.detach().clone()
        #x = F.elu(x); acts['relu2'] = x.detach().clone()

        if self.use_conv3:
            x = self.conv3(x, edge_index); acts['conv3'] = x.detach().clone()

        x = global_mean_pool(x, batch); acts['global_pool'] = x.detach().clone()
        x = self.fc(x); acts['fc'] = x.detach().clone()
        return x, acts


# ==============================
# 🔹 GraphSAGE
# ==============================
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, use_conv3=True):
        super(GraphSAGE, self).__init__()
        self.use_conv3 = use_conv3

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        if self.use_conv3:
            self.conv3 = SAGEConv(hidden_channels, out_channels)
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        acts = {}
        x = self.conv1(x, edge_index); acts['conv1'] = x.detach().clone()
        x = F.relu(x); acts['relu1'] = x.detach().clone()

        x = self.conv2(x, edge_index); acts['conv2'] = x.detach().clone()
        #x = F.relu(x); acts['relu2'] = x.detach().clone()

        if self.use_conv3:
            x = self.conv3(x, edge_index); acts['conv3'] = x.detach().clone()

        x = global_mean_pool(x, batch); acts['global_pool'] = x.detach().clone()
        x = self.fc(x); acts['fc'] = x.detach().clone()
        return x, acts
