import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

num_bond_type = len(features['possible_bonds']) + 1
num_bond_direction = len(features['possible_bond_dirs']) + 1

class R2eGIN(MessagePassing):
    def __init__(self, in_dim, out_dim, eps=0, train_eps=False):
        super(R2eGIN, self).__init__(aggr='add') # Menggunakan agregasi penjumlahan

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 2 * out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * out_dim, out_dim)
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, out_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, out_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.tensor([eps]))
        self.train_eps = train_eps

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = len(features['possible_bonds'])
        self_loop_attr[:, 1] = len(features['possible_bond_dirs'])
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out, x):
        if self.train_eps:
            aggr_out = (1 + self.eps) * x + aggr_out
        else:
            aggr_out = x + aggr_out

        return self.mlp(aggr_out)

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = R2eGIN(7, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = R2eGIN(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = R2eGIN(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        # Reconstruction Block
        self.recon_fc1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.recon_fc2 = torch.nn.Linear(hidden_channels // 2, 7)

        # Prediction Module
        self.pred_fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Residual Connection 1
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        # Residual Connection 2
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        # Combine with Residual Connections
        x = x1 + x2 + x3

        # Reconstruction
        recon_x = global_mean_pool(x, batch)
        recon_x = F.relu(self.recon_fc1(recon_x))
        recon_x = self.recon_fc2(recon_x)

        pred_x = global_mean_pool(x, batch)
        pred_x = self.pred_fc(pred_x)
        pred_x = torch.sigmoid(pred_x).squeeze(1)

        return pred_x, recon_x