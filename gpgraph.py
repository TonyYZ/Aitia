import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class EdgeClassifierGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super(EdgeClassifierGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_edge_features)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # Assuming edge_attr is 1D, we can use it directly. If it's more complex,
        # additional processing might be necessary
        return torch.sigmoid((x[edge_index[0]] * edge_attr * x[edge_index[1]]).sum(dim=1))

# Example Data (placeholder, replace with your actual data)
num_nodes = 10
num_node_features = 5
num_edge_features = 1
x = torch.rand((num_nodes, num_node_features))  # Node features
edge_index = torch.randint(0, num_nodes, (2, 20))  # Edges
edge_attr = torch.rand(20, num_edge_features)  # Edge features

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

model = EdgeClassifierGNN(num_node_features, num_edge_features)
out = model(data)

import torch
from torchviz import make_dot

# Assume num_node_features and num_edge_features are defined
model = EdgeClassifierGNN(num_node_features, num_edge_features)

# Create dummy input data
dummy_node_features = torch.rand(10, num_node_features)  # 10 nodes, each with 'num_node_features' features
dummy_edge_index = torch.randint(0, 10, (2, 20))  # 20 edges in a graph of 10 nodes
dummy_edge_attr = torch.rand(20, num_edge_features)  # Edge features

# Assuming the model expects a data object like those used in PyTorch Geometric
class DummyData:
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

dummy_data = DummyData(dummy_node_features, dummy_edge_index, dummy_edge_attr)

# Generate the graph
dot = make_dot(model(dummy_data), params=dict(model.named_parameters()))

# Add annotations to highlight edge features
dot.node('A', 'Edge Features Used Here', shape='box')
dot.edge('A', next(iter(dot.body))[0])  # Connect annotation to the first operation

dot.render('network_structure', format='png')  # Saves the diagram as 'network_structure.png'
