import os
import sys
import pickle
import torch
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score

from src.constants import CONCEPT_GRAPH_FILE, GAT_MODEL_FILE, NODE_MAPPING_FILE

EPOCHS = 1000
LEARNING_RATE = 0.01


# --- 1. Data Loading and Preparation ---
def load_and_prepare_data(graph_path, mapping_path):
    """
    Loads the NetworkX graph and the node mapping, then converts the graph
    into a PyTorch Geometric Data object.
    """
    if not os.path.exists(graph_path) or not os.path.exists(mapping_path):
        print(
            f"Error: Graph file '{graph_path}' or mapping file '{mapping_path}' not found."
        )
        print("Please run the main script first to build the graphs and mapping.")
        return None, None

    try:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        with open(mapping_path, "rb") as f:
            node_mapping = pickle.load(f)
        print(f"Successfully loaded '{graph_path}' and '{mapping_path}'.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

    # Relabel the graph nodes using the pre-defined mapping
    G = nx.relabel_nodes(G, node_mapping)

    # Use one-hot encoded vectors as initial node features
    x = F.one_hot(
        torch.arange(len(node_mapping)), num_classes=len(node_mapping)
    ).float()

    # Get the edge list in the format PyG requires
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index)

    # Split edges for training, validation, and testing
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)

    print("\nData prepared for GAT training:")
    print(data)

    return data, node_mapping


class GATLinkPredictor(torch.nn.Module):
    """
    A Graph Attention Network model for link prediction.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1)

    def encode(self, x, edge_index):
        """
        Generates node embeddings.
        """
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """
        Given node embeddings, predicts the probability of an edge.
        We use a simple dot product to score the likelihood of a link.
        """
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        """
        Predicts probabilities for all possible node pairs.
        """
        prob_adj = z @ z.t()
        return prob_adj


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.train_pos_edge_index)

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
        method="sparse",
    )

    edge_label_index = torch.cat(
        [data.train_pos_edge_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat(
        [
            torch.ones(data.train_pos_edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )

    out = model.decode(z, edge_label_index).view(-1)

    loss = F.binary_cross_entropy_with_logits(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)

    results = {}
    for prefix in ["val", "test"]:
        pos_edge_index = data[f"{prefix}_pos_edge_index"]
        neg_edge_index = data[f"{prefix}_neg_edge_index"]

        pos_pred = model.decode(z, pos_edge_index).sigmoid()
        neg_pred = model.decode(z, neg_edge_index).sigmoid()

        pred = torch.cat([pos_pred, neg_pred])
        true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        auc = roc_auc_score(true.cpu().numpy(), pred.cpu().numpy())
        results[f"{prefix}_auc"] = auc

    return results


if __name__ == "__main__":
    data, node_mapping = load_and_prepare_data(CONCEPT_GRAPH_FILE, NODE_MAPPING_FILE)  # type: ignore

    if data is None:
        print("Exiting training due to data loading failure.")
        sys.exit(1)

    model = GATLinkPredictor(data.num_features, 128, 64)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting GAT Model Training ---")
    best_val_auc = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, optimizer)
        results = test(model, data)
        val_auc, test_auc = results["val_auc"], results["test_auc"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), GAT_MODEL_FILE)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}"
            )

    print("\n--- Training Complete ---")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Trained model saved to '{GAT_MODEL_FILE}'")
