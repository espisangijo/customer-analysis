import os
import pickle
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score

from src.constants import CONCEPT_GRAPH_FILE, NODE_MAPPING_FILE


def evaluate_jaccard_auc(graph_path, mapping_path):
    """
    Loads the graph, splits it into train/test sets, and evaluates the
    Jaccard Coefficient link prediction method using the AUC metric.
    """
    if not os.path.exists(graph_path) or not os.path.exists(mapping_path):
        print("Error: Graph or mapping file not found.")
        print("Please run the main script first to build the graphs and mapping.")
        return

    try:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        with open(mapping_path, "rb") as f:
            node_mapping = pickle.load(f)
        print(f"Successfully loaded '{graph_path}' and '{mapping_path}'.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    G = nx.relabel_nodes(G, node_mapping)

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    data = Data(edge_index=edge_index, num_nodes=G.number_of_nodes())

    data_split = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)

    print("\nData split for evaluation.")

    train_graph = nx.Graph()
    train_edges = data_split.train_pos_edge_index.t().cpu().numpy()
    train_graph.add_nodes_from(range(data_split.num_nodes))
    train_graph.add_edges_from(train_edges)

    print("Calculating Jaccard scores for the test set...")

    pos_test_edges = data_split.test_pos_edge_index.t().cpu().numpy()
    neg_test_edges = data_split.test_neg_edge_index.t().cpu().numpy()

    pos_preds = [
        score for _, _, score in nx.jaccard_coefficient(train_graph, pos_test_edges)
    ]
    neg_preds = [
        score for _, _, score in nx.jaccard_coefficient(train_graph, neg_test_edges)
    ]

    predictions = pos_preds + neg_preds
    true_labels = [1] * len(pos_preds) + [0] * len(neg_preds)

    auc_score = roc_auc_score(true_labels, predictions)

    print("\n--- Jaccard Coefficient Evaluation ---")
    print(f"Test AUC Score: {auc_score:.4f}")

    return auc_score


if __name__ == "__main__":
    evaluate_jaccard_auc(CONCEPT_GRAPH_FILE, NODE_MAPPING_FILE)
