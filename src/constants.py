import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_FILE = os.path.join(ROOT_DIR, "data", "processed", "conversations-1750493143.csv")
CONCEPT_GRAPH_FILE = os.path.join(ASSETS_DIR, "holistic_concept_graph.gpickle")
EMOTION_GRAPH_FILE = os.path.join(ASSETS_DIR, "emotion_word_graph.gpickle")
CROSS_SELL_GRAPH_FILE = os.path.join(ASSETS_DIR, "cross_sell_graph.gpickle")

GAT_MODEL_FILE = os.path.join(MODELS_DIR, "gat_model.pt")
NODE_MAPPING_FILE = os.path.join(ASSETS_DIR, "node_mapping.pkl")
