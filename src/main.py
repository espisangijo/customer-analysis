import argparse
import os
import pickle
import warnings

from dotenv import load_dotenv

from src.agent import Agent
from src.chat import start_chat_loop
from src.constants import (
    ASSETS_DIR,
    CONCEPT_GRAPH_FILE,
    CROSS_SELL_GRAPH_FILE,
    DATA_FILE,
    EMOTION_GRAPH_FILE,
    MODELS_DIR,
    NODE_MAPPING_FILE,
    GAT_MODEL_FILE,
)
from src.graph import GraphBuilder
from src.recommender import RecommenderSystem

warnings.filterwarnings("ignore")
load_dotenv()


def main():
    """
    Main entry point for the application.
    Handles argument parsing, initializes components, and starts the chat loop.
    """
    parser = argparse.ArgumentParser(description="Run the conversational AI agent.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["link", "gat"],
        default="link",
        help="Recommendation mode to use: 'link' for Jaccard link prediction (default) or 'gat' for GAT model.",
    )
    args = parser.parse_args()

    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    required_files = [
        CONCEPT_GRAPH_FILE,
        EMOTION_GRAPH_FILE,
        CROSS_SELL_GRAPH_FILE,
        NODE_MAPPING_FILE,
    ]
    if not all(os.path.exists(f) for f in required_files):
        print(
            "One or more graph/mapping files not found. Building graphs from source data..."
        )
        builder = GraphBuilder(DATA_FILE)
        builder.build_and_save_graphs(
            CONCEPT_GRAPH_FILE,
            EMOTION_GRAPH_FILE,
            CROSS_SELL_GRAPH_FILE,
            NODE_MAPPING_FILE,
        )

    try:
        with open(CONCEPT_GRAPH_FILE, "rb") as f:
            G = pickle.load(f)
        with open(EMOTION_GRAPH_FILE, "rb") as f:
            B = pickle.load(f)
        with open(CROSS_SELL_GRAPH_FILE, "rb") as f:
            X = pickle.load(f)
        with open(NODE_MAPPING_FILE, "rb") as f:
            node_mapping = pickle.load(f)
        print("All graphs and node mapping loaded successfully.")
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(
            f"Error loading graph files: {e}. Please delete them and rerun to rebuild."
        )
        return

    if args.mode == "gat":
        recommender = RecommenderSystem(G, B, X, GAT_MODEL_FILE, node_mapping)
    else:
        recommender = RecommenderSystem(G, B, X)

    agent = Agent()

    start_chat_loop(recommender, agent, mode=args.mode)


if __name__ == "__main__":
    main()
