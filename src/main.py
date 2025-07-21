import os
import pickle

from src.agent import Agent
from src.graph import GraphBuilder
from src.recommender import RecommenderSystem
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT_DIR, "data/processed/conversations-1750493143.csv")
CONCEPT_GRAPH_FILE = "holistic_concept_graph.gpickle"
EMOTION_GRAPH_FILE = "emotion_word_graph.gpickle"


def main():
    """
    Main entry point for the application.
    Initializes components and runs the chat loop.
    """
    if not os.path.exists(CONCEPT_GRAPH_FILE) or not os.path.exists(EMOTION_GRAPH_FILE):
        print("One or more graph files not found. Building graphs from source data...")
        builder = GraphBuilder(DATA_FILE)
        builder.build_and_save_graphs(CONCEPT_GRAPH_FILE, EMOTION_GRAPH_FILE)

    try:
        with open(CONCEPT_GRAPH_FILE, "rb") as f:
            G = pickle.load(f)
        with open(EMOTION_GRAPH_FILE, "rb") as f:
            B = pickle.load(f)
        print("Graphs loaded successfully.")
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(
            f"Error loading graph files: {e}. Please delete them and rerun to rebuild."
        )
        return

    recommender = RecommenderSystem(G, B)
    agent = Agent()

    print("\n--- Starting Interactive Chat Session ---")
    print("Type 'exit' to end the conversation.")

    conversation_history = ""
    conversation_context = []
    turn_counter = 0

    while True:
        customer_input = input("\nCustomer: ")
        if customer_input.lower() == "exit":
            print("Agent: Thank you for chatting with us. Goodbye!")
            break

        # can be tuned to reduce overassumption
        if turn_counter > 0:
            detected_emotion = recommender.detect_emotion_from_text(customer_input)
            found_concepts = recommender.extract_concepts_from_text(customer_input)

            if (
                detected_emotion != "neutral"
                and detected_emotion not in conversation_context
            ):
                conversation_context.append(detected_emotion)
            for concept in found_concepts:
                if concept not in conversation_context:
                    conversation_context.append(concept)

            recommendations = recommender.get_recommendations(conversation_context)
            recommendation_list = [item[0] for item in recommendations]
        else:
            detected_emotion = "N/A"
            found_concepts = []
            recommendation_list = []

        print("\n--- SYSTEM ANALYSIS ---")
        print(f"Detected Emotion: {detected_emotion}")
        print(f"Found Concepts: {found_concepts}")
        print(f"Updated Conversation Context: {conversation_context}")
        print(f"Top Recommendations: {recommendation_list}")
        print("-----------------------\n")

        agent_reply = agent.get_reply(
            conversation_history, recommendation_list, customer_input
        )
        print(f"Agent: {agent_reply}")

        conversation_history += f"Customer: {customer_input}\nAgent: {agent_reply}\n"
        turn_counter += 1


if __name__ == "__main__":
    main()
