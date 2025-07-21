import os
import pickle
import warnings

from dotenv import load_dotenv

from src.agent import Agent
from src.graph import GraphBuilder
from src.recommender import RecommenderSystem

warnings.filterwarnings("ignore")
load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    os.path.dirname(ROOT_DIR), "data", "processed", "conversations-1750493143.csv"
)
CONCEPT_GRAPH_FILE = "holistic_concept_graph.gpickle"
EMOTION_GRAPH_FILE = "emotion_word_graph.gpickle"
CROSS_SELL_GRAPH_FILE = "cross_sell_graph.gpickle"  # New graph file


def main():
    """
    Main entry point for the application.
    Initializes components and runs the chat loop.
    """
    if not all(
        os.path.exists(f)
        for f in [CONCEPT_GRAPH_FILE, EMOTION_GRAPH_FILE, CROSS_SELL_GRAPH_FILE]
    ):
        print("One or more graph files not found. Building graphs from source data...")
        builder = GraphBuilder(DATA_FILE)
        builder.build_and_save_graphs(
            CONCEPT_GRAPH_FILE, EMOTION_GRAPH_FILE, CROSS_SELL_GRAPH_FILE
        )

    try:
        with open(CONCEPT_GRAPH_FILE, "rb") as f:
            G = pickle.load(f)
        with open(EMOTION_GRAPH_FILE, "rb") as f:
            B = pickle.load(f)
        with open(CROSS_SELL_GRAPH_FILE, "rb") as f:
            X = pickle.load(f)
        print("All graphs loaded successfully.")
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(
            f"Error loading graph files: {e}. Please delete them and rerun to rebuild."
        )
        return

    recommender = RecommenderSystem(G, B, X)
    agent = Agent()
    emotions = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}

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

        recommendation_list = []
        cross_sell_suggestion = None

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

            # uncomment this to avoid cross selling when user has negative emotion
            # negative_emotions = {
            #     "anxious",
            #     "worried",
            #     "frustrated",
            #     "confused",
            #     "overwhelmed",
            # }
            # if turn_counter > 2 and detected_emotion not in negative_emotions:
            if turn_counter > 2:
                current_products = [
                    concept
                    for concept in conversation_context
                    if not concept.startswith("pref:") and concept not in emotions
                ]
                cross_sell_suggestion = recommender.get_cross_sell_recommendation(
                    current_products
                )

        else:
            detected_emotion = "N/A"
            found_concepts = []

        print("\n--- SYSTEM ANALYSIS ---")
        print(f"Detected Emotion: {detected_emotion}")
        print(f"Found Concepts: {found_concepts}")
        print(f"Updated Conversation Context: {conversation_context}")
        print(f"Topic Recommendations: {recommendation_list}")
        if cross_sell_suggestion:
            print(f"Cross-Sell Suggestion: {cross_sell_suggestion}")
        print("-----------------------\n")

        # --- Generate and print agent reply ---
        # Note: You will need to update the Agent class to handle the new cross_sell_suggestion parameter
        agent_reply = agent.get_reply(
            conversation_history,
            recommendation_list,
            customer_input,
            cross_sell_suggestion,
        )
        print(f"Agent: {agent_reply}")

        conversation_history += f"Customer: {customer_input}\nAgent: {agent_reply}\n"
        turn_counter += 1


if __name__ == "__main__":
    main()
