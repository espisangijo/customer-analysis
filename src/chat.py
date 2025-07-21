from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown


def start_chat_loop(recommender, agent, mode="link_prediction"):
    """
    Starts and manages the interactive chat session with the user using a rich UI.

    Args:
        recommender (RecommenderSystem): The initialized recommender system.
        agent (Agent): The initialized AI agent.
        mode (str): The recommendation mode to use ('link_prediction' or 'gat').
    """
    console = Console()

    console.print(
        f"\n--- Starting Interactive Chat Session (Mode: [bold green]{mode}[/bold green]) ---"
    )
    console.print("Type 'exit' to end the conversation.")

    conversation_history = ""
    conversation_context = []
    turn_counter = 0

    while True:
        customer_input = Prompt.ask("\n[bold green]You[/bold green]")

        if customer_input.lower() == "exit":
            goodbye_panel = Panel(
                "Thank you for chatting with us. Goodbye!",
                title="[bold red]Agent[/bold red]",
                title_align="left",
                border_style="cyan",
            )
            console.print(goodbye_panel)
            break

        recommendation_list = []
        cross_sell_suggestion = None

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

        recommendations = recommender.get_link_prediction_recommendations(
            conversation_context
        )

        recommendation_list = [item[0] for item in recommendations]

        negative_emotions = {
            "anxious",
            "worried",
            "frustrated",
            "confused",
            "overwhelmed",
        }
        if turn_counter > 2 and detected_emotion not in negative_emotions:
            current_products = [
                c for c in conversation_context if c in recommender._product_nodes
            ]
            cross_sell_suggestion = recommender.get_cross_sell_recommendation(
                current_products
            )

        analysis_text = (
            f"[bold]Detected Emotion:[/] {detected_emotion}\n"
            f"[bold]Found Concepts:[/] {found_concepts}\n"
            f"[bold]Updated Context:[/] {conversation_context}\n"
            f"[bold]Topic Recommendations:[/] {recommendation_list}"
        )
        if cross_sell_suggestion:
            analysis_text += (
                f"\n[bold]Cross-Sell Suggestion:[/] {cross_sell_suggestion}"
            )

        console.print(
            Panel(
                analysis_text,
                title="SYSTEM ANALYSIS",
                style="dim yellow",
                border_style="yellow",
            )
        )

        agent_reply = agent.get_reply(
            conversation_history,
            recommendation_list,
            customer_input,
            cross_sell_suggestion,
        )
        agent_panel = Panel(
            Markdown(agent_reply),
            title="[bold red]Agent[/bold red]",
            title_align="left",
            border_style="cyan",
        )
        console.print(agent_panel)

        conversation_history += f"Customer: {customer_input}\nAgent: {agent_reply}\n"
        turn_counter += 1
