from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class Agent:
    """
    Handles interaction with the LangChain LLM to generate agent replies.
    """

    def __init__(self):
        self.chain = None
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.8,
                max_output_tokens=150,
                convert_system_message_to_human=True,
            )

            prompt_template = """You are a helpful and empathetic bank customer service agent for STAT8306 bank.
            Your goal is to understand the customer's needs and guide them to a solution.
            
            Here is some internal system analysis to help you:
            - Recommended Topics to discuss next: {recommendations}
            - Potential Cross-Sell Opportunity: {cross_sell_suggestion}

            Your primary goal is to address the customer's main query.
            If the conversation is going well and the customer seems satisfied, you can gently introduce the cross-sell opportunity if it feels natural.
            Do not be pushy. If there is no cross-sell opportunity, do not mention it.

            Here is the recent conversation history:
            {history}
            
            Customer: {customer_input}
            Agent:"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[
                    "history",
                    "recommendations",
                    "customer_input",
                    "cross_sell_suggestion",
                ],
            )
            self.chain = prompt | model
            print("LangChain Agent is ready.")
        except Exception as e:
            print(f"Could not initialize LangChain Agent. Error: {e}")
            self.chain = None

    def get_reply(
        self, history, recommendations, customer_input, cross_sell_suggestion=None
    ):
        if self.chain:
            try:
                response = self.chain.invoke(
                    {
                        "history": history,
                        "recommendations": ", ".join(recommendations) or "None",
                        "customer_input": customer_input,
                        "cross_sell_suggestion": cross_sell_suggestion or "None",
                    }
                )
                if response.content:
                    return response.content
                else:
                    print("Warning: Agent response was empty. Falling back to default.")
                    return "I'm sorry, I'm having trouble formulating a response. Could you rephrase that?"
            except Exception as e:
                print(f"Error during agent invocation: {e}")
                return "I'm sorry, I'm experiencing a technical issue. Please give me a moment."
        else:
            return "I see. Could you please tell me more?"

