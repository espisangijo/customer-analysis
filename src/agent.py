from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)


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
            
            Here are some internal topics your system recommends you discuss next, based on the conversation so far.
            Try to subtly weave one of these topics into your response if it feels natural.
            Recommended Topics: {recommendations}
            
            Here is the recent conversation history:
            {history}
            
            Customer: {customer_input}
            Agent:"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["history", "recommendations", "customer_input"],
            )
            self.chain = prompt | model
            print("LangChain Agent is ready.")
        except Exception as e:
            print(f"Could not initialize LangChain Agent. Error: {e}")
            self.chain = None

    def get_reply(self, history, recommendations, customer_input):
        if self.chain:
            try:
                response = self.chain.invoke(
                    {
                        "history": history,
                        "recommendations": ", ".join(recommendations) or "None",
                        "customer_input": customer_input,
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
