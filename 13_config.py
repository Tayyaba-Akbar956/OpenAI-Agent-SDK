import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings
from agents.run import RunConfig
import asyncio
from agents.tracing import trace
# Set the thread ID for tracing
thread_id = "13_config"




# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)





model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)
async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.",model = model)

    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn

        result = await Runner.run(agent, "Which country is most populated?")
        print(result.final_output)
        # San Francisco

        # Second turn
        new_input = result.to_input_list() + [{"role": "user", "content": "What is the capital of that country?"}]
        result = await Runner.run(agent, new_input)
        print(result.final_output)
        # California

if __name__ == "__main__":

    asyncio.run(main())
# Set the thread ID for tracing

