import os
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings
from agents.run import RunConfig
import asyncio
from agents.tracing import trace

# --- AgentOps Import ---
import agentops

# Load the environment variables from the .env file
load_dotenv()

# --- AgentOps Initialization ---
# Get your AgentOps API key from your .env file
agentops_api_key = os.getenv("AgentOps_API_KEY")

if not agentops_api_key:
    raise ValueError(
        "AGENTOPS_API_KEY is not set. Please ensure it is defined in your .env file."
    )

# Initialize AgentOps. Do this BEFORE any LLM calls or agent runs.
agentops.init(api_key=agentops_api_key)
# You can also set it as an environment variable and call agentops.init() without arguments:
# os.environ["AGENTOPS_API_KEY"] = "YOUR_API_KEY"
# agentops.init()


gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash", # Adjusted model name if "gemini-2.0-flash" causes issues
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

# agent = Agent(
#     name="Assistant",
#     instructions="You will be given a city name and you will return the weather in that city using get_weather tool.",
#     model=model,
#     tools=[get_weather],
#     # model_settings = ModelSettings(max_tokens=19),
# )

# print("\nCALLING AGENT\n")
# user_input = input("Enter the area: ")
# result = Runner.run_sync(agent, user_input, run_config=config)

# print(result.final_output)

# # --- End the AgentOps session ---
# # This is important to ensure all data is sent and the session is marked as complete.
# # You can pass "Success", "Fail", or "Indeterminate" as the end state.
# agentops.end_session("Success")

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.",model = model)

    with trace(workflow_name="Conversation", group_id="thread_id_13"):
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

agentops.end_session("Success")