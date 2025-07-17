import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
from typing import Optional

# Load the environment variables from the .env file
load_dotenv()

# ✅ Load Gemini API Key from .env
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# ✅ Configure Gemini-compatible OpenAI client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ✅ Gemini wrapped as OpenAI model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# ✅ Runner config
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# ✅ A tool to be used by the agent
@function_tool
def get_weather(city: str) -> str:
    """
    Retrieves the current weather for a specified city.
    Args:
        city (str): The name of the city.
    Returns:
        str: A string indicating the weather in the city.
    """
    return f"The weather in {city} is sunny."

# ✅ Choose user role dynamically
user_role = input("Enter user role (student/developer/guest): ").strip().lower()
user_input = input("Ask something like 'What is the weather in Lahore?': ")

# ✅ Determine instructions based on user role before creating the agent
instructions_text: str
if user_role == "student":
    instructions_text = "You're a weather tutor. Explain how to use the get_weather tool step by step."
elif user_role == "developer":
    instructions_text = "You're a code-savvy assistant. Use the get_weather tool and explain how it works."
else:
    instructions_text = "You're a friendly assistant. Use the get_weather tool to help the user."

# ✅ Create the agent with the pre-determined instructions
agent = Agent(
    name="GeminiAssistant",
    model=model,
    tools=[get_weather],
    instructions=instructions_text,  # <== NOW A STATIC STRING
)

# ✅ Run agent without context metadata
result = Runner.run_sync(
    agent,
    input=user_input,
    run_config=config,
)

# ✅ Output
print("\n🔁 Final Output:")
print(result.final_output)
