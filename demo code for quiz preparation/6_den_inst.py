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
    model="gemini-2.0-pro",
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
    return f"The weather in {city} is sunny."

# ✅ Dynamic instructions based on metadata
def get_instructions() -> str:
    user_role = context.metadata.get("user_role", "guest")

    if user_role == "student":
        return "You're a weather tutor. Explain how to use the get_weather tool step by step."
    elif user_role == "developer":
        return "You're a code-savvy assistant. Use the get_weather tool and explain how it works."
    else:
        return "You're a friendly assistant. Use the get_weather tool to help the user."

# ✅ Create the agent
agent = Agent(
    name="GeminiAssistant",
    model=model,
    tools=[get_weather],
    instructions=get_instructions,  # <== DYNAMIC
)

# ✅ Choose user role dynamically
user_role = input("Enter user role (student/developer/guest): ").strip().lower()
user_input = input("Ask something like 'What is the weather in Lahore?': ")

# ✅ Run agent with context metadata
result = Runner.run_sync(
    agent,
    input=user_input,
    context=Context(metadata={"user_role": user_role}),
    config=config,
)

# ✅ Output
print("\n🔁 Final Output:")
print(result.final_output)
