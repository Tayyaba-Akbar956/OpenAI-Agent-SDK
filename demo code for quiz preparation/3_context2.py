import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings
from agents.run import RunConfig
import asyncio
from dataclasses import dataclass
from agents import RunContextWrapper, function_tool

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

dict1 = {"user1":{"uid":123,"name": "John", "age": 13},
         "user2":{"uid":456,"name": "Alice", "age": 47},
         "user3":{"uid":789,"name": "Bob", "age": 30},
         "user4":{"uid":101,"name": "Charlie", "age": 25},
         "user5":{"uid":102,"name": "David", "age": 35}}

@function_tool
async def print_user_details(wrapper: RunContextWrapper[dict]) -> str:
    """Print the details of the users."""
    user_details = []
    for user in wrapper.context.values():
        user_details.append(f"User {user['uid']}: Name: {user['name']}, Age: {user['age']}")
    return "\n".join(user_details)

agent = Agent[dict](
    name="Assistant",
    tools=[print_user_details],
    model=model,
)

result = Runner.run_sync(
    agent,input ="print the details of the users",context = dict1,run_config=config)

print (result.final_output)
 


#
