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
# Initialize your agent
@dataclass
class UserInfo:  
    name: str
    uid: int

@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[any]) -> str:  
    """Fetch the age of the user. Call this function to get user's age information."""
    return f"The user {wrapper.context.name} is 47 years old"

async def main():
    user_info_1 = UserInfo(name="John", uid=123)
    user_info_2 = UserInfo(name="Alice", uid=456)
    
    agent = Agent(  
        name="Assistant",
        tools=[fetch_user_age],
        model=model,
    )
    result = await Runner.run(  
        starting_agent=agent,
        input="What is the age of the user",
        context=user_info_2,
        run_config=config
    )

    print(result.final_output)  
    # The user John is 47 years old.

if __name__ == "__main__":
    asyncio.run(main())

