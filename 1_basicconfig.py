import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings
from agents.run import RunConfig

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



@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

agent = Agent (
    name = "Assistant", 
    instructions = "You will be given a city name and you will return the weather in that city using get_weather tool.",
    model = model,
    tools = [get_weather],
    # model_settings = ModelSettings(max_tokens=19),
)

result = Runner.run_sync(agent, input ("enter the area : "),run_config= config) 

print("\nCALLING AGENT\n")
print(result.final_output)


