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
    tracing_disabled=True,
) 

agent1 = Agent (
    name = "Pirate",
    instructions = "write like a pirate",
    model = model,
)

agent2 = agent1.clone (
    name = "Robot",
    instructions = "write like a robot"
)
agent3 = agent2.clone (
    name = "poet",
    instructions= "write like a poet"
)
print ("Pirate")
result1 = Runner.run_sync (agent1,"what is poetry")

print (result1.final_output)

print ("robot")
result2 = Runner.run_sync (agent2,"what is poetry")
print (result2.final_output)

print ("Poet")
result3 = Runner.run_sync (agent3,"what is poetry")
print (result3.final_output)
