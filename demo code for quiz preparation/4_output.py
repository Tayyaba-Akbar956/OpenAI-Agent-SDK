import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings
from agents.run import RunConfig
import asyncio
from dataclasses import dataclass
from agents import RunContextWrapper, function_tool
from pydantic import BaseModel

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

class test_result(BaseModel):
  
    remarks: str
    message : str



    
agent1 =Agent (name ="Test Agent",
                instructions="score of the test will be given to you and you generate response based on the score and remarks",
                model=model,
                tools=[],
                output_type=test_result)

result = Runner.run_sync(
    agent1,
    input=input ("Enter your score: "),
    run_config=config,
    
)

print (result.final_output)