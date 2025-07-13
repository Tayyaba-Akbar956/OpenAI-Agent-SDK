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



@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

agent1 = Agent (
    name = "Assistant", 
    instructions = "You will be given a city name and you will return the weather in that city using get_weather tool.",
    model = model,
    tools = [get_weather],
    model_settings = ModelSettings(tool_choice="required"),
)

result = Runner.run_sync(agent1, input ("enter the area : "), )

print("\nCALLING AGENT\n")
# print (result)
print(result.new_items.ToolCallItem)
print (result.raw_responses)

from agents import Agent, Runner, RunResult
from agents.run_item import (
    MessageOutputItem,
    HandoffCallItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
)

async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant",model = model,tools=[get_weather], model_settings=ModelSettings(tool_choice="required"))
    result: RunResult = await Runner.run(agent, "Some input that triggers different items")

    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            # Access the message content
            print(f"Message from LLM: {item.raw_item}")

        elif isinstance(item, HandoffCallItem):
            # Access the tool call for handoff
            print(f"Handoff called: {item.raw_item}")

        elif isinstance(item, HandoffOutputItem):
            # Access the response from the handoff tool call
            # and the source and target agents
            print(f"Handoff occurred from {item.source_agent} to {item.target_agent}")
            print(f"Handoff tool response: {item.raw_item}")

        elif isinstance(item, ToolCallItem):
            # Access the tool call from the LLM
            print(f"Tool called: {item.raw_item}")

        elif isinstance(item, ToolCallOutputItem):
            # Access the output from the tool
            print(f"Tool output: {item.tool_output}")
            print(f"Tool response: {item.raw_item}")

        elif isinstance(item, ReasoningItem):
            # Access the reasoning from the LLM
            print(f"Reasoning: {item.raw_item}")