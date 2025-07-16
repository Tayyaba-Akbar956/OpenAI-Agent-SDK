from agents import Agent, Runner, RunResult
from agents import (
    MessageOutputItem,
    HandoffCallItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
)
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


async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant",model = model,tools=[get_weather] )
    result: RunResult = await Runner.run(agent, "Some input that triggers different items", )
    # print (result.last_agent.name)
    # print (result.last_agent.instructions)
    # print (result.last_agent.model)
    # print (result.input_guardrail_results)
    print (result.input)
    print (result.raw_responses)
    # for item in result.new_items:
    #     if isinstance(item, MessageOutputItem):
    #         # Access the message content
    #         print(f"Message from LLM: {item.raw_item}")

    #     elif isinstance(item, HandoffCallItem):
    #         # Access the tool call for handoff
    #         print(f"Handoff called: {item.raw_item}")

    #     elif isinstance(item, HandoffOutputItem):
    #         # Access the response from the handoff tool call
    #         # and the source and target agents
    #         print(f"Handoff occurred from {item.source_agent} to {item.target_agent}")
    #         print(f"Handoff tool response: {item.raw_item}")

    #     elif isinstance(item, ToolCallItem):
    #         # Access the tool call from the LLM
    #         print(f"Tool called: {item.raw_item}")

    #     elif isinstance(item, ToolCallOutputItem):
    #         # Access the output from the tool
    #         print(f"Tool output: {item.tool_output}")
    #         print(f"Tool response: {item.raw_item}")

    #     elif isinstance(item, ReasoningItem):
    #         # Access the reasoning from the LLM
    #         print(f"Reasoning: {item.raw_item}")

# Run the main function
import asyncio
asyncio.run(main())