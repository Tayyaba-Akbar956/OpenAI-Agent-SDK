import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings,handoff, RunContextWrapper
from agents.run import RunConfig
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
    tracing_disabled=True,
) 


class EscalationData(BaseModel):
    reason: str

async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent called with reason: {input_data.reason}")

agent = Agent(name="Escalation agent",model =model)

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    input_type=EscalationData,
)

source_agent = Agent(
    name="Source agent",
    instructions="You will be given a task and you can delegate it to the escalation agent.",
    model=model,
    handoffs=[handoff_obj],
)
result = Runner.run_sync(
    source_agent,
    "any propt that need to be delegated to the escalation agent for further processing."
)

print("\nCALLING AGENT\n")
print(result.final_output)
print("\nHandoff callback executed successfully.\n")
print(f"Last agent called: {result.last_agent.name}")
