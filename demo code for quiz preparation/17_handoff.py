import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings,handoff
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


billing_agent = Agent(name="Billing agent",model = model)
refund_agent = Agent(name="Refund agent",model = model)


triage_agent = Agent(
    name="Triage agent", 
    handoffs=[billing_agent,
        handoff(
            refund_agent,
            name = "refund",
            instructions = "Handle all refund-related queries",
           )],

    model = model)

result = Runner.run_sync(
    triage_agent,
    "I want to cancel my subscription and get a refund.",
  
)

print(result.last_agent.name)