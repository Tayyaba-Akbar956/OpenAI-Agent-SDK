import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,ModelSettings,RunContextWrapper,handoff
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


# --- 2. Define the Handoff Callback and Destination Agent ---

# This function will be called when the handoff occurs
def on_handoff_callback(ctx: RunContextWrapper[None]):
    """A function that is triggered when a handoff is executed."""
    print("\n--- Handoff callback triggered! ---\n")

# This is the agent that will receive the task
destination_agent = Agent(
    name="My agent",
    instructions="You are a helpful agent that tells jokes.",
    model=model,
)

# --- 3. Create the Configured Handoff Object ---

# This creates the special one-way handoff with your custom settings
handoff_obj = handoff(
    agent=destination_agent,
    on_handoff=on_handoff_callback,
    tool_name_override="delegate_to_joke_agent",
    tool_description_override="Use this tool to delegate a joke-telling task to 'My agent'.",
)

# --- 4. Create the Source Agent ---

# This agent's job is to receive a task and delegate it.
# Notice its `handoffs` list contains the `handoff_obj`.
source_agent = Agent(
    name="Triage agent",
    instructions=(
        "You are a router. If the user asks for a joke, you must use the "
        "'delegate_to_joke_agent' tool to pass the request to 'My agent'."
    ),
    model=model,
    handoffs=[handoff_obj],
)

# --- 5. Run the Interaction ---

# The user's prompt will trigger the handoff logic
user_prompt = "I need you to tell me a joke about computers."

print(f"User Prompt: {user_prompt}\n")

# Start the process with the source_agent
result = Runner.run_sync(
    source_agent,
    input=user_prompt,
)

print(f"Final Output:\n{result.final_output}")
print (f"Last Agent: {result.last_agent.name}\n")


