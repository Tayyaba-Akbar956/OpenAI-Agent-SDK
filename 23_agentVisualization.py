import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
from agents.extensions.visualization import draw_graph # Import draw_graph

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Use a compatible model for Gemini
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash", # Changed to 1.5-flash for broader availability
    openai_client=external_client,
)

# --- Define the shared RunConfig ---
# While not strictly needed for draw_graph itself, agents would use this if run
shared_config = RunConfig(
    model=model,
    model_provider=external_client,
)

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

# --- Define the specialized agents ---
# These agents are defined with a model so they *could* be run,
# but for visualization, only their definitions and handoffs matter.
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model=model, # Assign the model
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model=model, # Assign the model
)

# --- Define the Triage agent with handoffs ---
# This agent orchestrates the handoffs based on language.
triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent], # Define the agents it can handoff to
    tools=[get_weather], # It can also use tools itself before handing off
    model=model, # Assign the model
)

# --- Visualization Part ---
print("Generating agent graph...")
# draw_graph returns a graphviz.Digraph object
graph = draw_graph(triage_agent).view()


graph.render(filename="triage_agent_graph", format="png", view=False)

print("Agent graph saved as 'triage_agent_graph.png'")

# # --- The original agent run (optional to keep, but distinct from graph drawing) ---
# # If you still want to run your original "Assistant" agent:
# print("\n--- Running the Assistant Agent ---")
# assistant_agent = Agent (
#     name = "Assistant",
#     instructions = "You will be given a city name and you will return the weather in that city using get_weather tool.",
#     model = model,
#     tools = [get_weather],
# )

# # Note: Using run_sync here. If you need async behavior with the visualization server
# # (from a previous conversation) you'd need to use asyncio.
# user_input_for_assistant = input("Enter the area for the Assistant (e.g., London): ")
# result_assistant = Runner.run_sync(assistant_agent, user_input_for_assistant, run_config=shared_config)

# print("\nCALLING ASSISTANT AGENT\n")
# print(result_assistant.final_output)