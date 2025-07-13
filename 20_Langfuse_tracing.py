import os
from dotenv import load_dotenv

# --- OpenTelemetry and Langfuse Configuration ---
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


# Initialize dotenv to load environment variables
load_dotenv()

# --- Langfuse Environment Variables ---
# Set these in your .env file or directly in your environment
# Example .env entries:
# LANGFUSE_PUBLIC_KEY="pk-lf-YOUR_PUBLIC_KEY"
# LANGFUSE_SECRET_KEY="sk-lf-YOUR_SECRET_KEY"
# LANGFUSE_HOST="https://cloud.langfuse.com" # Or your self-hosted URL

langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") # Default to cloud if not set

if not langfuse_public_key or not langfuse_secret_key:
    raise ValueError(
        "Langfuse API keys (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY) are not set. "
        "Please ensure they are defined in your .env file or environment variables."
    )

# Set up OpenTelemetry Resource (identifies your service in tracing UI)
resource = Resource.create({"service.name": "gemini-agent-weather-app"})

# Set up TracerProvider
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Configure OTLP Exporter to send traces to Langfuse
# The OTLP ingestion endpoint for Langfuse Cloud is typically /api/public/ingestion/otlp
langfuse_otlp_endpoint = f"{langfuse_host}/api/public/ingestion/otlp"

otlp_exporter = OTLPSpanExporter(
    endpoint=langfuse_otlp_endpoint,
    headers={
        "x-langfuse-public-key": langfuse_public_key,
        "x-langfuse-secret-key": langfuse_secret_key,
    }
)

# Add a BatchSpanProcessor to send spans efficiently
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)

# --- Your original Agent Code ---
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, ModelSettings
from agents.run import RunConfig

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-pro", # Changed to 1.5-flash as 2.0-flash is less common
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    
)

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Assistant",
    instructions="You will be given a city name and you will return the weather in that city using get_weather tool.",
    model=model,
    tools=[get_weather],
    # model_settings = ModelSettings(max_tokens=19),
)

print("\nCALLING AGENT\n")
result = Runner.run_sync(agent, input("Enter the area: "), run_config=config)

print(result.final_output)

# --- Shutdown OpenTelemetry Provider ---
# This is crucial to ensure all buffered traces are sent before the program exits.
trace.get_tracer_provider().shutdown()