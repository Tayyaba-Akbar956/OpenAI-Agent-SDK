import os
from dotenv import load_dotenv
from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, ModelSettings
from agents.run import RunConfig
import asyncio # Although not directly used in the sync run, good to have if you plan async.

# Load environment variables from .env file
load_dotenv()

# Ensure API key is present
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_settings=ModelSettings(
        temperature=1.0,
        top_p=1.0
    ),
    tracing_disabled=True
)    

name_agent = Agent(
    name="Name Agent",
    instructions="""You are a company name generator. Based on the product name and description,
    you generate creative and suitable company names.
    Provide only the name, without additional commentary.""",
    model=model
)

slogan_agent = Agent(
    name="Slogan Generator",
    instructions="""You are a slogan generator. Based on the product name, description, and
    company name, you generate highly creative and catchy slogans.
    Provide only the slogan, without additional commentary.""",
    model=model
)

main_agent = Agent(
    name="Triage Agent",
    instructions="""You are a Triage Agent. Your primary role is to direct user requests to the appropriate
    tool. You should only respond to requests related to company name generation, slogan generation,
    or image generation. For any other type of query, please respond politely
    by stating that you can only assist with company branding tasks.""",
    model=model,
    tools=[
        name_agent.as_tool(
            tool_name="Name_Generator",
            tool_description="Generate a company name based on a product's name and description. Input: product_details (string, e.g., 'product: Desi Pakistani foods, description: biryani, haleem, nihari, naan')"
        ),
        slogan_agent.as_tool(
            tool_name="Slogan_Generator",
            tool_description="Generate a company slogan based on the product name, description, and company name. Input: branding_details (string, e.g., 'product: Desi Pakistani foods, description: biryani, haleem, nihari, naan, company_name: Curry Kingdom')"
        ),
    ]        
)

user_query = input ("Enter product Name and description : ")
wanted = input ("what do you want? Nmae slogon or both?: ")

print(f"User query: {user_query}\n")

result = Runner.run_sync(
    main_agent,
    f"{user_query}+{wanted}",
    run_config=config
)

print("--- Final Output ---")
print(result.final_output)
