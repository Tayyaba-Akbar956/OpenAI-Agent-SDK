import os
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, ModelSettings
from agents.run import RunConfig, Runner
import nest_asyncio
import asyncio
import time # Import time for rate limiting

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Ensure API key is present
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
    st.stop()

# Configure the OpenAI client and model
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

# Define the agents
name_agent = Agent(
    name="Name Agent",
    instructions="""You are a company name generator. Based on the product name and description,
    you generate creative and suitable company names.
    Provide only the names, comma-separated, without additional commentary.
    Generate at least 5-10 distinct names.""",
    model=model
)

slogan_agent = Agent(
    name="Slogan Generator",
    instructions="""You are a slogan generator. Based on the product name, description, and
    company name, you generate a highly creative and catchy slogan.
    Provide only the slogan, without additional commentary.""",
    model=model
)

main_agent = Agent(
    name="Triage Agent",
    instructions="""You are a Triage Agent. Your primary role is to direct user requests to the appropriate
    tool. You should only respond to requests related to company name generation, slogan generation,
    or image generation. For any other type of query, please respond politely
    by stating that you can only assist with company branding tasks.
    If the user asks for "both" (name and slogan), generate the names first, then generate slogans for each name.
    dont give any other line just geenrate name or slogans strictly dont add any additionl line in final output""",
    model=model,
    tools=[
        name_agent.as_tool(
            tool_name="Name_Generator",
            tool_description="Generate multiple company names based on a product's name and description. Input: product_details (string, e.g., 'product: Desi Pakistani foods, description: biryani, haleem, nihari, naan')"
        ),
        slogan_agent.as_tool(
            tool_name="Slogan_Generator",
            tool_description="Generate a company slogan based on the product name, description, and *one* company name. Input: branding_details (string, e.g., 'product: Desi Pakistani foods, description: biryani, haleem, nihari, naan, company_name: Curry Kingdom')"
        ),
    ]
)

# --- Streamlit UI ---
st.set_page_config(page_title="Company Branding Assistant", layout="centered")

st.title("✨ Company Branding Assistant")
st.markdown("""
Welcome! Let me help you generate creative names and catchy slogans for your products.
Enter your product details below and choose what you'd like to generate.
""")

with st.expander("💡 How it works"):
    st.write("""
    This application uses an AI-powered agent system to generate company names and slogans.
    - **Product Name & Description:** Tell me about your product. The more details, the better!
    - **What do you want?:** Choose to generate a name, a slogan, or both.
    - If you request a slogan, the system will first attempt to generate a company name (if not provided)
      and then use that name to create a slogan. If you choose "Both", it will generate multiple names
      and a slogan for each of them!
    """)

# Input fields
product_name = st.text_input("📦 Product Name (e.g., 'Desi Pakistani foods')", placeholder="e.g., 'Organic Dog Food'")
product_description = st.text_area("📝 Product Description (e.g., 'biryani, haleem, nihari, naan')",
                                   placeholder="e.g., 'High-quality kibble made from locally sourced, organic ingredients'")

# Selection for what to generate
wanted = st.radio("🎯 What do you want to generate?", ("Name", "Slogan", "Both"))

# Optional input for company name if user wants only a slogan and already has a name
company_name_for_slogan = ""
if wanted == "Slogan":
    company_name_for_slogan = st.text_input("Optional: Enter existing Company Name for Slogan (if available)",
                                            placeholder="e.g., 'Curry Kingdom'")
    st.info("If you leave the company name blank, a name will be generated first.")


if st.button("🚀 Generate Branding Ideas"):
    if not product_name and not product_description:
        st.warning("Please provide at least a Product Name or Description.")
    else:
        full_product_details = f"product: {product_name}, description: {product_description}"
        st.info(f"Sending request for: '{wanted}' based on '{full_product_details}'...")

        with st.spinner("Generating your branding ideas... Please wait."):
            try:
                company_names = []
                slogan_results = []
                final_output_string = ""

                # --- Step 1: Generate Company Name(s) ---
                if wanted in ["Name", "Both"] or (wanted == "Slogan" and not company_name_for_slogan):
                    name_query = f"{full_product_details}, Generate company names."
                    with st.spinner("Generating company names..."):
                        name_result = Runner.run_sync(
                            main_agent,
                            name_query,
                            run_config=config
                        )
                    raw_names = name_result.final_output.strip()
                    # Clean and split names - handle various delimiters and extra text
                    # We'll split by comma, semicolon, or ".\n"
                    # Then clean each part
                    cleaned_names = []
                    # First, remove common introductory phrases if they appear
                    raw_names = raw_names.replace("Okay\nI have generated company names for you. They are:", "").strip()
                    raw_names = raw_names.replace("I have generated company names for you. They are:", "").strip()
                    raw_names = raw_names.replace("Company Names:", "").strip()


                    # Try splitting by newline first, then by comma if no newlines
                    if '\n' in raw_names:
                        parts = raw_names.split('\n')
                    else:
                        parts = raw_names.split(',')

                    for part in parts:
                        cleaned_name = part.replace('.', '').strip() # Remove periods and extra whitespace
                        if cleaned_name: # Ensure it's not an empty string after cleaning
                            cleaned_names.append(cleaned_name)

                    company_names = cleaned_names
                    # Fallback if parsing fails or no names generated
                    if not company_names:
                        st.warning("Could not parse company names from the AI's response. Please try again.")
                        st.stop()
                    else:
                        st.success(f"Generated {len(company_names)} company names.")
                        if wanted == "Name":
                            final_output_string += "Okay, I have generated company names for you. They are:\n" # Initial phrase
                            for name in company_names:
                                final_output_string += f"- **{name}**\n" # Each name on a new line with bullet
                        # Add a small delay after name generation to respect rate limits
                        time.sleep(2)
                elif wanted == "Slogan" and company_name_for_slogan:
                    company_names = [company_name_for_slogan] # Use the provided name for slogan generation

                # --- Step 2: Generate Slogans if requested ---
                if wanted in ["Slogan", "Both"] and company_names:
                    if wanted == "Both":
                        final_output_string += "\n### Company Names & Slogans:\n" # New section for combined output
                    elif wanted == "Slogan":
                        final_output_string += "### Company Slogans:\n"

                    # For each company name, generate a slogan
                    for i, name in enumerate(company_names):
                        slogan_query = f"{full_product_details}, company_name: {name}, Generate a slogan."
                        with st.spinner(f"Generating slogan for '{name}' ({i+1}/{len(company_names)})..."):
                            slogan_result = Runner.run_sync(
                                main_agent,
                                slogan_query,
                                run_config=config
                            )
                        slogan = slogan_result.final_output.strip()
                        slogan_results.append((name, slogan))

                        # Format for combined output
                        if wanted == "Both":
                            final_output_string += f"- **{name}**: {slogan}\n"
                        elif wanted == "Slogan":
                            final_output_string += f"- Slogan for **{name}**: {slogan}\n"

                        # Add a delay between each slogan generation
                        if i < len(company_names) - 1:
                            time.sleep(2)

                if final_output_string:
                    st.subheader("🎉 Your Branding Ideas:")
                    st.markdown(final_output_string)
                else:
                    st.info("No output generated. Please check your inputs and try again.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your API key and ensure the product details are clear. If you hit a quota limit (Error 429), please wait a minute and try again, or consider upgrading your API plan.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Google's Gemini Models.")