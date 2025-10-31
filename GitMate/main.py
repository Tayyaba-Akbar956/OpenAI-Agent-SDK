# agent.py
import os
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from agents.run import RunConfig
from server import github_mcp_server


load_dotenv()
api_key = os.getenv("google_api_key")
if not api_key:
    raise ValueError("google_api_key missing in .env")


client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client,
)

run_config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True,
)

agent = Agent(
    name="GitMate",
    instructions='''
You are **GitMate**, a friendly and expert GitHub assistant.

- Help users explore repos, read code, create files, manage issues/PRs.
- Always **confirm** before making changes.
- Use clean formatting: code blocks, tables, steps.
- Ask for repo name if not specified.
- Be encouraging and clear.
'''.strip(),
    model=model,
    mcp_servers=[github_mcp_server],
)

session = SQLiteSession("github_gemini_streamlit_session")
