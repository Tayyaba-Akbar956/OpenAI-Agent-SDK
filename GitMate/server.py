# server.py
import os
from dotenv import load_dotenv
from agents.mcp import MCPServerStreamableHttp

load_dotenv()

MCP_TOKEN = os.getenv("MCP_TOKEN")
if not MCP_TOKEN:
    raise ValueError("MCP_TOKEN missing in .env")

github_mcp_server = MCPServerStreamableHttp(
    name="GitHub MCP",
    params={
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {MCP_TOKEN}"},
        "timeout": 60,               
    },
    cache_tools_list=True,
    max_retry_attempts=5,
)
