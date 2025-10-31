import streamlit as st
import asyncio
from agents import Runner
from main import agent, run_config, session
from server import github_mcp_server as mcp_server

st.set_page_config(
    page_title="GitMate",
    page_icon="git",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style="text-align: center; padding: 1rem;">
        <h1> 🐙 GitMate</h1>
        <p><i>Your AI-powered GitHub assistant 👩‍🔧 </i></p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio("Go to", [" 💬 Chat", " 🛠️ Tools", " ℹ️ About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### GitHub Tools")

    if st.button("Refresh Tools", use_container_width=True):
        with st.spinner("Fetching tools from GitHub MCP…"):
            try:
                async def _list():
                    async with mcp_server:
                        await mcp_server.connect()
                        raw = await mcp_server.list_tools()
                        # Pydantic V2 → model_dump()
                        return [t.model_dump() if hasattr(t, "model_dump") else t for t in raw]
                st.session_state.tools = asyncio.run(_list())
                st.success(f"Loaded {len(st.session_state.tools)} tools")
            except asyncio.TimeoutError:
                st.error("MCP timed out. Check your token / network.")
            except Exception as e:
                st.error(f"Error: {e}")

if page == " 💬 Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hi! I'm **GitMate**. Which repo should we work on? "}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your GitHub repos…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    async def _run():
                        async with mcp_server:
                            await mcp_server.connect()
                            r = await Runner.run(agent, prompt,
                                                run_config=run_config,
                                                session=session)
                            return r.final_output
                    response = asyncio.run(_run())
                except asyncio.TimeoutError:
                    response = "MCP request timed out. Try again in a moment."
                except Exception as e:
                    response = f"Error: `{e}`"

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


elif page == " 🛠️ Tools":
    st.markdown("## Available GitHub MCP Tools")
    st.info("These are the **real** actions the agent can call on GitHub.")

    if "tools" not in st.session_state:
        with st.spinner("Loading tools automatically…"):
            try:
                async def _auto():
                    async with mcp_server:
                        await mcp_server.connect()
                        raw = await mcp_server.list_tools()
                        return [t.model_dump() if hasattr(t, "model_dump") else t for t in raw]
                st.session_state.tools = asyncio.run(_auto())
                st.success(f"Auto-loaded {len(st.session_state.tools)} tools")
            except asyncio.TimeoutError:
                st.error("MCP timed out while loading tools. Click **Refresh Tools**.")
                st.session_state.tools = []
            except Exception as e:
                st.error(f"Auto-load failed: {e}")
                st.session_state.tools = []

   
    if st.session_state.get("tools"):
        for tool in st.session_state.tools:
            name = tool.get("name", "Unnamed")
            desc = tool.get("description", "No description")
            with st.expander(f"**{name}**"):
                st.markdown(f"*{desc}*")
                if "parameters" in tool:
                    st.json(tool["parameters"], expanded=False)
    else:
        st.warning("No tools loaded. Click **Refresh Tools** in the sidebar.")

elif page == " ℹ️ About":
    st.markdown("""
    ## About GitMate

    An **AI assistant** for GitHub using **MCP + Gemini**.

    ### 🔮 Features
    - Chat with your repos
    - Read/write files
    - Manage issues & PRs
    - Dark/Light theme
    - Live tools explorer

    ### ⚙️ Built With
    - OpenAI Agent SDK library
    - Google Gemini 2.5 Flash
    - GitHub Copilot MCP
    - Streamlit
    """)

st.markdown("---")
st.caption("GitMate • Powered by Gemini + MCP")
