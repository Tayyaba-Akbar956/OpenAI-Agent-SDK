import os
import asyncio
import streamlit as st
import json
from agents import Runner

# Import agent, config, and session from main.py
from main import agent, config, session, UPLOADS_DIR

# The session ID is now implicitly handled by the imported session object from main.py
SESSION_ID = session.session_id 

# === Chat History File ===
CHAT_HISTORY_FILE = f"{SESSION_ID}_messages.json"

def load_chat_history():
    """Load chat history from JSON file, tolerating empty/corrupt files."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            # If file exists but is empty, return empty history
            try:
                if os.path.getsize(CHAT_HISTORY_FILE) == 0:
                    return []
            except Exception:
                return []

            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except json.JSONDecodeError:
        # Corrupt or partial file – reset history gracefully
        try:
            os.remove(CHAT_HISTORY_FILE)
        except Exception:
            pass
        return []
    except Exception as e:
        st.warning(f"Could not load chat history: {e}")
    return []

def save_chat_history(messages):
    """Save chat history to JSON file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save chat history: {e}")

# === Streamlit UI ===
st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.caption("Your AI-powered study companion 🤖")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question, request a quiz, or explain a concept..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            message_placeholder = st.empty()
            message_placeholder.markdown("Generating response...")

            try:
                result = asyncio.run(
                    Runner.run(
                        agent,
                        prompt,
                        run_config=config,
                        session=session
                    )
                )
                response = result.final_output
            except Exception as e:
                response = f"Error: {str(e)}"
                st.error("Agent failed to respond.")

            message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_chat_history(st.session_state.messages)

# === Sidebar ===
with st.sidebar:
    st.header("💡 AI Tutor Features")
    st.write(
        """
        - **Create Quizzes:** Test your knowledge on any subject.
        - **Explain Concepts:** Get clear explanations, from simple to advanced.
        - **Make Flashcards:** Generate flashcards for quick review.
        - **Generate Problems:** Practice with problems for subjects like math and physics.
        - **Process Files:** Upload a text file and ask questions about it.
        """
    )

    st.divider()

    st.header("📝 How to Use")
    st.write("Just type what you want in the chat. For example:")
    st.code("Quiz me on Python")
    st.code("Explain black holes for a beginner")

    st.divider()

    st.header("📤 Upload a File")
    uploaded_file = st.file_uploader(
        "Upload a text file for the agent to read.",
        type=['txt', 'md', 'py', 'json', 'csv']
    )
    if uploaded_file is not None:
        if not os.path.exists(UPLOADS_DIR):
            os.makedirs(UPLOADS_DIR)
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"File '{uploaded_file.name}' uploaded. You can now ask questions about it.")

    st.divider()

    st.caption(f"Session ID: `{SESSION_ID}`")
    if st.button("Clear Chat History"):
        st.session_state.messages = []

        db_file = f"{SESSION_ID}.db"
        if os.path.exists(db_file):
            os.remove(db_file)

        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        st.success("History cleared!")
        st.rerun()