# AI Study Tutor

An AI-powered study assistant designed to help you learn more effectively. This agent provides a suite of tools to create study materials, explain concepts, and find up-to-date information on any topic.

## 🚀 Features

*   **Interactive Chat:** Engage with the tutor through a user-friendly web interface or a command-line app.
*   **Web Search:** The agent can search the web to provide answers on recent events and topics it wasn't trained on.
*   **File Processing:** Upload your text documents, and the agent can answer questions based on their content.
*   **Quiz Generation:** Create custom quizzes (multiple choice or true/false) on any subject.
*   **Concept Explanations:** Get clear and concise explanations for complex topics, tailored to your level of understanding.
*   **Flashcard Creation:** Instantly generate flashcards for quick review of key terms.
*   **Practice Problems:** Get practice problems for subjects like math, physics, and more.
*   **Persistent Sessions:** Your conversation history is saved, so you can pick up where you left off.

## 📋 Requirements

*   Python 3.13 or higher.
*   An active Google API key with access to the Gemini models.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management. Install the required packages using:
    ```bash
    pip install .
    ```
    This command reads the dependencies from the `pyproject.toml` file.

## 🔑 Configuration

1.  Create a file named `.env` in the root directory of the project.
2.  Add your Google API key to the `.env` file as follows:
    ```
    google_api_key="YOUR_GEMINI_API_KEY"
    ```
    Replace `"YOUR_GEMINI_API_KEY"` with your actual API key.

## ▶️ Usage

You can interact with the AI Study Tutor through either the web interface or the command-line interface.

### Web Interface (Recommended)

For a rich, interactive experience with file uploads and a graphical interface:

1.  Run the Streamlit app:
    ```bash
    streamlit run UI.py
    ```
2.  Open your web browser to the local URL provided by Streamlit.

### Command-Line Interface

For a lightweight, terminal-based experience:

1.  Run the main script:
    ```bash
    python main.py
    ```
2.  Type your questions directly into the terminal.
