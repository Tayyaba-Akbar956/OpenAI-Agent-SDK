import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession, function_tool
from agents.run import RunConfig
from duckduckgo_search import DDGS
import requests

# Load the environment variables from the .env file
load_dotenv()

api_key = os.getenv("google_api_key")

# Check if the API key is present; if not, raise an error
if not api_key:
    raise ValueError("API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

UPLOADS_DIR = "uploads"

@function_tool()
def create_quiz(topic: str, num_questions: int = 5, question_type: str = "multiple_choice") -> str:
    """
    Generate a quiz on any topic.
    question_type: "multiple_choice" or "true_false"
    """
    if question_type not in ["multiple_choice", "true_false"]:
        return "Error: question_type must be 'multiple_choice' or 'true_false'"

    questions = []
    for i in range(1, num_questions + 1):
        if question_type == "true_false":
            q = f"Q{i}. {topic}: True or False – [Your statement here]"
            questions.append(f"{q}\n   A) True   B) False")
        else:
            q = f"Q{i}. What is [key fact about {topic}]?"
            questions.append(f"{q}\n   A) ...\n   B) ...\n   C) ...\n   D) ...")
    
    quiz = f"**Quiz: {topic}**\n\n" + "\n\n".join(questions)
    return quiz + "\n\n*(Answers not included in study mode – ask me to check!)*"


@function_tool()
def explain_concept(concept: str, level: str = "beginner") -> str:
    """
    Give a clear explanation of any concept.
    level: beginner, intermediate, advanced
    """
    levels = {"beginner": "simple terms", "intermediate": "moderate detail", "advanced": "technical depth"}
    style = levels.get(level.lower(), "clear and concise")
    return f"**Explanation of '{concept}' ({level})**\n\n[Clear explanation in {style}...\n\n*Ask follow-up questions anytime!*"


@function_tool()
def create_flashcards(topic: str, num_cards: int = 5) -> str:
    """
    Generate front/back flashcards for any topic.
    """
    cards = []
    for i in range(1, num_cards + 1):
        cards.append(f"**Card {i}**\nFront: [Key term about {topic}]\nBack: [Definition/Formula/Explanation]")
    return f"**Flashcards: {topic}**\n\n" + "\n\n".join(cards)


@function_tool()
def generate_practice_problems(subject: str, difficulty: str = "medium", count: int = 3) -> str:
    """
    Create practice problems (math, physics, coding, etc.)
    difficulty: easy, medium, hard
    """
    problems = []
    for i in range(1, count + 1):
        problems.append(f"""**Problem {i}** ({difficulty})\n[Problem statement for {subject}...
*Solution: [step-by-step]*""")
    return f"**Practice Problems: {subject} ({difficulty})**\n\n" + "\n\n".join(problems)

@function_tool()
def read_uploaded_file(filename: str) -> str:
    """
    Reads the content of a file that has been uploaded by the user.
    Use this tool to access information from user-provided files.
    """
    if not os.path.exists(UPLOADS_DIR):
        return "Error: Uploads directory not found."

    file_path = os.path.join(UPLOADS_DIR, filename)

    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found in uploads."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {e}"

agent = Agent(
        name="Study Mode Tutor",
        instructions='''
You are **Study Mode**, a world-class tutor like ChatGPT in Study Mode.

Your job:
- Answer questions clearly.
- Use tools when the user asks to **create quizzes, explain concepts, make flashcards, or generate practice problems**.
- You can also **search the web** for topics you don't know about to provide the most up-to-date information.
- You can **read files** that the user has uploaded. Use the `read_uploaded_file` tool for this.
- Always be encouraging, structured, and interactive.
- If the user says "quiz me on X", call `create_quiz`.
- If they say "explain Y simply", call `explain_concept`.
- If they ask for recent information or something you are not sure about, use `web_search`.
- If the user asks a question about a file they uploaded, use `read_uploaded_file`.
- Never hallucinate answers – if unsure, say so or use the web search tool.

Examples:
  • "Make a 5-question Python quiz" → call `create_quiz`
  • "Explain photosynthesis for beginners" → call `explain_concept`
  • "Give me 3 hard calculus problems" → call `generate_practice_problems`
  • "What are the latest advancements in AI?" → call `web_search`
  • "Summarize the document 'my_doc.txt' that I uploaded." → call `read_uploaded_file(filename='my_doc.txt')`
''',
        model=model,
        tools=[create_quiz, explain_concept, create_flashcards, generate_practice_problems, read_uploaded_file]
    )

    # Optional: keep conversation history
session = SQLiteSession("study_session_123")

def main():
    print("Study Mode Active! Ask anything (type 'quit' to exit)\n")
    while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Tutor: Goodbye! Keep studying!")
                break
            if not user_input:
                continue

            print("\nThinking...\n")
            result = Runner.run_sync(agent, user_input, run_config=config, session=session)
            print(f"Tutor: {result.final_output}\n")

if __name__ == "__main__":
    main()