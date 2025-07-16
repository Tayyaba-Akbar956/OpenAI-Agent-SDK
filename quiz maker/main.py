import random
import os
import asyncio
from dotenv import load_dotenv
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError, Field

# Assuming 'agents' module is correctly installed and configured
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

# Load API key from .env
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Create Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model settings
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", # Or 'gemini-1.5-flash-latest' if available and preferred
    openai_client=client,
)

# Pydantic models for quiz structure
class QuizQuestion(BaseModel):
    question: str = Field(..., description="The text of the multiple-choice question.")
    options: List[str] = Field(..., min_length=4, max_length=4, description="A list of exactly four answer options.")
    answer: str = Field(..., description="The correct answer, which must be one of the options.")

class QuizOutput(BaseModel):
    questions: List[QuizQuestion] = Field(..., description="A list of quiz questions.")

# --- New Pydantic Model for Review Agent Output ---
class ReviewOutput(BaseModel):
    overall_remark: str = Field(..., description="An overall remark about the user's performance, including an emoji.")
    weak_sub_topics: List[str] = Field(..., description="A list of specific sub-topics where the user showed weakness, derived from the questions they answered incorrectly.")
    encouragement: str = Field(..., description="A concluding encouraging remark.")

# Define the quiz generation agent (no changes here)
quiz_generator_agent = Agent(
    name="QuizGenerator",
    model=model,
    output_type = QuizOutput,
    instructions=f"""
    You are a quiz generator. Your sole task is to generate a list of multiple-choice questions.
    You MUST return ONLY a JSON object that strictly adheres to the following Pydantic schema:

    {QuizOutput.model_json_schema()}

    Each question must have exactly four distinct options, and the 'answer' field
    must be one of the provided 'options'.
    Do NOT include any additional text, markdown formatting (like ```json), or explanations outside of the JSON object.
    Ensure the JSON is well-formed and valid.
    """
)

# --- Define the new Quiz Review Agent ---
quiz_review_agent = Agent(
    name="QuizReviewer",
    model=model,
    output_type=ReviewOutput,
    instructions=f"""
    You are a quiz review agent. Your task is to analyze a user's quiz performance
    and provide constructive feedback.

    You will receive the quiz topic, the list of all questions, and information
    about which questions the user answered correctly or incorrectly.

    Your output MUST be a JSON object that strictly adheres to the following Pydantic schema:

    {ReviewOutput.model_json_schema()}

    Based on the correct and incorrect answers:
    1. Provide an 'overall_remark' that includes an emoji, reflecting the user's general performance.
       Examples: "Excellent work! 🎉", "Good effort, keep going! 👍", "You're on your way! 💡"
    2. Identify 'weak_sub_topics'. For each question the user got wrong, try to infer the specific
       sub-topic or concept it tested within the broader quiz topic. List these as concise strings.
       If the user got everything right, this list should be empty.
       Examples: ["Python lists", "World War II dates", "Chemical bonding principles"].
    3. Provide an 'encouragement' message.

    Do NOT include any additional text, markdown formatting (like ```json), or explanations outside of the JSON object.
    Ensure the JSON is well-formed and valid.
    """
)

# Function to ask a question and validate the answer
def ask_question(question_data: QuizQuestion):
    print("\n" + question_data.question)
    
    # Randomize the display order of options for each question
    display_options = list(question_data.options)
    random.shuffle(display_options)

    option_map = {chr(65 + i): option for i, option in enumerate(display_options)}
    for char_code, option_text in option_map.items():
        print(f" {char_code}. {option_text}")
    
    correct_option_char = ""
    # Find the character for the correct answer based on its text
    for char_code, option_text in option_map.items():
        if option_text == question_data.answer:
            correct_option_char = char_code
            break
    
    # This block handles cases where the correct answer might not be in the options
    # due to a generation error. It's a fallback but the agent instructions should
    # ideally prevent this.
    if not correct_option_char:
        print(f"Warning: The correct answer '{question_data.answer}' was not found in the options for this question.")
        # If the answer is truly missing, we can't validate. For robustness, let's treat it as incorrect.
        return False


    while True:
        user_input = input("Enter your choice (A, B, C, D): ").strip().upper()
        if user_input in option_map:
            break
        else:
            print("Invalid choice, please enter a valid option (A, B, C, D).")

    return user_input == correct_option_char

# Function to run the quiz
async def start_quiz():
    print("Welcome to the Dynamic Quiz Game!\n")

    topic = input("Enter the topic for the quiz (e.g., Python programming, World History): ").strip()

    while True:
        try:
            num_questions = int(input("Enter the number of questions you want (e.g., 5, 10, 15): ").strip())
            if num_questions > 0:
                break
            else:
                print("Please enter a positive number of questions.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    difficulty_levels = ["easy", "medium", "hard"]
    print("\nSelect difficulty level:")
    for i, level in enumerate(difficulty_levels, 1):
        print(f"{i}. {level.capitalize()}")

    while True:
        try:
            difficulty_choice = int(input("\nEnter difficulty number: ").strip())
            if 1 <= difficulty_choice <= len(difficulty_levels):
                selected_difficulty = difficulty_levels[difficulty_choice - 1]
                break
            else:
                print("Invalid difficulty choice.")
        except ValueError:
            print("Please enter a valid number.")

    # Construct the prompt for the quiz generation agent
    quiz_generation_prompt = (
        f"Generate a quiz about {topic} with {num_questions} multiple-choice questions "
        f"at a {selected_difficulty} difficulty level.")
    

    print("\nGenerating quiz questions, please wait...\n")
    try:
        # Run the quiz generation agent
        quiz_result = await Runner.run(quiz_generator_agent, quiz_generation_prompt)
        
        quiz_data: QuizOutput = quiz_result.final_output

        questions = quiz_data.questions
        if not questions:
            print("Could not generate any questions for the given topic and parameters. Please try again with different inputs.")
            return

        # Ensure we have the requested number of questions, or use all generated if fewer
        if len(questions) > num_questions:
            random.shuffle(questions)
            questions = questions[:num_questions]
        elif len(questions) < num_questions:
            print(f"Warning: Only {len(questions)} questions were generated, instead of the requested {num_questions}.")

        # Start quiz
        score = 0
        incorrectly_answered_questions = [] # Store questions answered incorrectly

        print("\n--- Starting Quiz ---\n")
        for i, q_data in enumerate(questions):
            print(f"Question {i+1} of {len(questions)}:")
            is_correct = ask_question(q_data)
            if is_correct:
                print("Correct!\n")
                score += 1
            else:
                print(f"Wrong! The correct answer was: {q_data.answer}\n")
                incorrectly_answered_questions.append(q_data) # Add to list for review

        # Display final score
        print(f"Quiz Over! Your Score: {score}/{len(questions)}")

        # --- Run the Quiz Review Agent ---
        print("\nAnalyzing your performance, please wait for the review...\n")
        
        # Prepare the prompt for the review agent
        # Provide all questions and highlight which ones were incorrect
        review_prompt_data = {
            "quiz_topic": topic,
            "total_questions_asked": len(questions),
            "score": score,
            "all_questions": [q.model_dump_json() for q in questions], # Convert to JSON strings for prompt
            "incorrectly_answered_questions": [q.model_dump_json() for q in incorrectly_answered_questions]
        }
        
        review_prompt = f"""
        Quiz Topic: {review_prompt_data['quiz_topic']}
        Total Questions Asked: {review_prompt_data['total_questions_asked']}
        Your Score: {review_prompt_data['score']}

        All Questions:
        {review_prompt_data['all_questions']}

        Questions you answered incorrectly:
        {review_prompt_data['incorrectly_answered_questions']}

        Please provide a review based on the above data.
        """
        
        review_result = await Runner.run(quiz_review_agent, review_prompt)
        review_output: ReviewOutput = review_result.final_output

        print("\n--- Your Quiz Review ---")
        print(f"{review_output.overall_remark}")
        if review_output.weak_sub_topics:
            print("\nBased on your answers, you might want to review these sub-topics:")
            for sub_topic in review_output.weak_sub_topics:
                print(f"- {sub_topic}")
        else:
            print("\nExcellent! You showed strong understanding across all topics.")
        print(f"\n{review_output.encouragement}")
        print("------------------------")

    except ValidationError as e:
        print(f"Error: The generated data is not in the expected format.")
        print(f"Details: {e}")
        print("Please check the agent's instructions and the Pydantic model definition.")
        print("Raw agent output (if available):", getattr(e, 'raw_output', 'N/A'))
    except Exception as e:
        print(f"An unexpected error occurred during quiz generation or review: {e}")
        print("Please try again.")

# Run the quiz
if __name__ == "__main__":
    asyncio.run(start_quiz())