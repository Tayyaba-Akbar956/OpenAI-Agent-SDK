import streamlit as st
import random
import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field, ValidationError
# Assuming 'agents' module is correctly set up and available
# You might need to adjust this import based on your actual 'agents' module structure
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner 
import asyncio
import json

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
    st.stop()

# Initialize Gemini client
# Note: AsyncOpenAI is typically used for OpenAI's API.
# For Google's Gemini, you might typically use google.generativeai or specialized client.
# However, if 'AsyncOpenAI' from your 'agents' module is designed to work with
# the OpenAI-compatible Gemini endpoint, this setup is valid.
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model settings
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash", # Changed to 1.5-flash as 2.0-flash might not be generally available or has specific naming
    openai_client=client,
)

# Pydantic models
class QuizQuestion(BaseModel):
    question: str = Field(..., description="The text of the multiple-choice question.")
    options: List[str] = Field(..., min_length=4, max_length=4, description="A list of exactly four answer options.")
    answer: str = Field(..., description="The correct answer, which must be one of the options.")

class QuizOutput(BaseModel):
    questions: List[QuizQuestion] = Field(..., description="A list of quiz questions.")

class ReviewOutput(BaseModel):
    overall_remark: str = Field(..., description="An overall remark about the user's performance, including an emoji.")
    weak_sub_topics: List[str] = Field(..., description="A list of specific sub-topics where the user showed weakness.")
    encouragement: str = Field(..., description="A concluding encouraging remark.")

# Define agents
quiz_generator_agent = Agent(
    name="QuizGenerator",
    model=model,
    output_type=QuizOutput,
    instructions=f"""
    You are a quiz generator. Your sole task is to generate a list of multiple-choice questions.
    You MUST return ONLY a JSON object that strictly adheres to the following Pydantic schema:
    {QuizOutput.model_json_schema()}
    Each question must have exactly four distinct options, and the 'answer' field
    must be one of the provided 'options'.
    Do NOT include any additional text, markdown formatting, or explanations outside of the JSON object.
    Ensure the JSON is well-formed and valid.
    """
)

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
    2. Identify 'weak_sub_topics'. For each question the user got wrong, try to infer the specific
       sub-topic or concept it tested within the broader quiz topic. List these as concise strings.
       If the user got everything right, this list should be empty.
    3. Provide an 'encouragement' message.
    Do NOT include any additional text, markdown formatting, or explanations outside of the JSON object.
    Ensure the JSON is well-formed and valid.
    """
)

# Helper function to run async code in Streamlit
def run_async(coro):
    """
    Runs an asynchronous coroutine in a synchronous context.
    Ensures that an event loop is available and running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError: # No running loop
        loop = None

    if loop and loop.is_running():
        # If a loop is already running, create a new task
        return asyncio.create_task(coro)
    else:
        # Otherwise, run the coroutine directly
        return asyncio.run(coro)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'input'
    st.session_state.quiz_data = None
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.incorrect_questions = []
    st.session_state.user_answers = []
    st.session_state.shuffled_options = []
    st.session_state.topic = ""
    st.session_state.num_questions = 0
    st.session_state.difficulty = ""
    st.session_state.review_output = None

# Main UI
st.title("🎯QUIZ BOT")

if st.session_state.step == 'input':
    st.header("🧪Create Your Quiz")
    with st.form("quiz_form"):
        topic = st.text_input("Enter the topic for the quiz (e.g., Python programming, World History):")
        num_questions = st.number_input("Enter the number of questions you want:", min_value=1, step=1)
        difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard"])
        submitted = st.form_submit_button("Generate Quiz")

    if submitted:
        if not topic:
            st.error("Please enter a topic.")
        elif num_questions < 1:
            st.error("Please enter a positive number of questions.")
        else:
            st.session_state.topic = topic
            st.session_state.num_questions = int(num_questions)
            st.session_state.difficulty = difficulty
            with st.spinner("Generating quiz questions..."):
                try:
                    quiz_prompt = f"Generate a quiz about {topic} with {num_questions} multiple-choice questions at a {difficulty} difficulty level."
                    quiz_result = run_async(Runner.run(quiz_generator_agent, quiz_prompt))
                    
                    # Assuming quiz_result.final_output is already a Pydantic QuizOutput object
                    quiz_data = quiz_result.final_output
                    questions = quiz_data.questions

                    if not questions:
                        st.error("Could not generate questions. Please try again or try a different topic/number of questions.")
                        # Reset state to allow new input
                        st.session_state.step = 'input' 
                        st.stop() # Stop execution after error

                    # Adjust the number of questions if the model returned more or less than requested
                    if len(questions) > st.session_state.num_questions:
                        random.shuffle(questions)
                        questions = questions[:st.session_state.num_questions]
                        st.warning(f"Generated {len(questions)} questions. Displaying the first {st.session_state.num_questions}.")
                    elif len(questions) < st.session_state.num_questions:
                        st.warning(f"Only {len(questions)} questions were generated, instead of {st.session_state.num_questions}.")
                    
                    st.session_state.quiz_data = questions
                    st.session_state.step = 'question'
                    st.session_state.current_question_index = 0
                    st.session_state.score = 0
                    st.session_state.incorrect_questions = []
                    st.session_state.user_answers = [None] * len(questions)
                    st.session_state.shuffled_options = []
                    for q in questions:
                        options = list(q.options)
                        random.shuffle(options)
                        st.session_state.shuffled_options.append(options)
                    st.rerun() # Use st.rerun() for immediate state update

                except Exception as e:
                    st.error(f"An error occurred during quiz generation: {e}")
                    # Reset step to input on error
                    st.session_state.step = 'input'
                    st.rerun() # Rerun to display the input form again

elif st.session_state.step == 'question':
    question_index = st.session_state.current_question_index
    questions = st.session_state.quiz_data

    if not questions or question_index >= len(questions):
        st.error("Quiz data not available or invalid question index. Please generate a new quiz.")
        st.session_state.step = 'input'
        st.rerun()
    else:
        question_data = questions[question_index]
        st.header(f"Question {question_index + 1} of {len(questions)}")
        st.write(question_data.question)

        # Get shuffled options
        options = st.session_state.shuffled_options[question_index]
        option_labels = [chr(65 + i) for i in range(len(options))] # A, B, C, D
        display_options = [f"{label}) {opt}" for label, opt in zip(option_labels, options)]

        # Determine the user's previously selected answer if navigating back/forth
        # Streamlit radio buttons need a unique key per question, and the default value set.
        # The key is already f"q{question_index}", which is good.
        # We need to find the index of the previously selected option for 'index' parameter.
        default_index = None
        if st.session_state.user_answers[question_index] is not None:
            try:
                # Find the index of the user's previous choice based on its label (A, B, C, D)
                default_index = option_labels.index(st.session_state.user_answers[question_index])
            except ValueError:
                # This could happen if options change between reruns, or invalid stored answer
                default_index = None

        user_choice_display = st.radio(
            "Select your answer:", 
            display_options, 
            key=f"q{question_index}",
            index=default_index # Set default selection if user already answered
        )
        
        # Extract the character label (A, B, C, D) from the display string
        user_selected_char = user_choice_display[0] if user_choice_display else None

        if st.button("Submit Answer"):
            # Find the actual text of the user's selected option
            selected_answer_text = None
            if user_selected_char and user_selected_char in option_labels:
                selected_answer_text = options[option_labels.index(user_selected_char)]

            is_correct = (selected_answer_text == question_data.answer)
            st.session_state.user_answers[question_index] = user_selected_char # Store the label (A,B,C,D)

            if is_correct:
                st.success("Correct!")
                st.session_state.score += 1
            else:
                st.error(f"Wrong! The correct answer was: {question_data.answer}")
                # Only append if not already marked incorrect (e.g., if user revisits question)
                if question_data not in st.session_state.incorrect_questions:
                    st.session_state.incorrect_questions.append(question_data)
            
            # Move to next question or review
            if question_index < len(questions) - 1:
                st.session_state.current_question_index += 1
                st.rerun() # Use st.rerun()
            else:
                st.session_state.step = 'review'
                st.rerun() # Use st.rerun()

elif st.session_state.step == 'review':
    st.header("📋Quiz Results")
    st.write(f"Your Score: {st.session_state.score}/{len(st.session_state.quiz_data)}")
    
    # Only run review agent if it hasn't been run or if quiz data changed
    if st.session_state.review_output is None:
        with st.spinner("Analyzing your performance..."):
            try:
                # Construct the prompt for the review agent.
                # It's often better to send structured data to an LLM as a concise string
                # or a JSON string if the agent is explicitly instructed to parse JSON input.
                # Here, we'll format it for readability by the LLM.
                
                all_questions_formatted = []
                for i, q in enumerate(st.session_state.quiz_data):
                    is_correct = q not in st.session_state.incorrect_questions
                    all_questions_formatted.append(f"Question {i+1}: '{q.question}' (Correct Answer: '{q.answer}', User Correct: {is_correct})")

                incorrect_questions_formatted = []
                for i, q in enumerate(st.session_state.incorrect_questions):
                    # Find the original question index to provide more context
                    try:
                        original_idx = st.session_state.quiz_data.index(q)
                        user_answer = st.session_state.user_answers[original_idx]
                        selected_option_text = None
                        if user_answer and original_idx < len(st.session_state.shuffled_options):
                             # Map the stored label (A,B,C,D) back to the actual text option
                            options_for_q = st.session_state.shuffled_options[original_idx]
                            option_labels_for_q = [chr(65 + i) for i in range(len(options_for_q))]
                            if user_answer in option_labels_for_q:
                                selected_option_text = options_for_q[option_labels_for_q.index(user_answer)]


                        incorrect_questions_formatted.append(
                            f"Question {original_idx+1}: '{q.question}' -- Correct Answer: '{q.answer}' "
                            f"-- Your Answer: '{selected_option_text or user_answer}'"
                        )
                    except ValueError:
                        # Fallback if question not found (shouldn't happen with correct state management)
                        incorrect_questions_formatted.append(f"Incorrect question: '{q.question}'")


                review_prompt = f"""
                Analyze the following quiz performance data for the topic: "{st.session_state.topic}".

                Total Questions Asked: {len(st.session_state.quiz_data)}
                Your Score: {st.session_state.score}

                All Questions and their correctness:
                {json.dumps(all_questions_formatted, indent=2)}

                Questions you answered incorrectly (with correct and your answers):
                {json.dumps(incorrect_questions_formatted, indent=2) if incorrect_questions_formatted else "None"}

                Based on this information, provide an 'overall_remark', identify 'weak_sub_topics' (if any),
                and give an 'encouragement' message.
                Remember to strictly adhere to the ReviewOutput Pydantic schema for your JSON output.
                """
                
                review_result = run_async(Runner.run(quiz_review_agent, review_prompt))
                review_output = review_result.final_output
                st.session_state.review_output = review_output

            except Exception as e:
                st.error(f"An error occurred during review: {e}")
                # Do not proceed with displaying review if error
                st.session_state.review_output = None # Clear previous review if error occurs
                if st.button("Try Review Again"):
                    st.rerun() # Allow user to retry review

    if st.session_state.review_output: # Only display if review was successful
        st.subheader("🔍Your Quiz Review")
        st.write(st.session_state.review_output.overall_remark)
        if st.session_state.review_output.weak_sub_topics:
            st.write("**🔁Areas to Review:**")
            for topic in st.session_state.review_output.weak_sub_topics:
                st.write(f"- {topic}")
        else:
            st.write("🏆🌟Excellent! You showed strong understanding across all topics.")
        st.write(st.session_state.review_output.encouragement)
        
    if st.button("Start New Quiz"):
        # Reset all session state variables for a fresh start
        st.session_state.step = 'input'
        st.session_state.quiz_data = None
        st.session_state.current_question_index = 0
        st.session_state.score = 0
        st.session_state.incorrect_questions = []
        st.session_state.user_answers = []
        st.session_state.shuffled_options = []
        st.session_state.topic = ""
        st.session_state.num_questions = 0
        st.session_state.difficulty = ""
        st.session_state.review_output = None
        st.rerun() # Use st.rerun()