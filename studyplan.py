import pandas as pd
import spacy
import matplotlib.pyplot as plt
import google.generativeai as genai

# Load English language model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Step 1: Data Collection
data = pd.read_csv('student_scores.csv')

# Convert 'Test' column to numeric for analysis
data['Test'] = data['Test'].str.extract(r'(\d+)').astype(int)

# Function to generate study recommendations and timetable using Gemini API
def generate_study_recommendations(student_id, topics=None):
    """
    Generates personalized study recommendations and timetable using Google's Gemini API.

    Args:
    - student_id (int): ID of the student
    - topics (dict): Dictionary containing topics for the next test, with subjects as keys

    Returns:
    - dict: Study recommendations for each subject
    """
    # Use the provided Gemini API key
    api_key = "AIzaSyCnkGOxJyoE4pXQBfC4ihowmYP-1ZWzEZg"
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Create the model
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash",
      generation_config=generation_config,
    )

    # Start chat session
    chat_session = model.start_chat(history=[])
    
    # Send messages and receive responses
    recommendations = {}
    timetable = {}
    for subject in data['Subject'].unique():
        if topics and subject in topics:
            response = chat_session.send_message(f"I want to study {subject} next. Topics: {', '.join(topics[subject])}.")
        else:
            response = chat_session.send_message(f"I want to study {subject}.")
        recommendations[subject] = response.text
        
        # Generate timetable based on Gemini response
        timetable[subject] = generate_timetable(response.text)
    
    return recommendations, timetable

# Function to generate timetable based on Gemini response
def generate_timetable(gemini_response):
    """
    Generates a personalized timetable based on the Gemini response.

    Args:
    - gemini_response (str): Response from the Gemini API

    Returns:
    - pd.DataFrame: Timetable with guidance for each subject
    """
    # Dummy implementation: Generate a sample timetable DataFrame with guidance
    timetable_data = {
        "Time": ["8:00 AM - 10:00 AM", "10:00 AM - 12:00 PM", "12:00 PM - 2:00 PM"],
        "Guidance": ["Focus more on this topic", "Review this topic", "You're doing well in this topic"]
    }
    timetable_df = pd.DataFrame(timetable_data)
    return timetable_df

# Get input from user
student_id = int(input("Enter Student ID: "))

# Get topics for the next test from the user
topics = {}
for subject in data['Subject'].unique():
    topic_input = input(f"Enter topics for the next test in {subject} (comma-separated): ")
    topics[subject] = [topic.strip() for topic in topic_input.split(',')]

# Generate study recommendations and timetable
study_recommendations, timetable = generate_study_recommendations(student_id, topics)

# Print study recommendations
print("Study Recommendations:")
for subject, recommendation in study_recommendations.items():
    print(f"- {subject}: {recommendation}")

# Print timetable
print("\nTimetable:")
for subject, df in timetable.items():
    print(f"\nSubject: {subject}")
    print(df)
