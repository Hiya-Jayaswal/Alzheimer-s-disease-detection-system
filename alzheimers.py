import openai
import pyttsx3
import speech_recognition as sr
import re
import spacy
from spacy.matcher import PhraseMatcher
from fuzzywuzzy import fuzz
import difflib

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000
recognizer.pause_threshold = 0.8

# Set OpenAI API key
openai.api_key = 'my-openai-key'

nlp = spacy.load("en_core_web_sm")

alzheimers_keywords = ["alzheimer's", "alzheimers", "alzheimer", "dementia", "memory loss", "cognitive impairment", "symptoms"]

# Create a PhraseMatcher with the keywords
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(text) for text in alzheimers_keywords]
matcher.add("AlzKeywords", None, *patterns)

def text_to_speech(text):
    """Convert text to speech using pyttsx3."""
    engine.say(text)
    engine.runAndWait()

def speech_to_text():
    """Convert speech to text using speech recognition."""
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand that.")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

def process_input(text):
    """Process user input to handle variations in 'yes'."""
    if re.search(r'\b(yes|yeah|yep|sure)\b', text, re.IGNORECASE):
        return "yes"
    elif re.search(r'\b(no|nope)\b', text, re.IGNORECASE):
        return "no"
    else:
        return text.lower()

def is_alzheimers_related(text):
    """Check if the text contains Alzheimer's-related keywords."""
    doc = nlp(text)
    matches = matcher(doc)
    return bool(matches)

def answer_alzheimers_questions(question):
    """Answer Alzheimer's-related questions using OpenAI's GPT-3."""
    if is_alzheimers_related(question):
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="You are Alzea, a financial AI. Respond to user queries with a financial lens, even if non-financial also give me advice based on it and provide me the current data of stock prices of the companies. Educate and empower users with financial knowledge, providing full but short and crisp information." + question + "in short",
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].text.strip()
    else:
        return "I'm sorry, I can only answer questions related to Alzheimer's disease."

def alzheimers_questionnaire(input_method):
    """Conduct Alzheimer's symptom questionnaire."""
    print("Let's assess your Alzheimer's symptoms.")
    symptoms = [
        "Do you experience greater memory loss, such as forgetting recent conversations and appointments?",
        "Have you noticed wandering and getting lost, especially in places you used to know well?",
        "Are you taking longer than usual to complete normal daily tasks?",
        "Do you struggle with word-finding difficulties and expressing thoughts or taking part in conversations?",
        "Have you observed repetitive questioning behavior, along with mood changes like agitation or aggression?",
        "Are you becoming less flexible and hesitant to try new things?",
        "Have you noticed a decline in your ability to learn new things?",
        "Do you have trouble recognizing family and friends?",
        "Are you experiencing hallucinations, delusions, or paranoia?",
        "Have you had difficulty solving basic problems or keeping track of important tasks?",
        "Do you find yourself increasingly confused or disoriented about time, place, or life events?",
        "Are you experiencing mood swings, depression, or anxiety?",
        "Have you observed obsessive, repetitive, or impulsive behavior?",
        "Do you have trouble with speech or language difficulties, such as trouble finding the right word?",
        "Have you noticed significant short- and long-term memory problems?"
    ]
    severity = 0
    for symptom in symptoms:
        while True:
            print("Finch:", symptom)
            text_to_speech(symptom)
            if input_method == 'text':
                response = input("You: ")
            else:
                response = speech_to_text()
            processed_response = process_input(response)
            similarity = fuzz.partial_ratio(symptom.lower(), processed_response)
            if similarity > 70:  # Adjust threshold as needed
                if processed_response in ["yes", "no"]:
                    if processed_response == "yes":
                        severity += 1
                    break
                else:
                    print("Wrong input. Please answer with 'yes' or 'no'.")
                    text_to_speech("Wrong input. Please answer with 'yes' or 'no'.")
            else:
                print("Could not understand your response. Please try again.")
                text_to_speech("Could not understand your response. Please try again.")
    print("Analyzing your symptoms...")
    if severity >= 8:
        text_to_speech("Based on your symptoms, it seems like you may have severe Alzheimer's disease.")
    elif severity >= 4:
        text_to_speech("Based on your symptoms, it seems like you may have moderate Alzheimer's disease.")
    elif severity >= 2:
        text_to_speech("Based on your symptoms, it seems like you may have mild Alzheimer's disease.")
    else:
        text_to_speech("Your symptoms do not seem to indicate Alzheimer's disease.")

def list_doctors_treating_alzheimers():
    """Retrieve a list of doctors treating Alzheimer's disease."""
    question = "List of 10 doctors in India treating Alzheimer's disease along with their contact information"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=question + " Please provide a short answer:",
        temperature=0.5,
        max_tokens=500 
    )
    return response.choices[0].text.strip()

def main():
    print("Hello! I'm your virtual assistant, Alzea.")
    text_to_speech("Hello! I'm your virtual assistant...Alzea")

    while True:
        print("Do you want to answer using text or speech?")
        print("1. Text")
        print("2. Speech")
        choice = input("Enter your choice (1/2): ")

        if choice == '1':
            input_method = 'text'
        elif choice == '2':
            input_method = 'speech'
        else:
            print("Invalid choice. Defaulting to text input.")
            input_method = 'text'

        print("Select an option:")
        print("1. Alzheimer's Symptom Questionnaire")
        print("2. Ask Alzheimer's Related Questions")
        print("3. Know about doctors treating Alzheimers in INDIA")
        print("4. Press 4 to exit")
        option = input("Enter your choice (1/2/3/4): ")
        
        if option == '1':
            alzheimers_questionnaire(input_method)
        elif option == '2':
            print("You can ask me any question related to Alzheimer's disease.")
            if input_method == 'text':
                question = input("You: ")
            else:
                question = speech_to_text()
            response = answer_alzheimers_questions(question)
            text_to_speech(response)
            print(response)
        elif option == '3':
            text_to_speech("Here are the top doctors treating Alzheimer's disease in India:")
            doctors_info = list_doctors_treating_alzheimers()
            print(doctors_info)
        elif option == '4':
            print("Take Care!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
