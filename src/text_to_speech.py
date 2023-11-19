import pyttsx3
from gtts import gTTS

def speak(text):
    # Convert text to speech using pyttsx3 or gTTS
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_speech_text(object_names):
    # Generate text for speech
    if object_names:
        text = f"I see {', '.join(object_names)}"
    else:
        text = "No objects detected."
    return text
