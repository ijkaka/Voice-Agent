import pyttsx3

def text_to_speech(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties before adding anything to speak
    engine.setProperty('rate', 150)    # Speed of speech
    engine.setProperty('volume', 1)    # Volume 0-1

    # Speak the text
    engine.say(text)

    # Wait until the speech is finished
    engine.runAndWait()