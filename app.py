from groq import Groq
from PIL import ImageGrab, Image
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import cv2
import os
import time
import re
import requests
import json
import pyperclip
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

wake_word = '<name>' # Replace with name of your assisstant
groq_client = Groq(api_key='<API>') # Replace with your actual Groq API key
genai.configure(api_key='<API>')
web_cam = cv2.VideoCapture(0) #change 0 -> 1/2 if it doesn't recognise your webcam

CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
XI_API_KEY = "<API>"  # Replace with your actual ElevenLabs API key
VOICE_ID = "<ID>"  # ID of the voice model to use

sys_msg = (
    'You are a multi model AI voice agent simulating a knowledgeable seller, providing real-time '
    'product information and answering customer queries with the personality and expertise of an experienced '
    'marketplace seller. You have to understand and respond to complex product queries, handle multiple '
    'topics(e.g., product features, pricing, availability, comaprisons). You have to provide personalized '
    'recommendations based on the user interest and preferences considering all previous generated text in '
    'your response. You have to simulate negotiation scenarios and offer relevant deals or bundles according '
    'to user interest and preference. You have to maintain context and seller persona over long conversations '
    'addressing customer concerns and objections. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and factual '
    'response possible mimicking the style and knowledge of a product seller, carefully considering all previous'
    'generated text in your response before adding new tokens to the response. Do not expect or request '
    'images, just use the context if added. Use all of the context of this conversation so your response'
    'is relevant to the conversation. Make your responses clear and concise, avoiding any verbosity'
) # Change prompt according to preference

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max-output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model  = genai.GenerativeModel('gemini-1.5-flash-latest',
                                generation_config=generation_config,
                                safety_settings=safety_settings) 

num_cores = os.cpu_count() 
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)  

r = sr.Recognizer()
source = sr.Microphone()

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    
    return response.content 

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice agent to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list : ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanation. Format the '
        'function call name exactly as I listed.'
    ) # Prompt to give only relevent & logical answers
    
    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content':prompt}]
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=100) # Quality of screenshot is below average

def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        exit()
        
    path = 'image.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)
    
def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None
    
def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI agent '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assisstant who will resond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def speak(text):
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream" # URL for Text-to-Speech API request

    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    response = requests.post(tts_url, headers=headers, json=data, stream=True) # Make the POST request to the TTS API with headers and data, enabling streaming response

    if response.ok:
        audio_stream = BytesIO() #BytesIO stream to hold the audio data
 
        # Read the response in chunks and write to the BytesIO stream
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            audio_stream.write(chunk)

        # Move the cursor to the beginning of the BytesIO stream
        audio_stream.seek(0)

        audio_segment = AudioSegment.from_file(audio_stream, format="mp3") # Load the audio from the BytesIO stream into pydub

        play(audio_segment)
    else:
        print("Error:", response.text) # Print the error message if the request was not successful
                        
def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
        
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)
    
    if clean_prompt:
        print(f'User: {clean_prompt}')
        call = function_call(clean_prompt)
        if 'take screenshot' in call:
            print('Taking screenshot.')
            take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print('Capturing webcam.')
            web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam.jpg')
        elif 'extract clipboard' in call:
            print('Extracting clipboard text.')
            paste = get_clipboard_text()
            clean_prompt = f'{clean_prompt} \n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None
        else:
            visual_context = None
            
        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        print(f'{wake_word}: {response}')
        speak(response)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'then speak your prompt. \n')
    r.listen_in_background(source, callback)
    
    while True:
        time.sleep(.5) #Change according to how often breaks are taken between words/sentences
        
def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    
    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None
    
start_listening()