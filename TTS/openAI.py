from openai import openai_client
from openai import OpenAI
import pyaudio

openai_client = OpenAI(api_key='<API>')

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start=False

    with openai_client.audio.speech.with_streaming_response.create(
        model = 'tts-1',
        voice = 'onyx',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold= 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True