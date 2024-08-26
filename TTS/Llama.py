import pyaudio
import requests

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    # url = "https://api.llama.ai/v1/audio/speech"  # Replace with the actual Llama API endpoint
    url = "https://api.llama-api.com"  # Replace with the actual Llama API endpoint
    headers = {
        "Authorization": "LA-851ad79db0f548caae3c42e62ca6ad409217d05524b24da0b5945121317fa463",  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-tts",  # Specify the model if needed
        "input": text,
        "voice": "default",  # Specify the desired voice
        "response_format": "pcm"  # Adjust response format as needed
    }

    with requests.post(url, headers=headers, json=data, stream=True) as response:
        silence_threshold = 0.01
        for chunk in response.iter_content(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

    player_stream.stop_stream()
    player_stream.close()