from locust import HttpUser, task, between
import base64
import uuid
import os
from pathlib import Path
import requests
import time

output_dir = Path("./test_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Load a sample WAV file as base64 string
with open("sample_0.wav", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode("utf-8")


def generate_tts(seed=1986, url="http://localhost:2022/tts"):
    files = {
        "audio_prompt": ("sample_0.wav", base64.b64decode(audio_data), "audio/wav")
    }
    data = {
        "text": "Bubbling with happiness<|endofprompt|>The laughter of children playing in the park fills the air, and it reminds me of the simple joys of life. It's a beautiful day to be alive.",
        "seed": 1986
    }
        # Send POST request
    response = requests.post(url, data=data, files=files)
    
    if response.status_code == 200:
        # Save the returned audio
        filename = output_dir / f"audio_{uuid.uuid4().hex[:8]}.wav"
        with open(filename, "wb") as out_file:
            out_file.write(response.content)


# calculate the time it takes to generate 100 tts
start_time = time.time()
for i in range(0, 100):
    generate_tts()
end_time = time.time()
print(f"Time taken to generate 100 tts: {end_time - start_time} seconds")