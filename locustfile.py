from locust import HttpUser, task, between
import base64
import uuid
import os
from pathlib import Path
output_dir = Path("./test_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Load a sample WAV file as base64 string
with open("sample_0.wav", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode("utf-8")

class TTSUser(HttpUser):
    wait_time = between(0.5, 1)

    @task
    def tts_request(self):
        files = {
            "audio_prompt": ("sample_0.wav", base64.b64decode(audio_data), "audio/wav")
        }
        data = {
            "text": "Bubbling with happiness<|endofprompt|>The laughter of children playing in the park fills the air, and it reminds me of the simple joys of life. It's a beautiful day to be alive.",
            "seed": 1986
        }

        try:
            response = self.client.post("/tts", data=data, files=files)
            if response.status_code == 200:
                # Save the returned audio
                filename = output_dir / f"audio_{uuid.uuid4().hex[:8]}.wav"
                with open(filename, "wb") as out_file:
                    out_file.write(response.content)
                print(f"✅ Saved: {filename}")
            else:
                print(f"❌ Failed with status {response.status_code}")
        except Exception as e:
            print(f"❗ Request failed: {e}")