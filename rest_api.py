from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os
import torch
import random
import numpy as np
import torchaudio
import tempfile
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice
from cosyvoice.utils.file_utils import load_wav
import sys

# Add Matcha-TTS to the path
sys.path.append('third_party/Matcha-TTS')

app = FastAPI(title="CosyVoice TTS API")

# Set seeds
def set_seeds(seed=1986):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Initialize models
set_seeds()
cosyvoice_model = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)


@app.post("/tts")
async def tts_generate(
    text: str = Form(...),
    seed: int = Form(1986),
    audio_prompt: UploadFile = File(...)
):
    try:
        # Set seed
        set_seeds(seed)

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio_prompt.read())
            tmp_path = tmp.name

        # Load audio prompt
        prompt_speech_16k = load_wav(tmp_path, 16000)

        model = cosyvoice_model

        outputs = list(model.inference_cross_lingual(text, prompt_speech_16k, stream=False))
        if not outputs:
            return JSONResponse(status_code=500, content={"error": "No output generated"})

        output_path = "output.wav"
        torchaudio.save(output_path, outputs[0]['tts_speech'], model.sample_rate)

        return FileResponse(output_path, media_type="audio/wav", filename="generated.wav")
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)