import gradio as gr
import torch
import random
import numpy as np
import torchaudio
import os
import sys
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import time

# Add Matcha-TTS to the path
sys.path.append('third_party/Matcha-TTS')

# Set seeds
def set_seeds(seed=1986):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Initialize models and load default prompt
set_seeds()
cosyvoice_model = CosyVoice2('pretrained_models/cosyvoice2-0.5b-15epochs-elise-data', 
                            load_jit=False, 
                            load_trt=False, 
                            fp16=False, 
                            use_flow_cache=False)

# Load the default audio prompt
DEFAULT_PROMPT_PATH = "sample_0.wav"  # Adjust this path to your actual prompt file
default_prompt_speech = load_wav(DEFAULT_PROMPT_PATH, 16000)

def tts_generate(text: str, progress=gr.Progress()):
    try:
        progress(0, desc="Starting generation...")
        set_seeds()
        
        progress(0.3, desc="Processing text...")
        # Generate speech using the default prompt
        outputs = list(cosyvoice_model.inference_cross_lingual(text, default_prompt_speech, stream=False))
        if not outputs:
            raise gr.Error("No output generated")
        
        progress(0.7, desc="Saving audio...")    
        # Save the output
        output_path = "output.wav"
        torchaudio.save(output_path, outputs[0]['tts_speech'], cosyvoice_model.sample_rate)
        
        progress(1.0, desc="Done!")
        return output_path
    
    except Exception as e:
        raise gr.Error(str(e))

with gr.Blocks() as demo:
    gr.Markdown("# CosyVoice Text-to-Speech")
    gr.Markdown("Generate speech using CosyVoice TTS model with a pre-loaded voice prompt.")
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Text to synthesize",
            placeholder="Enter the text you want to convert to speech...",
            lines=3
        )
    
    with gr.Row():
        generate_btn = gr.Button("Generate Speech", variant="primary")
        clear_btn = gr.Button("Clear")
    
    with gr.Row():
        audio_output = gr.Audio(label="Generated Speech")
    
    # Set up event handlers
    generate_btn.click(
        fn=tts_generate,
        inputs=[text_input],
        outputs=[audio_output],
        show_progress=True,
    )
    
    clear_btn.click(
        fn=lambda: [None, ""],
        inputs=[],
        outputs=[audio_output, text_input],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True) 