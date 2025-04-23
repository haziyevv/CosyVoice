import sys
import os
import gradio as gr
import torch
import torchaudio
import random
import numpy as np
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice
from cosyvoice.utils.file_utils import load_wav

# Set seeds for reproducibility
def set_seeds(seed=1986):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds at startup
set_seeds()

# Initialize CosyVoice models
cosyvoice = CosyVoice('old/CosyVoice-300M', load_jit=False, load_trt=False, fp16=False)
cosyvoice_12epoch = CosyVoice('pretrained_models/CosyVoice-300M-finetuned2epochs', load_jit=False, load_trt=False, fp16=False)
cosyvoice2_base = CosyVoice2('old/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
cosyvoice2 = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)


def process_tts(text, audio_prompt, model_choice, seed=1986):
    try:
        print(f"\n=== Starting TTS Process ===")
        print(f"Input Text: {text}")
        print(f"Audio Prompt Path: {audio_prompt}")
        print(f"Selected Model: {model_choice}")
        print(f"Using Seed: {seed}")
        
        # Reset seeds before each generation
        set_seeds(seed)
        print("Random seeds set for reproducibility")
        
        # Load the audio prompt
        print("Loading audio prompt...")
        prompt_speech_16k = load_wav(audio_prompt, 16000)
        print(f"Audio prompt loaded - Shape: {prompt_speech_16k.shape}, Type: {prompt_speech_16k.dtype}")
        
        # Select the model based on choice
        print("Selecting model...")
        if model_choice == "CosyVoice 300M":
            model = cosyvoice
        elif model_choice == "CosyVoice 300M-12epochs":
            model = cosyvoice_12epoch
        elif model_choice == "CosyVoice2 0.5B":  # CosyVoice2 0.5B
            model = cosyvoice2
        else:
            model = cosyvoice2_base
        print(f"Model selected: {model.__class__.__name__}")
        
        # Generate TTS
        print("Generating speech...")
        outputs = list(model.inference_cross_lingual(text, prompt_speech_16k, stream=False))

        print(f"Speech generated - Got {len(outputs)} outputs")
        
        if not outputs:
            print("Error: No outputs generated")
            return None, "No audio was generated"
        

        # Extract and concatenate audio tensors
        speech_segments = [out["tts_speech"] for out in outputs if "tts_speech" in out]
        full_audio = torch.cat(speech_segments, dim=-1)
        
        # Save the concatenated result
        output_path = "output.wav"
        torchaudio.save(output_path, full_audio, model.sample_rate)
        print("Concatenated audio saved")

        return output_path, f"Successfully generated and concatenated {len(outputs)} segments using {model_choice}!"



        # # Save the output temporarily
        # output_path = "output.wav"
        # print(f"Saving output to {output_path}")
        # print(f"Output audio shape: {outputs[0]['tts_speech'].shape}")
        # torchaudio.save(output_path, outputs[0]['tts_speech'], model.sample_rate)
        # print("=== TTS Process Complete ===\n")
        
        # return output_path, f"Successfully generated audio using {model_choice}!"
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="CosyVoice TTS") as demo:
    gr.Markdown("# CosyVoice Text-to-Speech")
    gr.Markdown("Upload a voice prompt and enter text to generate speech in that voice.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to Convert",
                placeholder="Enter the text you want to convert to speech...",
                lines=3
            )
            audio_prompt = gr.Audio(
                label="Voice Prompt (16kHz WAV file)",
                type="filepath"
            )
            model_choice = gr.Radio(
                choices=["CosyVoice 300M", "CosyVoice 300M-12epochs", "CosyVoice2 0.5B", "CosyVocie2 0.5B base"],
                value="CosyVoice2 0.5B",
                label="Model Selection",
                info="CosyVoice2 0.5B is newer and may produce better quality"
            )
            seed_input = gr.Number(
                label="Random Seed",
                value=1986,
                precision=0,
                minimum=0,
                maximum=999999
            )
            submit_btn = gr.Button("Generate Speech")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")
            status_output = gr.Textbox(label="Status")
    
    submit_btn.click(
        fn=process_tts,
        inputs=[text_input, audio_prompt, model_choice, seed_input],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
