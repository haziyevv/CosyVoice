import sys
import os
import gradio as gr
import torch
import torchaudio
import random
import numpy as np
import re
from huggingface_hub import snapshot_download

# Required paths for CosyVoice
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2, AutoModel

# This logic replicates your folder scanning to create a Dropdown list
speaker_map = {'Speaker_Samantha', 'Speaker_Honey', 'Speaker_Autumn', 'Speaker_RachelMcAdams', 'Speaker_AlexandraJames', 'Speaker_JoanAllen', 'Speaker_DuaSaleh', 'Speaker_NataliePortman', 'Speaker_AbbieCornish', 'Speaker_ThaddeaGraham', 'Speaker_AimeeLouWood', 'Speaker_GillianAnderson', 'Speaker_ScarletJohanson', 'Speaker_HannahGadsby', 'Speaker_EmmaMackey', 'Speaker_MimiKeene', 'Speaker_DoreeneBlackstock', 'Speaker_LisaMcGrillis', 'Speaker_CateBlanchet', 'Speaker_SharonDuncanBrewster', 'Speaker_SimoneAshley', 'Speaker_AnneMarieDuff', 'Speaker_EvaGreen', 'Speaker_ChinenyeEzeudu', 'Speaker_JemimaKirke', 'Speaker_TanyaReynolds', 'Speaker_SamanthaSpiro', 'Speaker_RakheeThakrar', 'Speaker_AnthonyLexa', 'Speaker_Despina', 'Speaker_Aoede', 'Speaker_Autonoe', 'Speaker_Achernar', 'Speaker_Callirhoe', 'Speaker_Kore', 'Speaker_Pulcherrima', 'Speaker_Vindemiatrix', 'Speaker_Leda', 'Speaker_Laomodeia', 'Speaker_Sulafat', 'Speaker_Erinome', 'Speaker_Zephyr', 'Speaker_Phoenix', 'Speaker_Swiss'}

speaker_choices = sorted(list(speaker_map))

# --- 2. Initialize Model ---
def set_seeds(seed=1986):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

token = os.getenv("HUGGINGFACE_TOKEN")  # Get token from env variable
#model_path = snapshot_download('identityailabs-com/CosyVoice-3-2025.12.24', token=token)

model = AutoModel(model_dir='/workspace/CosyVoice-mine-current/Fun-CosyVoice3-0.5B')

# --- 3. Generation Logic ---
def process_and_generate(text, speaker_name):
    set_seeds(1986)
    if '<|endofprompt|>' not in text:
        text = "You are a helpfull assistant.<|endofprompt|>" +  text

    # Text Pre-processing
    # processed_text = text.replace("!", "")
    processed_text = text.replace("’", "'")
    # processed_text = processed_text.replace("!", ".")
    # processed_text = processed_text.replace("–", ".")
    processed_text = re.sub(r"\.{2,}", " ", processed_text)

    # Prepend speaker context as in your original loop
    processed_text = f"You are a helpfull assistant. You are {speaker_name.split('Speaker_')[1]}. {processed_text}"

    print("FINAL TEXT >>>", repr(processed_text))
    print("FINAL SPEAKER >>>", repr(speaker_name))

    import pdb; pdb.set_trace()
    # Inference
    outputs = list(model.inference_sft(processed_text, speaker_name, stream=False))
    
    audio_data = outputs[0]['tts_speech'].numpy().flatten()
    return (model.sample_rate, audio_data)

# --- 4. Gradio UI ---
with gr.Blocks(title="CosyVoice-3 Speaker Studio") as demo:
    gr.Markdown("## 🎙️ CosyVoice-3 Speaker Selection")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Input Text", 
                lines=5,
                value="Hey, it's Lina here in Dubai!  I'm so glad you're back chatting with me."
            )
            
            with gr.Row():
                # Dropdown for Speaker Selection
                speaker_dropdown = gr.Dropdown(
                    choices=speaker_choices, 
                    label="Select Speaker", 
                    value=speaker_choices[0] if speaker_choices else None
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
            
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Synthesized Audio", interactive=False)

        
    generate_btn.click(
        fn=process_and_generate,
        inputs=[input_text, speaker_dropdown],
        outputs=output_audio
    )

if __name__ == "__main__":
    # Corrected 'server_port' and added 'share=True'
    demo.launch(
        server_name="0.0.0.0", 
        server_port=1992, 
        share=True
    )