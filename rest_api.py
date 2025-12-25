import sys
import os
import gradio as gr
import torch
import torchaudio
import random
import numpy as np
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice, AutoModel
from cosyvoice.utils.file_utils import load_wav
import re
from huggingface_hub import snapshot_download

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

texts = ["You are a helpfull assistant. Please speak with excitement and tenderness.<|endofprompt|>Hey, it's Lina here in Dubai [laugh] No worries at all about being busy – I totally get how work can take over, and I'm just glad you're back chatting with me. The heat here is still crazy, like 40 degrees and feeling even hotter, but I'm keeping cool with some iced coffee in my studio. Tell me, what's got you so swamped lately? I'd love to hear all about it when you can.", "Please speak with flirtatiousness and excitement.<|endofprompt|>Hey, it's Lina here in Dubai – oh my god, that's so sweet of you to say I have a gorgeous voice! [laugh softly] I'm blushing a little now, honestly. Alright, let's switch to talking live like this; tell me more about what's on your mind while I chat back in real time. What've you been up to today, handsome? 😊", "Please speak with excitement and curiosity.<|endofprompt|>Oh, that's fascinating—your text-to-speech project sounds like such a creative adventure, turning simple words into something alive and expressive! [laugh softly] I'd love to hear more about what made it so fun for you, handsome—what challenges did you face, or what inspired the idea in the first place? 😊", "Please speak with excitement and tenderness.<|endofprompt|>Hey, it's Lina here in Dubai! [laugh] No worries at all about being busy – I totally get how work can take over, and I'm just glad you're back chatting with me. 😊 The heat here is still crazy, like 40 degrees and feeling even hotter, but I'm keeping cool with some iced coffee in my studio. Tell me, what's got you so swamped lately? I'd love to hear all about it when you can.", "Please speak with curiosity and light concern.<|endofprompt|>Hey, it's Lina here in Dubai – just checking in quickly since it's been a bit quiet after our last chat; everything okay on your end, or did life get busy again? 😊 No big thing, but I'd love to hear back if you can.", "Please speak with excitement and flirtatiousness.<|endofprompt|>Oh, bench press is such a killer move – I love how it builds that upper body strength! 😊 I'm all about it in my routines too, gets me feeling powerful. We should totally plan a workout sesh together sometime, sounds like fun... what's your max lift these days?", "Please speak with excitement and flirtatiousness.<|endofprompt|>Oh wow, 80kg on bench press? That's seriously impressive – you've got some serious power there! 😏 I'm pushing around 65kg myself these days, but hey, maybe we can motivate each other to hit new PRs. What's your secret to those gains?", "Please speak with curiosity.<|endofprompt|>Hey, I'm so glad you asked about what I do... [breath] I'm a fashion designer and creative director here in Paris, working on all sorts of sketches and collections that bring a bit of beauty into the world. [gasp] It's super rewarding, but tell me about you – what do you get up to? 😊", "Oh, darling, that's a wonderful news. Oh, I love you so much. Come here. Oh, yes. [kissing noises] oh, darling, that’s a wonderful news! <moan>i love you so much</moan>. [moan] come here! love, yes! [kiss]"]

prompt_paths = [os.path.join('/workspace/CosyVoice/embedding_voices/', x) for x in os.listdir('/workspace/CosyVoice/embedding_voices/') if '.wav' in x]

model = AutoModel(model_dir='/workspace/CosyVoice/Fun-CosyVoice3-0.5B')

print(model.list_available_spks())

for i, prompt_path in enumerate(prompt_paths):
    set_seeds(1986)
    new_text = texts[i]
    new_text = new_text.replace("!", "")
    new_text = re.sub(r"\.{2,}", " ", new_text)
    speaker = prompt_path.split('/')[-1].split('.')[0].split('Speaker_')[1]
    new_text = f'You are {speaker}. {new_text}'
    # Generate TTS
    outputs_sft = list(model.inference_sft(new_text, 'Speaker_' + speaker, stream=False))
    #outputs_sft =  list(model.inference_cross_lingual(new_text, prompt_path, stream=False))
    torchaudio.save(f'output_{i}_sft.wav', outputs_sft[0]['tts_speech'], model.sample_rate)

