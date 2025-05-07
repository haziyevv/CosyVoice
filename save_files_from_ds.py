import os
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf  # Make sure you have this installed: pip install soundfile
from datasets import load_from_disk
import pdb

NAME = "Trixie"
# Load the dataset
ds = load_from_disk("filtered_pony_speech_female")
twilight_sparkle_data = ds.filter(lambda x: x['speaker'] == NAME)

# Create output directory
output_dir =NAME
os.makedirs(output_dir, exist_ok=True)

# Iterate and save files
for idx, sample in tqdm(enumerate(twilight_sparkle_data), total=len(twilight_sparkle_data)):
    style = sample['style'].lower()  # Ensure safe filenames
    noise = sample['noise'].lower()
    if noise == "very noisy":
        continue

    # Save audio
    audio_path = os.path.join(output_dir, f"{idx}_{style}_{noise}.wav")
    sf.write(audio_path, sample['audio']['array'], sample['audio']['sampling_rate'])

    # Save text
    text_path = os.path.join(output_dir, f"{idx}_{style}_{noise}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(sample['transcription'])