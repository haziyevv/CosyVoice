#!/usr/bin/env python3
import os
import sys

# Add Matcha-TTS to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
matcha_path = os.path.join(current_dir, 'third_party', 'Matcha-TTS')
sys.path.append(matcha_path)

from cosyvoice.cli.cosyvoice import create_traced_model

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_traced_model.py <model_dir>")
        print("Example: python create_traced_model.py pretrained_models/CosyVoice2-0.5B")
        sys.exit(1)

    model_dir = sys.argv[1]
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist")
        sys.exit(1)

    try:
        traced_path = create_traced_model(model_dir)
        print(f"Successfully created traced model at: {traced_path}")
    except Exception as e:
        print(f"Error creating traced model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 