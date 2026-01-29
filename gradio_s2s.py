"""
Speech-to-Speech Conversational AI App
Pipeline: Speech → Whisper ASR → LLM → CosyVoice TTS (streaming)
Continuous conversation mode - no button clicks needed!
"""
import sys
import os
import gradio as gr
import torch
import torchaudio
import random
import numpy as np
import re
import whisper
from openai import OpenAI
from huggingface_hub import snapshot_download

# Required paths for CosyVoice
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2, AutoModel

# Speaker options
speaker_map = {
    'Speaker_Samantha', 'Speaker_Honey', 'Speaker_Autumn', 'Speaker_RachelMcAdams',
    'Speaker_AlexandraJames', 'Speaker_JoanAllen', 'Speaker_DuaSaleh', 'Speaker_NataliePortman',
    'Speaker_AbbieCornish', 'Speaker_ThaddeaGraham', 'Speaker_AimeeLouWood', 'Speaker_GillianAnderson',
    'Speaker_ScarletJohanson', 'Speaker_HannahGadsby', 'Speaker_EmmaMackey', 'Speaker_MimiKeene',
    'Speaker_DoreeneBlackstock', 'Speaker_LisaMcGrillis', 'Speaker_CateBlanchet',
    'Speaker_SharonDuncanBrewster', 'Speaker_SimoneAshley', 'Speaker_AnneMarieDuff',
    'Speaker_EvaGreen', 'Speaker_ChinenyeEzeudu', 'Speaker_JemimaKirke', 'Speaker_TanyaReynolds',
    'Speaker_SamanthaSpiro', 'Speaker_RakheeThakrar', 'Speaker_AnthonyLexa', 'Speaker_Despina',
    'Speaker_Aoede', 'Speaker_Autonoe', 'Speaker_Achernar', 'Speaker_Callirhoe', 'Speaker_Kore',
    'Speaker_Pulcherrima', 'Speaker_Vindemiatrix', 'Speaker_Leda', 'Speaker_Laomodeia',
    'Speaker_Sulafat', 'Speaker_Erinome', 'Speaker_Zephyr', 'Speaker_Phoenix', 'Speaker_Swiss'
}
speaker_choices = sorted(list(speaker_map))

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "api key here")

# System prompt for the LLM
DEFAULT_SYSTEM_PROMPT = """You are a friendly and helpful voice assistant. 
Keep your responses concise and conversational since they will be spoken aloud.
Aim for 1-3 sentences unless the user asks for more detail."""

# --- Initialize Models ---
print("Loading Whisper ASR model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded!")

print("Loading CosyVoice TTS model...")
model_path = snapshot_download(
    'identityailabs-com/CosyVoice-3-2026.01.23',
    token="huggingface token here"
)
tts_model = AutoModel(model_dir=model_path)
print("CosyVoice model loaded!")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def set_seeds(seed=1986):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_audio_for_whisper(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file using torchaudio (no ffmpeg needed)"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    
    audio = waveform.squeeze().numpy()
    return audio


def transcribe_audio(audio_path: str, language: str = None) -> str:
    """Transcribe audio using Whisper"""
    if audio_path is None:
        return ""
    
    audio = load_audio_for_whisper(audio_path)
    options = {"language": language} if language else {}
    result = whisper_model.transcribe(audio, **options)
    return result["text"].strip()


def get_llm_response(user_message: str, system_prompt: str, conversation_history: list) -> str:
    """Get response from LLM (OpenAI GPT)"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"


def process_text_for_tts(text: str) -> str:
    """Pre-process text for TTS"""
    if not text:
        return ""
    
    processed_text = text.replace("'", "'")
    processed_text = re.sub(r"\s{2,}", " ", processed_text).strip()
    
    if '<|endofprompt|>' not in processed_text:
        processed_text = "You are a helpful assistant.<|endofprompt|>" + processed_text
    
    return processed_text


def synthesize_speech_stream(text: str, speaker_name: str):
    """Generate speech using CosyVoice with streaming"""
    if not text or not speaker_name:
        return
    
    set_seeds(1986)
    processed_text = process_text_for_tts(text)
    
    print(f"Synthesizing (streaming): {repr(text[:100])}...")
    print(f"Speaker: {repr(speaker_name)}")
    
    for output in tts_model.inference_sft(processed_text, speaker_name, stream=True):
        audio_data = output['tts_speech'].numpy().flatten()
        yield (tts_model.sample_rate, audio_data)


def process_audio_input(
    audio_input,
    speaker_name: str,
    language: str,
    system_prompt: str,
    chat_history: list
):
    """
    Process audio input automatically when recording stops.
    Pipeline: ASR → LLM → TTS (streaming)
    """
    if audio_input is None:
        yield None, "Waiting for your voice...", chat_history
        return
    
    # Step 1: ASR
    yield None, "🎤 Listening...", chat_history
    
    lang = None if language == "Auto-detect" else language.lower()
    user_text = transcribe_audio(audio_input, lang)
    
    if not user_text:
        yield None, "❌ Couldn't hear that. Please try again.", chat_history
        return
    
    print(f"User said: {user_text}")
    yield None, f"📝 You: \"{user_text}\"\n\n🤔 Thinking...", chat_history
    
    # Step 2: LLM
    conversation_context = []
    for msg in chat_history:
        if msg["role"] in ["user", "assistant"]:
            conversation_context.append({"role": msg["role"], "content": msg["content"]})
    
    llm_response = get_llm_response(user_text, system_prompt, conversation_context)
    print(f"AI response: {llm_response}")
    
    # Update chat history
    new_history = chat_history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": llm_response}
    ]
    
    status = f"📝 You: \"{user_text}\"\n\n🤖 AI: \"{llm_response}\""
    
    # Step 3: TTS - Stream the response
    for audio_chunk in synthesize_speech_stream(llm_response, speaker_name):
        yield audio_chunk, status, new_history


def clear_conversation():
    """Clear the conversation history"""
    return [], None, "Conversation cleared. Start speaking!"


# --- Gradio UI ---
with gr.Blocks(
    title="Voice Chat AI",
    theme=gr.themes.Soft(),
    css="""
    .settings-panel { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .chat-area {
        min-height: 400px;
    }
    """
) as demo:
    
    # Conversation history state
    chat_history = gr.State([])
    
    gr.Markdown("""
    # 🎙️ Voice Chat AI
    
    **Just talk!** Select your AI voice below, then speak into the microphone.
    The AI will automatically listen, think, and respond.
    """)
    
    # Settings Panel (select speaker first)
    with gr.Group():
        gr.Markdown("### ⚙️ Setup - Select AI Voice First")
        with gr.Row():
            speaker_dropdown = gr.Dropdown(
                choices=speaker_choices,
                label="AI Voice",
                value="Speaker_Samantha" if "Speaker_Samantha" in speaker_choices else speaker_choices[0],
                scale=2
            )
            language_dropdown = gr.Dropdown(
                choices=["Auto-detect", "English", "Chinese", "Japanese", "Spanish", "French", "German"],
                label="Your Language",
                value="English",
                scale=1
            )
        
        with gr.Accordion("🎭 AI Personality (Advanced)", open=False):
            system_prompt = gr.Textbox(
                label="System Prompt",
                lines=3,
                value=DEFAULT_SYSTEM_PROMPT
            )
    
    gr.Markdown("---")
    
    # Main Chat Area
    with gr.Row():
        # Left - Input
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Speak Here")
            gr.Markdown("*Recording stops automatically when you pause*")
            
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Click to Record",
                show_label=False
            )
            
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
        
        # Right - Output
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 AI Response")
            
            output_audio = gr.Audio(
                label="AI Voice",
                interactive=False,
                autoplay=True,
                streaming=True,
                show_label=False
            )
            
            status_text = gr.Textbox(
                label="Current Exchange",
                interactive=False,
                lines=4,
                value="👋 Ready! Click the microphone and start talking."
            )
    
    # Chat History
    gr.Markdown("### 💬 Conversation History")
    chatbot = gr.Chatbot(
        label="Chat",
        height=250,
        type="messages",
        show_label=False
    )
    
    # Auto-trigger when audio recording stops
    audio_input.stop_recording(
        fn=process_audio_input,
        inputs=[audio_input, speaker_dropdown, language_dropdown, system_prompt, chat_history],
        outputs=[output_audio, status_text, chat_history]
    ).then(
        fn=lambda h: h,
        inputs=[chat_history],
        outputs=[chatbot]
    ).then(
        # Clear the audio input for next recording
        fn=lambda: None,
        outputs=[audio_input]
    )
    
    # Also trigger on change (for uploaded files)
    audio_input.change(
        fn=process_audio_input,
        inputs=[audio_input, speaker_dropdown, language_dropdown, system_prompt, chat_history],
        outputs=[output_audio, status_text, chat_history]
    ).then(
        fn=lambda h: h,
        inputs=[chat_history],
        outputs=[chatbot]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        outputs=[chat_history, chatbot, status_text]
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - Click the microphone, speak, then **pause** - recording stops automatically
    - Change AI voice anytime from the dropdown above
    - Clear chat to start a fresh conversation
    """)


if __name__ == "__main__":
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=1993,
        share=True
    )
