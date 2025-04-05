##New project which i made in free time, can be expanded to add more features


import gradio as gr
import ollama
from gtts import gTTS
import tempfile
import os
import whisper

# Load Whisper model
whisper_model = whisper.load_model("base")

# Language mapping for TTS
lang_map = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
    "Chinese": "zh-CN", "Arabic": "ar", "Russian": "ru", "Japanese": "ja", "Korean": "ko",
    "Italian": "it", "Portuguese": "pt", "Bengali": "bn", "Tamil": "ta", "Telugu": "te",
    "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Odia": "or",
    "Punjabi": "pa", "Urdu": "ur", "Dutch": "nl", "Greek": "el", "Swedish": "sv",
    "Turkish": "tr", "Polish": "pl", "Czech": "cs", "Hungarian": "hu", "Romanian": "ro"
}

translated_text_global = ""

# Translate text using LLaMA via Ollama
def translate_text(input_text, source_lang, target_lang):
    global translated_text_global
    prompt = f"Translate the following sentence from {source_lang} to {target_lang}. Reply only in {target_lang}:\n\n{input_text}"
    try:
        response = ollama.chat(model="llama3", messages=[
            {"role": "user", "content": prompt}
        ])
        translated_text_global = response["message"]["content"].strip()
        return translated_text_global
    except Exception as e:
        return f"Error during translation: {e}"

# Transcribe audio to text using Whisper
def speech_to_text(audio):
    if not audio:
        return "No audio recorded."
    try:
        result = whisper_model.transcribe(audio)
        return result.get("text", "Transcription failed.")
    except Exception as e:
        return f"Error during transcription: {e}"

# Convert translated text to speech using gTTS
def speak_output(target_lang):
    global translated_text_global
    if not translated_text_global:
        return None, "Nothing to speak. Please translate first."
    lang_code = lang_map.get(target_lang, "en")
    try:
        tts = gTTS(translated_text_global, lang=lang_code)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name, "✅ Speaking complete!"
    except Exception as e:
        return None, f"Error in speaking: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## Your voice bot, now no language is a barrier 🎤🔊")

    with gr.Row():
        source_lang = gr.Dropdown(list(lang_map.keys()), label="Select Input Language", value="English")
        target_lang = gr.Dropdown(list(lang_map.keys()), label="Select Output Language", value="Hindi")

    with gr.Row():
        text_input = gr.Textbox(label="Enter Text to Translate")
        mic_input = gr.Audio(type="filepath", format="wav", label="🎤 Record Voice")

    translate_btn = gr.Button("Translate")
    speak_btn = gr.Button("Speak Output")

    translated_text = gr.Textbox(label="Translated Text")
    audio_output = gr.Audio(label="Audio Output", interactive=False)
    speak_status = gr.Textbox(label="Status")

    translate_btn.click(translate_text, inputs=[text_input, source_lang, target_lang], outputs=translated_text)
    speak_btn.click(speak_output, inputs=[target_lang], outputs=[audio_output, speak_status])
    mic_input.change(speech_to_text, inputs=[mic_input], outputs=text_input)

demo.launch()
