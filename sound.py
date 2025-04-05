import gradio as gr
from pydub import AudioSegment
import os
import whisper

model = whisper.load_model("base")
# Dummy transcription function
def transcribe_audio_file(filepath):
    result = model.transcribe(filepath)
    return result["text"]


def transcribe_audio(audio_file_path):
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Invalid or missing audio file"

    try:
        audio = AudioSegment.from_file(audio_file_path)
        audio.export("output.mp3", format="mp3")

        text = transcribe_audio_file("output.mp3")
        return text

    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Speak and get transcription")
    audio_input = gr.Audio(type="filepath", label="🎙️ Record Your Voice")
    output_text = gr.Textbox(label="📝 Transcribed Text")
    mic_button = gr.Button("Transcribe")

    mic_button.click(fn=transcribe_audio, inputs=audio_input, outputs=output_text)

demo.launch()
