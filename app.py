import gradio as gr
import yt_dlp as ydlp
from whispercpp import Whisper

def download_audio(youtube_url, output_folder='.'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_folder}/audio',
    }

    with ydlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


w = Whisper('tiny')


def process_general_transcription(transcription):
    formatted_transcription = []
    
    for line in transcription:
        if line.startswith('[') and line.endswith(']'):
            formatted_transcription.append(f'\n--- {line[1:-1].upper()} ---\n')
        else:
            formatted_transcription.append(line)

    transcript_str = "\n".join(formatted_transcription)
    
    return transcript_str
def transcribe_youtube(youtube_url):
    download_audio(youtube_url)
    result = w.transcribe("audio.wav")
    text = w.extract_text(result)
    return process_general_transcription(text)
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # CPP Whisperer - Transcribe YouTube Videos
    
    """)
    inp = gr.Textbox(placeholder="Insert YT Url here")
    result_button_transcribe = gr.Button('Transcribe')
    out = gr.Textbox()
    result_button_transcribe.click(transcribe_youtube, inputs = inp, outputs = out)


demo.launch()
