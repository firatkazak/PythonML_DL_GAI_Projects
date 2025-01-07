from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI()

speech_file_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/text-to-speech-alloy_output.mp3"

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="Generative AI is a wonderful field to build to build something people will love."
)

# Düzeltilmiş streaming kullanımı
with open(speech_file_path, 'wb') as file:
    for chunk in response.iter_bytes():
        file.write(chunk)
