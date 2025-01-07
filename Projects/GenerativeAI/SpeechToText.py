from dotenv import load_dotenv
from openai import OpenAI
from pytubefix import YouTube
import openai

load_dotenv(override=True)

client = OpenAI()

# İlk transkripsiyon işlemi
audio_file = open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/openai_demo_speech.wav", "rb")
transcript = client.audio.transcriptions.create(model="whisper-1",
                                                file=audio_file,
                                                response_format="verbose_json",
                                                timestamp_granularities=["word"]
                                                )
print("Metin: ", transcript.text)  # transcript.text kullanılmalı
# Metin:  The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.


# YouTube videosu
video = "https://www.youtube.com/shorts/ae1b_CONZlY"  # video URL'sini buraya yazın

# Video verisini çekme
data = YouTube(video)

# Sadece ses akışını almak
audio = data.streams.get_audio_only()

# Ses dosyasını indir
audio.download(output_path='C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler',
               filename='openai_youtube_speech_demo.mp4'
               )

# Dosyayı açma
audio_file_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/openai_youtube_speech_demo.mp4"

# Yeni API'yi kullanarak transkripsiyonu oluşturma
with open(audio_file_path, "rb") as audio_file:
    transcript = openai.audio.transcriptions.create(  # Doğru metod kullanılıyor
        model="whisper-1",
        file=audio_file,
    )

# Transkripti kaydetmek için dosya açma
with open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/speech_to_text_transcript.txt", "w", encoding="utf-8") as f:
    f.write(transcript.text)  # Metni dosyaya yaz

# Transkripti yazdırma
print("Transkript dosyaya kaydedildi ve içerik şu şekildedir:")
print(transcript.text)  # transcript.text özelliği ile metni al
