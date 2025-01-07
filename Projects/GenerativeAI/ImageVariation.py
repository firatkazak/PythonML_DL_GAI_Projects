from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests

load_dotenv(override=True)

# OpenAI istemcisini başlat
client = OpenAI()

# Görüntü varyasyonları oluştur
response = client.images.create_variation(
    image=open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/bird.png", "rb"),
    n=2,
    size="1024x1024",
)

# Varyasyon URL'lerini al
image_url_1 = response.data[0].url
image_url_2 = response.data[1].url

print(image_url_1)
print(image_url_2)

# Görüntüleri indir ve kaydet
save_path_1 = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/bird_variation_1.png"
save_path_2 = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/bird_variation_2.png"

# İlk resmi kaydet
image_data_1 = requests.get(image_url_1).content
with open(save_path_1, "wb") as file_1:
    file_1.write(image_data_1)

# İkinci resmi kaydet
image_data_2 = requests.get(image_url_2).content
with open(save_path_2, "wb") as file_2:
    file_2.write(image_data_2)

print(f"Varyasyon 1 {save_path_1} yoluna kaydedildi.")
print(f"Varyasyon 2 {save_path_2} yoluna kaydedildi.")

image = Image.open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/bird.png")
print(image)
print(image.format)
print(image.size)

width, height = 256, 256
image = image.resize((width, height))
print(image.size)
print(image)

byte_stream = BytesIO()
image.save(byte_stream, format="PNG")
byte_array = byte_stream.getvalue()
response = client.images.create_variation(image=byte_array,
                                          n=2,
                                          size="1024x1024",
                                          )

image_url_3 = response.data[0].url
print(image_url_3)
save_path_3 = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/bird_variation_3.png"
image_data_3 = requests.get(image_url_3).content
with open(save_path_3, "wb") as file_3:
    file_3.write(image_data_3)
print(f"Varyasyon 3 {save_path_3} yoluna kaydedildi.")
