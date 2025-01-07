from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv(override=True)

save_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/Dall-E-Response.jpeg"

client = OpenAI()
response = client.images.generate(
    model="dall-e-3",
    prompt="Galatasaray wins the Champions League on its home field and celebrates with its fans.",
    size="1024x1792",
    quality="hd",
    n=1,
)

image_url = response.data[0].url
print("Image URL:", image_url)
# Resmi indir ve kaydet
image_data = requests.get(image_url).content
with open(save_path, "wb") as file:
    file.write(image_data)

print(f"Resim {save_path} yoluna kaydedildi.")
