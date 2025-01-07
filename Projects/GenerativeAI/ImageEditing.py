from dotenv import load_dotenv
from openai import OpenAI
import requests

## https://labs.openai.com/editor
load_dotenv(override=True)
save_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/Dall-E-Response_Edit.png"
client = OpenAI()
response = client.images.edit(# model="dall-e-3", # Eğer dall e 2 kullanılacaksa parametrede belirt, 3 kullanacaksa belirtme hata veriyor.
                              image=open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/sunlit_lounge.png", "rb"),
                              mask=open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/sunlit-mask.png", "rb"),
                              prompt="A sunlit indoor lounge area with pool containing a flamingo.",
                              n=1,
                              size="1024x1024",
                              )
image_url = response.data[0].url
print(image_url)

# Resmi indir ve kaydet
image_data = requests.get(image_url).content
with open(save_path, "wb") as file:
    file.write(image_data)

print(f"Resim {save_path} yoluna kaydedildi.")
