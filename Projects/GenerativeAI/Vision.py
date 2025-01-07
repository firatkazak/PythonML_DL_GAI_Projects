from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        "detail": "high",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
# The image features a wooden path or boardwalk leading through a lush green landscape. The path appears to be surrounded by tall grasses and foliage, with a clear blue sky above and some clouds scattered throughout. The scene conveys a serene and natural environment, likely depicting a wetland or marsh area.
