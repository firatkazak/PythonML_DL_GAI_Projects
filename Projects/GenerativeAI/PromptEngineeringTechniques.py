from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()


def chat_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    response = client.chat.completions.create(model=model,
                                              messages=[{
                                                  "role": "user",
                                                  "content": prompt
                                              }],
                                              temperature=temperature,
                                              )
    print(response.choices[0].message.content)


prompt = """
my dog is sad --> 🐶 is 🥹
I love my wife --> 😀 ❤️ my wife
The boy love a girl --> 
"""
chat_completion(prompt)

prompt = "How much is 246 x 1235?"
chat_completion(prompt)

prompt = "How much is 246 x 1235? Let's think step by step."
chat_completion(prompt)

prompt = "Give a JSON output with 5 names of animals."
chat_completion(prompt)
