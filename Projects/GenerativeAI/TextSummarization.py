from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/transcript.txt"

with open(path, "r") as f:
    transcript = f.read()

print(transcript)

client = OpenAI()

response = client.chat.completions.create(model="gpt-3.5-turbo",
                                          messages=[
                                              {"role": "system", "content": "You are a helpful assistant."},
                                              {"role": "user", "content": "Summarize the following text in the language of the text"},
                                              {"role": "assistant", "content": "Okay."},
                                              {"role": "user", "content": transcript}
                                          ]
                                          )

print(response.choices[0].message.content)

response = client.chat.completions.create(model="gpt-3.5-turbo",
                                          messages=[
                                              {"role": "system", "content": "You are a helpful assistant."},
                                              {"role": "user", "content": "Summarize the following text in the language of the text in Turkish"},
                                              {"role": "assistant", "content": "Okay."},
                                              {"role": "user", "content": transcript}
                                          ]
                                          )

print(response.choices[0].message.content)
