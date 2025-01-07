from dotenv import load_dotenv
from openai import OpenAI
from typing import List

load_dotenv()
client = OpenAI()


def ask_chatgpt(messages):
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content


prompt_role = ("You are an asistant for journalist.\
               Your task is to write articles based on the facts given to you.\
               Consider to following as the instructions: TONE, LENGTH, STYLE and LANGUAGE.")


def new_generator(facts: List[str], tone: str, length_words: int, style: str, language: str):
    facts: ",".join(facts)
    prompt = f"{prompt_role} \
    FACTS : {facts}\
    TONE : {tone}\
    LENGTH : {length_words} words\
    STYLE : {style}\
    LANGUAGE : {language}"
    return ask_chatgpt([{"role": "user", "content": prompt}])


result_eng = new_generator(facts=["The sea is blue", "The tree is green"], tone="informal", length_words=100, style="blogpost", language="english")
print("Çıktı: ", result_eng)

result_tr = new_generator(facts=["Galatasaray", "Futbol"], tone="informal", length_words=100, style="blogpost", language="turkish")
print("Çıktı: ", result_tr)
