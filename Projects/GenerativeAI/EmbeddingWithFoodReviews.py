from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import tiktoken

load_dotenv(override=True)

df = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/fine_food_reviews_1k.csv", index_col=0)

print(df.head())
print(df.isnull().sum())

df["combined"] = ("Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip())
print(df.head(2))

top_n = 3
df = df.sort_values("Time").tail(top_n * 2)
df.drop(labels="Time", axis=1, inplace=True)
max_tokens = 8000

encoding = tiktoken.get_encoding("cl100k_base")
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)
print(len(df))

client = OpenAI()


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(
        input=[text], model=model
    ).data[0].embedding


df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model="text-embedding-ada-002"))

print(df.head())

df.to_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/fine_food_reviews_with_embeddings.csv", index=False)
