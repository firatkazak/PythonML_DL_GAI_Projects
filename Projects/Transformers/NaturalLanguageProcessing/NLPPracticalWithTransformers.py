from transformers import pipeline
import pandas as pd

# Duygu analizi için distilbert modelini yükleyip sınıflandırıcıyı oluşturuyoruz.
classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
text = "It's great to learn NLP for me"
print("Metnin Duygu Analizi: ", classifier(text))
# Metnin Duygu Analizi:  [{'label': 'POSITIVE', 'score': 0.999805748462677}]

outputs = classifier(text)
print("Pandas ile Duygu Analizi Çıktısı: ", pd.DataFrame(outputs))
# Pandas ile Duygu Analizi Çıktısı:        label     score
# 0  POSITIVE  0.999806

classifier = pipeline(task="zero-shot-classification", model="FacebookAI/roberta-large-mnli")
text = "This is a tutorial about Hugging Face"
labels = ["tech", "education", "business"]
outputs = classifier(text, labels)
print(pd.DataFrame(outputs))
#                                 sequence     labels    scores
# 0  This is a tutorial about Hugging Face  education  0.934516
# 1  This is a tutorial about Hugging Face       tech  0.058315
# 2  This is a tutorial about Hugging Face   business  0.007169

generator = pipeline(task="text-generation", model="openai-community/gpt2")
prompt = "This tutorial will walk you through how to "
outputs = generator(prompt, max_length=50)
print("Verilen prompt'tan üretilen 1. metin: ", outputs[0]["generated_text"])
# Verilen prompt'tan üretilen metin:  This tutorial will walk you through how to ikimizaka.com's ikimize-compose code for your company website in Javascript using the new ikimize-js-templates plugin. We'll use it to create

generator = pipeline(task="text-generation", model="distilbert/distilgpt2")
outputs = generator(prompt, max_length=50)
print("Verilen prompt'tan üretilen 2. metin: ", outputs[0]["generated_text"])
# Verilen prompt'tan üretilen 2. metin:  This tutorial will walk you through how to !! For the first time you will have this simple tutorial and you can actually enjoy the tutorial, but you can learn more on how to use these methods now!
# If you are interested in these tutorials you

ner = pipeline(task="ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
text = "My name is Fırat from Türkiye. Hugging Face is a nice platform."
outputs = ner(text)
print(pd.DataFrame(outputs))
#   entity_group     score     word  start  end
# 0          PER  0.983574    Fırat     11   16
# 1          ORG  0.969990        T     22   23
# 2          LOC  0.525651      ##ü     23   24
# 3          ORG  0.834833  ##rkiye     24   29

reader = pipeline(task="question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
result = reader(question="Hangi şehir başkenttir?", context="Ankara, Türkiye'nin başkentidir.")
print("Sorunun Cevabı: ", pd.DataFrame([result]))

summarizer = pipeline(task="summarization", model="google-t5/t5-small")

text = \
    """
    The 2024-2025 NBA season is shaping up to be one of the most exciting in recent history. Several teams have made significant moves in the offseason, strengthening their rosters in hopes of competing for the coveted NBA Championship. The reigning champions, the Denver Nuggets, are looking to defend their title after a historic performance in the 2023-2024 season, led by their star player, Nikola Jokić.
    
    In the Eastern Conference, the Milwaukee Bucks, led by two-time MVP Giannis Antetokounmpo, are a major contender. The Boston Celtics have also reloaded, bringing in fresh talent to complement their already formidable roster. The Miami Heat, with Jimmy Butler at the helm, are always a tough opponent in the playoffs, while the Philadelphia 76ers' performance hinges on the health and performance of their star, Joel Embiid.
    
    On the Western Conference side, the Golden State Warriors, with Stephen Curry and Klay Thompson, remain a top contender. The Los Angeles Lakers, led by LeBron James and Anthony Davis, are aiming for another deep playoff run. The Phoenix Suns, with newly acquired superstar Bradley Beal alongside Kevin Durant, are expected to make waves. The Dallas Mavericks, with Luka Dončić, continue to be a team to watch in the West.
    
    As the season progresses, fans will be looking for breakout performances from younger players, as the league continues to evolve with rising stars like Victor Wembanyama of the San Antonio Spurs. The race for MVP and Rookie of the Year is wide open, and every team is pushing hard for a spot in the playoffs.
    """

outputs = summarizer(text, max_length=60, clean_up_tokenization_spaces=True)
print("Özet: ", outputs[0]["summary_text"])
# Özet:  reigning champions, the Denver Nuggets, are looking to defend their title. in the Eastern Conference, the Milwaukee Bucks are a major contender. the Boston Celtics have also reloaded, bringing in fresh talent.

translator = pipeline(task="translation_en_to_fr", model="google-t5/t5-base")
text = "I love you."
outputs = translator(text, clean_up_tokenization_spaces=True)
print(outputs[0]["translation_text"])
# Je vous aime.
