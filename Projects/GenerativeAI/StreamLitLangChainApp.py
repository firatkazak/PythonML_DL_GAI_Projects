import streamlit as st
from langchain.llms import OpenAI

st.title("Basit Yapay Zeka Aplikasyonum")
open_ai_api_key = st.sidebar.text_input("OPEN AI API KEY'inizi Giriniz!", type="password")


def response(input_text):
    # Doğru parametrelerle OpenAI sınıfını oluşturuyoruz.
    llm = OpenAI(temperature=0.7, openai_api_key=open_ai_api_key)
    st.info(llm.predict(input_text))  # .predict metodu ile LLM'den yanıt alıyoruz.


with st.form("my_form"):
    text = st.text_area(label="Prompt'unuzu girin:", value="NLP hakkında 3 tane ipucu ver.")
    submitted = st.form_submit_button("Onayla")

if not open_ai_api_key.startswith("sk-"):
    st.warning("Lütfen Open AI API Key'inizi Giriniz!")
elif submitted:
    response(text)

# Projeyi ayağa kaldırmak için önce cd ile mevcut konumumuza geliyoruz;
# cd C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/GenerativeAI
# Daha sonra .py dosyamızı çalıştırıyoruz;
# streamlit run StreamLitLangChainApp.py

# Soruyu sorunca aşağıdaki 3 ipucunu verdi;
# NLP, doğal dil işleme anlamına gelir ve insan dilinin bilgisayarlar tarafından anlaşılması ve işlenmesi ile ilgilenir.
# NLP, dilbilim, bilgisayar bilimi ve yapay zeka gibi alanların kesişim noktasında yer alır.
# NLP, metin madenciliği, konuşma tanıma, duygu analizi ve dil çevirisi gibi uygulamaları kapsar ve günlük hayatta sıklıkla kullandığımız dijital asistanlar ve çeviri araçlarının temelini oluşturur.
