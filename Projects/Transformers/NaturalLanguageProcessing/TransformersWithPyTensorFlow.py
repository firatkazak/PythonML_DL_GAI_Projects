from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification
from datasets import load_dataset
from tf_keras.optimizers import Adam
import os
from tensorflow import math

# Veritabanı yükleniyor
dataset = load_dataset("rotten_tomatoes")

# Model dosyasının zaten eğitilip eğitilmediğini kontrol et
model_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/distilbert_model"

# Tokenizer'ı her durumda yükle
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

if not os.path.exists(model_path):
    print("Eğitim başlatılıyor...")


    # Tokenize fonksiyonu
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)


    # Dataset üzerinde ön işleme
    dataset = dataset.map(preprocess_function, batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Modeli yükle
    my_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

    # TensorFlow datasetleri hazırlama
    tf_train_set = my_model.prepare_tf_dataset(dataset["train"],
                                               shuffle=True,
                                               batch_size=16,
                                               collate_fn=data_collator,
                                               )

    tf_validation_set = my_model.prepare_tf_dataset(dataset["validation"],
                                                    shuffle=False,
                                                    batch_size=16,
                                                    collate_fn=data_collator,
                                                    )

    # Modeli compile et
    my_model.compile(optimizer=Adam(3e-5))

    # Modeli eğit
    my_model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2)

    # Eğitilmiş modeli kaydet
    my_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Model eğitim tamamlandı ve kaydedildi.")
else:
    print("Model zaten eğitilmiş, tahmin işlemine geçiliyor...")

    # Eğitilmiş modeli yükle
    my_model = TFAutoModelForSequenceClassification.from_pretrained(model_path)

# Burada modeli yükleyip tahmin yapma kısmı yazılabilir.
text = "I love NLP. It's fun to analyze the NLP tasks with Hugging Face."

# Tokenize edilmiş metni hazırlayın
tokenized_text = tokenizer(text, return_tensors="tf")

# Tahmin yapma
logits = my_model(**tokenized_text).logits
result = int(math.argmax(logits, axis=-1)[0])
print("Sonuç: ", result)  # Sonuç:  1
# Özet. Sonuç 1 yani olumlu döndü. Yazdığımız texti olumlu buldu.
