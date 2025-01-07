from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification, create_optimizer
import evaluate
import numpy as np
from transformers.keras_callbacks import KerasMetricCallback
import tensorflow as tf
import os

# Gerekliler klasörünün tam yolu
MODEL_SAVE_PATH = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/distilbert-base-uncased"

# IMDb veri setini yükleme
imdb = load_dataset("imdb").shuffle(seed=42)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Tokenizer oluşturma
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def train_and_save_model():
    # Veri setini işleme
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    # Veri düzenleyici
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Değerlendirme metriği
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Etiket haritaları
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # Model hiperparametreleri
    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

    # Optimizasyon ve öğrenme oranı planlayıcı
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    # Modeli yükleme ve yapılandırma
    model = TFAutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="distilbert/distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # TensorFlow veri kümeleri
    tf_train_set = model.prepare_tf_dataset(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_imdb["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    # Modeli derleme
    model.compile(optimizer=optimizer)

    # Değerlendirme geri çağırma
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

    # Modeli eğitme
    model.fit(
        x=tf_train_set,
        validation_data=tf_validation_set,
        epochs=num_epochs,
        callbacks=[metric_callback]
    )

    # Eğitilen modeli kaydetme
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)


# Modeli kontrol et ve yükle
if not os.path.exists(MODEL_SAVE_PATH):
    print("Model bulunamadı. Eğitiliyor ve kaydediliyor...")
    train_and_save_model()
else:
    print("Model zaten mevcut. Yükleniyor...")
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

# Tahmin yapma
text = "I have a strong affection for this movie. It's truly an incredible film."
inputs = tokenizer(text, return_tensors="tf")
logits = model(**inputs).logits
predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
result = model.config.id2label[predicted_class_id]
print("Sonuç:", result)
