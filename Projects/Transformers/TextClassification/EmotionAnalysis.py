from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import torch
import evaluate
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Model ve çıktı dizinini tanımla
MODEL_PATH = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/distilbert-emotion"
# Dataset yükleme
emotions = load_dataset("dair-ai/emotion")

# DataFrame dönüşümü ve görselleştirme
emotions.set_format(type="pandas")
df = emotions["train"][:]


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()
emotions.reset_format()

# Tokenizer ve model hazırlığı
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)


emotions_encoded = emotions.map(function=tokenize, batched=True, batch_size=None)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Model eğitimi için gerekli fonksiyonlar
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


# Model ve eğitim değişkenlerini tanımla
num_labels = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yükleme veya oluşturma kontrolü
try:
    print("Eğitilmiş model yükleniyor...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("Model başarıyla yüklendi. Yeni eğitim yapılmayacak.")

except (OSError, IOError):
    print("Önceden eğitilmiş model bulunamadı. Yeni model eğitimi başlatılıyor...")

    # Yeni model oluşturma
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_ckpt,
                                                               num_labels=num_labels
                                                               ).to(device)

    # Eğitim argümanlarının hazırlanması
    training_args = TrainingArguments(output_dir=MODEL_PATH,
                                      num_train_epochs=2,
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=64,
                                      weight_decay=0.01,
                                      eval_strategy="epoch",
                                      save_strategy="epoch",
                                      load_best_model_at_end=True,
                                      report_to="none"
                                      )

    # Trainer oluşturma
    trainer = Trainer(model=model,
                      args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=emotions_encoded["train"],
                      eval_dataset=emotions_encoded["validation"],
                      processing_class=tokenizer,
                      data_collator=data_collator
                      )

    # Model eğitimi
    trainer.train()

    # Model kaydetme
    trainer.save_model(MODEL_PATH)
    print("Model eğitimi tamamlandı ve kaydedildi.")

    # Değerlendirme ve görselleştirme
    preds_output = trainer.predict(emotions_encoded["validation"])
    print("Tahmin Metrikleri:", preds_output.metrics)

    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_valid = np.array(emotions_encoded["validation"]["label"])
    labels = emotions["train"].features["label"].names
    plot_confusion_matrix(y_preds, y_valid, labels)

# Test amaçlı tahmin
model.to(device)  # Modeli GPU'ya taşı (varsa)
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
custom_text = "I watched a movie yesterday. It was really good."
preds = classifier(custom_text, top_k=None)
preds_df = pd.DataFrame(preds)

# Sonuçları görselleştirme
labels = emotions["train"].features["label"].names
plt.bar(labels, 100 * preds_df["score"])
plt.title(f'"{custom_text}"')
plt.ylabel("Class probability (%)")
plt.show()
