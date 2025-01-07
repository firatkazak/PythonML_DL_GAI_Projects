from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset
import os
import torch

# Model dosyasının var olup olmadığını kontrol et
model_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/my_bert_model"
if os.path.exists(model_path):
    print("Eğitilmiş model yükleniyor...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    print("Model eğitiliyor...")
    # Dataset'i yükle
    dataset = load_dataset("rotten_tomatoes")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


    # Veriyi tokenize et
    def tokenize_dataset(dataset):
        return tokenizer(dataset["text"])


    dataset = dataset.map(tokenize_dataset, batched=True)

    # Modeli yükle
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

    # Data collator oluştur
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Eğitim ayarlarını oluştur
    training_args = TrainingArguments(output_dir=model_path,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8,
                                      num_train_epochs=2,
                                      report_to="none",
                                      )

    # Trainer ile modeli eğit
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=dataset["train"],
                      eval_dataset=dataset["test"],
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      )
    trainer.train()

    # Eğitilen modeli kaydet
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# Getting a text for prediction:
text = "I love NLP. It's fun to analyze the NLP tasks with Hugging Face"

# Preprocessing the text:
inputs = tokenizer(text, return_tensors="pt")
print("Input: ", inputs)
# Input:  {'input_ids': tensor([[  101,  1045,  2293, 17953,  2361,  1012,  2009,  1005,  1055,  4569,
#           2000, 17908,  1996, 17953,  2361,  8518,  2007, 17662,  2227,   102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

# Loading the model from the file:
model_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/my_bert_model/checkpoint-1000"
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=2)

# Calculating predictions:
with torch.no_grad():
    logits = model(**inputs).logits

# Looking the prediction:
predicted_class_id = logits.argmax().item()
print("Predicted class id: ", predicted_class_id)
# Predicted class id:  1
