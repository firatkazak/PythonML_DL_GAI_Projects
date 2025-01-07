from datasets import load_dataset, DatasetDict
from collections import defaultdict, Counter
import pandas as pd
from transformers import AutoTokenizer, XLMRobertaForTokenClassification, TrainingArguments, DataCollatorForTokenClassification, Trainer
import torch
from seqeval.metrics import f1_score
import numpy as np
import os

# Model ve veri kaydetme dizini
save_directory = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/multilingual-xlm-roberta-for-ner"

langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059]

# Veri setlerini indirme ve kırpma
panx_ch = defaultdict(DatasetDict)
for lang, frac in zip(langs, fracs):
    ds = load_dataset(path="xtreme", name=f"PAN-X.{lang}")
    for split in ds:
        panx_ch[lang][split] = (
            ds[split].shuffle(seed=0).select(range(int(frac * ds[split].num_rows))))

# Etiketler
tags = panx_ch["de"]["train"].features["ner_tags"].feature
index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

# Tokenizer ve cihaz seçimi
xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yükleme veya eğitme
if os.path.exists(save_directory):
    print("Kaydedilmiş model bulundu. Yükleniyor...")
    model = XLMRobertaForTokenClassification.from_pretrained(save_directory).to(device)
else:
    print("Kaydedilmiş model bulunamadı. Model eğitiliyor...")


    def tokenize_and_align_labels(examples):
        tokenized_inputs = xlmr_tokenizer(examples["tokens"],
                                          truncation=True,
                                          is_split_into_words=True)
        labels = []
        for idx, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def encode_panx_dataset(corpus):
        return corpus.map(tokenize_and_align_labels,
                          batched=True,
                          remove_columns=["langs", "ner_tags", "tokens"])


    panx_de_encoded = encode_panx_dataset(panx_ch["de"])


    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []
        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(index2tag[preds[batch_idx][seq_idx]])

            labels_list.append(example_labels)
            preds_list.append(example_preds)

        return preds_list, labels_list


    def compute_metrics(eval_pred):
        y_pred, y_true = align_predictions(
            eval_pred.predictions, eval_pred.label_ids)
        return {"f1": f1_score(y_true, y_pred)}


    xlmr_model = XLMRobertaForTokenClassification.from_pretrained(
        pretrained_model_name_or_path="xlm-roberta-base",
        num_labels=tags.num_classes,
        id2label=index2tag,
        label2id=tag2index
    ).to(device)

    training_args = TrainingArguments(output_dir=save_directory,
                                      log_level="error",
                                      num_train_epochs=3,
                                      per_device_train_batch_size=24,
                                      per_device_eval_batch_size=24,
                                      evaluation_strategy="epoch",
                                      save_steps=1e6,
                                      weight_decay=0.01,
                                      logging_steps=len(panx_de_encoded["train"]) // 24,
                                      report_to="none")

    data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

    trainer = Trainer(model=xlmr_model,
                      args=training_args,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      train_dataset=panx_de_encoded["train"],
                      eval_dataset=panx_de_encoded["validation"],
                      tokenizer=xlmr_tokenizer,
                      )
    trainer.train()

    # Modeli kaydet
    trainer.save_model(save_directory)
    print(f"Model '{save_directory}' klasörüne kaydedildi.")


# Tahmin fonksiyonu
def tag_text(text, tags, model, tokenizer):
    tokens = tokenizer(text).tokens()
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    outputs = model(input_ids).logits
    predictions = torch.argmax(outputs, dim=-1)
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame(data=[tokens, preds], index=["Tokens", "Tags"])


# Örnek tahmin
text = "Tim Sparrow lebt in San Diego!"
print(tag_text(text, tags, model, xlmr_tokenizer))
