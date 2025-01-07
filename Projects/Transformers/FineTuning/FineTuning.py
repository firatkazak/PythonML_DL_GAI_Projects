from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

base_model = "openai-community/gpt2-medium"

# Tokenizer ayarları
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                          padding_side="right",
                                          add_eos_token=True,
                                          )
tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization için ayarlar (gereksiz parametreler kaldırıldı)
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=False,
                                bnb_4bit_compute_dtype=torch.float16  # Daha yaygın olarak float16 kullanılır
                                )

# Modelin kaydedilip kaydedilmediğini kontrol et
model_save_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/model"

if os.path.exists(model_save_path):
    # Model kaydedildiyse, yükle
    print("Model daha önce kaydedildi, yükleniyor...")
    model = AutoModelForCausalLM.from_pretrained(model_save_path,
                                                 torch_dtype=torch.float16,  # Quantization config zaten modelde
                                                 device_map="auto")
else:
    # Model kaydedilmediyse, yeni yükle ve eğit
    print("Model daha önce kaydedilmemiş, yeni model yükleniyor...")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model,
                                                 torch_dtype=torch.float16,  # Quantization config zaten modelde
                                                 device_map="auto")

    # Modeli kaydet
    model.save_pretrained(model_save_path)
    print(f"Model kaydedildi: {model_save_path}")

# Dataset yükleme
dataset_name = "databricks/databricks-dolly-15k"

train_dataset = load_dataset(dataset_name, split="train[0:800]")
eval_dataset = load_dataset(dataset_name, split="train[800:1000]")

train_dataset.to_pandas()


# Prompt üretme fonksiyonu
def generate_prompt(sample):
    full_prompt = f"""<s>[INST]{sample['instruction']}
    {f"Here is some context: {sample['context']}" if len(sample["context"]) > 0 else None}
    [/INST] {sample['response']}</s>"""
    return {"text": full_prompt}


print("Veri Seti: ", train_dataset[0])
print("Generate Prompt: ", generate_prompt(train_dataset[0]))

# Veriyi map'leme
generated_train_dataset = train_dataset.map(generate_prompt, remove_columns=["instruction", "context", "response"])  # Doğru kolon isimleri
generated_val_dataset = eval_dataset.map(generate_prompt, remove_columns=["instruction", "context", "response"])  # Doğru kolon isimleri

print("Generated Train Dataset: ", generated_train_dataset[5]["text"])

tokenizer(generated_train_dataset[5]["text"])

model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)


# Eğitim parametrelerini yazdırma
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print_trainable_parameters(model)
print("Model: ", model)

# Eğitim argümanları
training_arguments = TrainingArguments(output_dir="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/results",
                                       num_train_epochs=1,
                                       per_device_train_batch_size=4,
                                       gradient_accumulation_steps=1,
                                       optim="paged_adamw_32bit",
                                       save_strategy="steps",
                                       save_steps=25,
                                       logging_steps=25,
                                       learning_rate=2e-4,
                                       weight_decay=0.001,
                                       max_steps=50,
                                       eval_strategy="steps",  # `evaluation_strategy` yerine `eval_strategy` kullanıldı
                                       eval_steps=25,
                                       do_eval=True,
                                       report_to="none",
                                       )

# Setting sft parameters
trainer = SFTTrainer(model=model,
                     processing_class=tokenizer,  # `tokenizer` yerine `processing_class` kullanıldı
                     args=training_arguments,
                     train_dataset=generated_train_dataset,
                     eval_dataset=generated_val_dataset,
                     peft_config=lora_config,
                     )

model.config.use_cache = False

# Eğitim başlatma, sadece model kaydedilmediyse eğitilecek
if not os.path.exists(model_save_path):
    trainer.train()

    # Eğitimi bitirdikten sonra, modeli kaydet
    model.save_pretrained(model_save_path)
    print(f"Model kaydedildi: {model_save_path}")
else:
    print("Model zaten kaydedilmiş, eğitim yapılmadı.")


# Eğitim tamamlandıktan sonra modelin doğruluğunu test etmek için örnek bir prompt verebiliriz.

def generate_response(model, tokenizer, prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Modelin cevabını al
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=100,  # Cevabın maksimum uzunluğunu belirleyebilirsin.
            num_beams=5,  # Daha doğru cevaplar için beam search
            no_repeat_ngram_size=2,  # Cevapta tekrarların olmasını engeller
            temperature=0.7  # Cevap çeşitliliğini belirler (0.7 makul bir değer)
        )

    # Tokenları metne çevir
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Örnek bir prompt
test_prompt = "What is the capital of France?"

# Modelden cevabı al
response = generate_response(model, tokenizer, test_prompt)

# Cevabı yazdır
print("Generated Response: ", response)
