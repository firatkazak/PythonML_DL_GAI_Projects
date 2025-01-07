from datasets import Audio, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch import nn

speech_recognizer = pipeline(task="automatic-speech-recognition", framework="pt", model="facebook/wav2vec2-base-960h")
dataset = load_dataset(path="PolyAI/minds14", name="en-US", split="train", trust_remote_code=True)

print("Veri Setinin Özellikleri: ", dataset.features)
# Veri Setinin Özellikleri:  {'path': Value(dtype='string', id=None), 'audio': Audio(sampling_rate=8000, mono=True, decode=True, id=None), 'transcription': Value(dtype='string', id=None), 'english_transcription': Value(dtype='string', id=None), 'intent_class': ClassLabel(names=['abroad', 'address', 'app_error', 'atm_limit', 'balance', 'business_loan', 'card_issues', 'cash_deposit', 'direct_debit', 'freeze', 'high_value_payment', 'joint_account', 'latest_transactions', 'pay_bill'], id=None), 'lang_id': ClassLabel(names=['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN'], id=None)}

dataset = dataset.cast_column(column="audio",
                              feature=Audio(
                                  sampling_rate=speech_recognizer.feature_extractor.sampling_rate
                              )
                              )

result = speech_recognizer(dataset[:4]["audio"])
print("Ses Çıktısı: ", [d["text"] for d in result])
# Ses Çıktısı:  ['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE APSO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AN I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I FURN A JOINA COUT']

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

result2 = classifier("C'est un film génial. J'aime ça.")
print("Tahmin: ", result2)
# Tahmin:  [{'label': '5 stars', 'score': 0.7349996566772461}]

encoding = tokenizer("This film is nice. I liked it.")
print("Encoding: ", encoding)
# Encoding:  {'input_ids': [101, 10372, 10388, 10127, 24242, 119, 151, 11531, 10163, 10197, 119, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

batch = tokenizer(
    ["I like NLP. This is an awosome area", "We hope you don't hate it."],
    max_length=512,
    truncation=True,
    padding=True,
    return_tensors="pt")

print("Batch: ", batch)
# Batch:  {'input_ids': tensor([[  101,   151, 11531, 19848, 10373,   119, 10372, 10127, 10144, 37079,
#          20104, 10688, 10793,   102],
#         [  101, 11312, 18763, 10855, 11530,   112,   162, 39487, 10197,   119,
#            102,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])}

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

outputs = model(**batch)
predictions = nn.functional.softmax(outputs.logits, dim=-1)
print("Tahminler: ", predictions)
# Tahminler:  tensor([[0.2104, 0.1720, 0.2181, 0.1997, 0.1997],
#         [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)

save_directory = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/save_pretrained"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
