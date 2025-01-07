import tensorflow as tf
import os

# Dosya yolu tanımları
filepath = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/tinyshakespeare.txt"
model_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/my_shakespeare_model.keras"

# 1. Veri Yükleme
with open(filepath) as f:
    text = f.read()

print("tinyshakespeare.txt dosyasının ilk 100 karakteri: ", text[:100])

# Karakter setini inceleme
"".join(sorted(set(text.lower())))
len("".join(sorted(set(text.lower()))))

# 2. Metin Ön İşleme
text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize="lower")
text_vec_layer.adapt([text])

print("Text'in Shape'i: ", text_vec_layer([text]).shape)
encoded = text_vec_layer([text])[0]
print("Encoded Text: ", encoded)

encoded -= 2
n_tokens = text_vec_layer.vocabulary_size() - 2
print("Tokens: ", n_tokens)
dataset_size = len(encoded)
print("Dataset Size: ", dataset_size)


# Veri seti oluşturma fonksiyonu
def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# Örnek test
list(to_dataset(text_vec_layer(["I like"])[0], length=5))

# Veri setleri
length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True, seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)
test_set = to_dataset(encoded[1_060_000:], length=length)

# 3. Model Eğitimi veya Yükleme
if os.path.exists(model_path):
    print("Kaydedilmiş model bulundu, modeli yüklüyorum.")
    model = tf.keras.models.load_model(model_path)
else:
    print("Kaydedilmiş model bulunamadı, modeli eğitiyorum.")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dense(n_tokens, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])

    # Model Checkpoint
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                    monitor="val_accuracy",
                                                    save_best_only=True)

    # Modeli eğit
    model.fit(train_set, validation_data=valid_set, epochs=3, callbacks=[model_ckpt])

print("Model hazır.")

# Eğitimli modeli yeni bir Sequential model içinde kullanıyoruz
shakespeare_model = tf.keras.Sequential([
    text_vec_layer,  # Karakter bazında tokenizasyon
    tf.keras.layers.Lambda(lambda X: X - 2),  # Token değerlerini düzeltme
    model  # Eğitilen model
])

# Tahmin yapmak için bir metin veriyoruz
input_text = tf.constant(["To be or not to b"])
y_proba = shakespeare_model.predict(input_text)[0, -1]  # Son karakterin tahmin olasılıklarını alıyoruz

# En olası tahmini buluyoruz
y_pred = tf.argmax(y_proba).numpy()

# Tahmin edilen karakteri alıyoruz
predicted_char = text_vec_layer.get_vocabulary()[y_pred + 2]
print(f"Tahmin edilen karakter: {predicted_char}")  # Tahmin edilen karakter: e

# Rastgele örnekleme için logaritmik olasılık hesaplaması
log_probas = tf.math.log([[0.6, 0.3, 0.1]])
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=10)


# Bir sonraki karakteri tahmin eden fonksiyon
def next_char(text, temperature=1):
    # Girdiyi tf.constant formatına dönüştürün
    input_text = tf.constant([text])
    y_proba = shakespeare_model.predict(input_text)[0, -1:]  # Son karakterin tahmin olasılıkları
    rescaled_logits = tf.math.log(y_proba) / temperature  # Olasılıkları sıcaklık ile yeniden ölçekleme
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]  # Rastgele örnekleme
    return text_vec_layer.get_vocabulary()[char_id + 2]  # Tahmin edilen karakter


# Metni genişleten fonksiyon
def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)  # Yeni karakteri metne ekle
    return text


# Tahmin yapma
tf.random.set_seed(42)
print(extend_text(text="I like", temperature=1))
