import tensorflow as tf  # TensorFlow kütüphanesi, derin öğrenme ve makine öğrenmesi modelleri için kullanılıyor.
import os  # İşletim sistemiyle etkileşim için gerekli kütüphane, dosya ve dizin işlemleri yapılacak.
import shutil  # Dosya ve dizinleri taşımak, silmek gibi işlemler için kullanılıyor.
import re  # Düzenli ifadeler (regex) ile metin işleme yapılacak.
import string  # Noktalama işaretlerini işlemekte kullanılacak.
import matplotlib.pyplot as plt  # Grafikler çizmek için kullanılan kütüphane.

# Veri setinin dizin yolunu oluşturuyoruz.
dataset_dir = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/datasets/aclImdb_v1/aclImdb"
train_dir = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/datasets/aclImdb_v1/aclImdb/train"
test_dir = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/datasets/aclImdb_v1/aclImdb/test"

# Örnek bir dosya seçiyoruz (pozitif yorum içeren bir dosya).
sample_file = os.path.join(train_dir, "pos/0_9.txt")
# Dosyanın içeriğini okuyup ekrana yazdırıyoruz.
with open(sample_file) as f:
    print(f.read())

# Eğitim verisini bir TensorFlow veri setine dönüştürüyoruz.
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,  # Eğitim verisinin bulunduğu dizin.
    batch_size=32,  # Veriler 32'lik gruplar (batch) halinde işlenecek.
    validation_split=0.2,  # Verilerin %20'si doğrulama (validation) için kullanılacak.
    subset="training",  # Eğitim verisini alıyoruz.
    seed=42,  # Rastgele bölme işlemi için sabit bir rastgelelik tohumu kullanıyoruz.
)
print(raw_train_ds)  # Veri setini yazdırıyoruz.

# İlk 3 yorumu ve etiketlerini ekrana yazdırıyoruz.
for text_batch, label_batch in raw_train_ds.take(1):  # 1 batch alıyoruz.
    for i in range(3):  # İlk 3 yorum ve etiketi ekrana yazdırıyoruz.
        print("Yorum:", text_batch.numpy()[i])
        print("Etiket:", label_batch.numpy()[i])

# Doğrulama veri setini oluşturuyoruz.
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,  # Eğitim verisinin bulunduğu dizin.
    batch_size=32,  # 32'lik batch'lerle işlenecek.
    validation_split=0.2,  # %20 doğrulama verisi.
    subset="validation",  # Doğrulama verisini alıyoruz.
    seed=42,  # Aynı rastgelelik tohumu.
)
print(raw_val_ds)  # Doğrulama veri setini yazdırıyoruz.

# Test veri setini oluşturuyoruz.
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,  # Test verisinin bulunduğu dizin.
    batch_size=32,  # 32'lik batch'lerle işlenecek.
)
print(raw_test_ds)  # Test veri setini yazdırıyoruz.


# Metinleri standart hale getirme fonksiyonu tanımlıyoruz.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)  # Metni küçük harflere çeviriyoruz.
    stripped_html = tf.strings.regex_replace(lowercase, "", "")  # HTML etiketlerini çıkarıyoruz (burada boş bırakılmış).
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", "")  # Noktalama işaretlerini temizliyoruz.


# Metni vektörleştirme katmanını oluşturuyoruz.
vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,  # Metinleri standartlaştırma işlemi.
    max_tokens=10000,  # Maksimum 10,000 farklı kelime kullanılacak.
    output_sequence_length=250  # Her metin 250 kelime uzunluğuna kadar kesilecek veya doldurulacak.
)

# Eğitim verisinden sadece metin kısmını alıyoruz.
train_text = raw_train_ds.map(lambda x, y: x)  # Eğitim verisindeki sadece metinleri alıyoruz.
vectorize_layer.adapt(train_text)  # Vektörleştirme katmanını bu metinlere adapte ediyoruz.


# Metinleri vektörleştiren bir fonksiyon tanımlıyoruz.
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)  # Metni bir boyutlu bir tensöre dönüştürüyoruz.
    return vectorize_layer(text), label  # Metin vektörleştirilmiş halde geri dönüyor.


# Eğitim verisinden bir batch alıyoruz.
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]

# İlk yorumu ve etiketini ekrana yazdırıyoruz.
print("Review: ", first_review)
print("Label: ", raw_train_ds.class_names[first_label])  # İlk yorumun etiketini sınıf adı olarak yazdırıyoruz.
print("Processed data: ", vectorize_text(first_review, first_label))  # İlk yorumu vektörleştiriyoruz.

# Vektörleştirme katmanında belirli indekslerde hangi kelimelerin olduğunu yazdırıyoruz.
print("345 --->", vectorize_layer.get_vocabulary()[345])
print("999 --->", vectorize_layer.get_vocabulary()[999])

# Eğitim, doğrulama ve test veri setlerini vektörleştirilmiş hale getiriyoruz.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Performans iyileştirmesi için cache ve prefetch kullanıyoruz.
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Modelin yapısını tanımlıyoruz.
max_features = 10000  # Vektörleştirilecek maksimum kelime sayısı.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1, 16),  # Her kelime için bir gömme (embedding) katmanı.
    tf.keras.layers.Dropout(0.2),  # Aşırı öğrenmeyi önlemek için dropout katmanı.
    tf.keras.layers.GlobalAveragePooling1D(),  # Tüm zaman adımları boyunca ortalama alınacak.
    tf.keras.layers.Dropout(0.2),  # Yine dropout uygulanıyor.
    tf.keras.layers.Dense(1),  # Tek bir çıkış nöronu, sınıflandırma için.
])

model.summary()  # Modelin özetini yazdırıyoruz.

# Modeli derliyoruz.
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # Binary sınıflandırma kaybı fonksiyonu.
    optimizer="adam",  # Adam optimizasyon algoritması.
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)]  # Doğruluk metriği.
)

# Modeli eğitim verileri üzerinde eğitiyoruz.
history = model.fit(train_ds, validation_data=val_ds, epochs=10)  # 10 epoch boyunca eğitim yapılacak.

# Test veri seti üzerinde modeli değerlendiriyoruz.
loss, accuracy = model.evaluate(test_ds)  # Kayıp ve doğruluğu hesaplıyoruz.
print("Loss: ", loss)  # Kayıp değerini yazdırıyoruz.
print("Accuracy: ", accuracy)  # Doğruluk değerini yazdırıyoruz.

# Eğitim geçmişinden elde edilen verileri alıyoruz.
history_dict = history.history
print(history_dict.keys())  # Eğitim boyunca kaydedilen metriklerin isimlerini yazdırıyoruz.

# Eğitim ve doğrulama doğruluğu ve kaybını grafiklerle gösteriyoruz.
acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

# Eğitim ve doğrulama kaybını çiziyoruz.
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validasyon loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluğunu çiziyoruz.
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validasyon acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Modeli dışa aktarıyoruz ve sigmoid aktivasyon fonksiyonu ekleyerek binary sınıflandırmayı tam bir model haline getiriyoruz.
export_model = tf.keras.Sequential([vectorize_layer, model, tf.keras.layers.Activation("sigmoid")])
# Export edilen modeli derliyoruz.
export_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=["accuracy"])

# Test veri setinde modelin performansını değerlendiriyoruz.
results = export_model.evaluate(raw_test_ds)
print("Evaluation Results:", results)  # Değerlendirme sonuçlarını yazdırıyoruz.

# AÇIKLAMA
# Bu kod, bir metin sınıflandırma modeli oluşturmak için TensorFlow ve Keras kullanıyor.
# IMDB film yorumlarını kullanarak, pozitif ve negatif yorumları ayırt eden bir sinir ağı eğitiyor.
# Metinleri sayısal vektörlere dönüştürmek için TextVectorization katmanını kullanıyor, ardından bir sinir ağı modeliyle bu vektörler üzerinde eğitim yapıyor.

# Bu koddan öğrenilmesi gerekenler:
# Veri seti indirme ve işleme: Veri setini dış kaynaklardan indirip, ön işleme yapmak.
# TextVectorization: Metinleri sayısal hale getirmek için nasıl kullanılacağı.
# Veri setleri ile çalışma: Eğitim, doğrulama ve test veri setlerini oluşturma ve bunları modele verme.
# Model oluşturma: Sinir ağlarıyla temel sınıflandırma modelleri kurma.
# Performans değerlendirme ve görselleştirme: Eğitim sürecindeki kayıplar ve doğruluk değerlerini görselleştirme.
# Model dışa aktarma: Eğitilen modeli nasıl dışa aktarılacağı ve kullanılacağı.
