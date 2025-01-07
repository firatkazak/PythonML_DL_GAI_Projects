import tensorflow as tf  # TensorFlow kütüphanesini içe aktarıyor
import tensorflow_datasets as tfds  # TensorFlow veri setlerini yüklemek için kullanılan tensorflow_datasets'i içe aktarıyor

# Model kaydetme yolu
model_save_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/"

# IMDb inceleme verilerini yükler, %90'ı eğitim, %10'u doğrulama ve geri kalanı test için ayrılıyor
raw_train_ds, raw_val_ds, raw_test_ds = tfds.load(name="imdb_reviews", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True)

# İlk 3 incelemeyi ve etiketlerini (pozitif/negatif) ekrana yazdırıyor
for review, label in raw_train_ds.take(3):
    print(review.numpy().decode("utf-8"))  # İnceleme verisini (byte formatında) metne çevirip yazdırıyor
    print("Label: ", label.numpy())  # İncelemenin pozitif/negatif etiketini yazdırıyor

# Rastgeleliği kontrol etmek için sabit bir rastgelelik tohumu ayarlıyor
tf.random.set_seed(42)
# Eğitim veri setini karıştırıyor, 32'lik batch'ler halinde grupluyor ve önceden yükleme (prefetch) yapıyor
train_ds = raw_train_ds.shuffle(5000, seed=42).batch(32).prefetch(1)
# Doğrulama veri setini batch'ler halinde grupluyor ve önceden yükleme yapıyor
val_ds = raw_val_ds.batch(32).prefetch(1)
# Test veri setini batch'ler halinde grupluyor ve önceden yükleme yapıyor
test_ds = raw_test_ds.batch(32).prefetch(1)

# Metin verilerini sayısal vektörlere çevirmek için bir TextVectorization katmanı oluşturuyor
text_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=20000, output_mode="multi_hot")
# Eğitim verisindeki sadece inceleme metinlerini almak için map fonksiyonuyla dönüştürüyor
text_only_train_ds = train_ds.map(lambda review_text, sentiment_label: review_text)
# Vectorization katmanını, eğitim verisindeki metinlere adapte ediyor
text_vectorization_layer.adapt(text_only_train_ds)


# Veri setini işleyip metin verilerini sayısal hale getiriyor
def preprocess_dataset(dataset, vectorization_layer):
    # Her bir incelemeyi vectorization katmanından geçirip etiketle birlikte döndürüyor
    return dataset.map(lambda review_text, sentiment_label: (vectorization_layer(review_text), sentiment_label), num_parallel_calls=4)


# Eğitim, doğrulama ve test veri setlerini işleyip binary 1-gram vektörlere çeviriyor
binary_1gram_train_ds = preprocess_dataset(train_ds, text_vectorization_layer)
binary_1gram_val_ds = preprocess_dataset(val_ds, text_vectorization_layer)
binary_1gram_test_ds = preprocess_dataset(test_ds, text_vectorization_layer)

# İlk işlenmiş eğitim verisini yazdırıyor
for review, label in binary_1gram_train_ds.take(1):
    print(review[0])  # İşlenmiş metin (vektör) yazdırılıyor
    print("Label: ", label[0])  # Etiket yazdırılıyor


# Modeli oluşturuyor
def get_model(max_tokens=20000, hidden_dim=16):
    # Giriş katmanı, vektör boyutu max_tokens olan bir tensör alıyor
    input_layer = tf.keras.Input(shape=(max_tokens,))
    # İlk gizli katman, ReLU aktivasyon fonksiyonunu kullanarak 16 boyutlu bir Dense katman ekliyor
    dense_layer = tf.keras.layers.Dense(hidden_dim, activation="relu")(input_layer)
    # Dropout katmanı, aşırı öğrenmeyi önlemek için bazı bağlantıları rastgele kapatıyor (0.5 oranında)
    dropout_layer = tf.keras.layers.Dropout(0.5)(dense_layer)
    # Çıkış katmanı, sigmoid aktivasyonuyla tek bir çıkış (0 veya 1, yani binary sınıflandırma için) veriyor
    output_layer = tf.keras.layers.Dense(units=1, activation="sigmoid")(dropout_layer)
    # Modeli tanımlıyor (girişten çıkışa)
    model = tf.keras.Model(input_layer, output_layer)
    # Modeli, binary_crossentropy kaybı ve rmsprop optimizasyonuyla derliyor
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Binary 1-gram modelini oluşturuyor
model_binary_1gram = get_model()
# Modelin en iyi halini kaydeden bir ModelCheckpoint callback'i oluşturuyor
# Binary 1-gram modelini kaydetme
callbacks = [tf.keras.callbacks.ModelCheckpoint(model_save_path + "binary_1gram.keras", save_best_only=True)]
# Modeli eğitiyor, eğitim veri setini cache'leyip doğrulama verisiyle birlikte 10 epoch boyunca eğitiyor
model_binary_1gram.fit(binary_1gram_train_ds.cache(), validation_data=binary_1gram_val_ds, epochs=10, callbacks=callbacks)
# Binary 1-gram modelini yükleme
model_binary_1gram = tf.keras.models.load_model(model_save_path + "binary_1gram.keras")
# En iyi modeli yükleyip test seti üzerinde değerlendiriyor
print(f"Test acc: {model_binary_1gram.evaluate(binary_1gram_test_ds)[1]:.3f}")  # Test doğruluğunu yazdırıyor

# 2-gram (ikili kelime grupları) olacak şekilde vectorization katmanını değiştiriyor
text_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=20000, output_mode="multi_hot", ngrams=2)
# Yeni katmanı eğitim verilerine adapte ediyor
text_vectorization_layer.adapt(text_only_train_ds)

# Eğitim, doğrulama ve test veri setlerini işleyip binary 2-gram vektörlere çeviriyor
binary_2gram_train_ds = preprocess_dataset(train_ds, text_vectorization_layer)
binary_2gram_val_ds = preprocess_dataset(val_ds, text_vectorization_layer)
binary_2gram_test_ds = preprocess_dataset(test_ds, text_vectorization_layer)

# Binary 2-gram modeli oluşturuyor ve en iyi model checkpoint ile eğitiliyor
model_binary_2gram = get_model()
# Binary 2-gram modelini kaydetme
callbacks = [tf.keras.callbacks.ModelCheckpoint(model_save_path + "binary_2gram.keras", save_best_only=True)]
model_binary_2gram.fit(binary_2gram_train_ds.cache(), validation_data=binary_2gram_val_ds, epochs=10, callbacks=callbacks)

# Eğitilmiş model yüklenip test ediliyor
# Binary 2-gram modelini yükleme
model_binary_2gram = tf.keras.models.load_model(model_save_path + "binary_2gram.keras")

print(f"Test acc: {model_binary_2gram.evaluate(binary_2gram_test_ds)[1]:.3f}")  # Test doğruluğu yazdırılıyor

# TF-IDF (Term Frequency-Inverse Document Frequency) ile vectorization katmanını ayarlıyor
text_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=20000, output_mode="tf_idf", ngrams=2)
# Eğitim verilerine adapte ediyor
text_vectorization_layer.adapt(text_only_train_ds)

# Eğitim, doğrulama ve test veri setlerini işleyip tf-idf 2-gram vektörlere çeviriyor
tfidf_2gram_train_ds = preprocess_dataset(train_ds, text_vectorization_layer)
tfidf_2gram_val_ds = preprocess_dataset(val_ds, text_vectorization_layer)
tfidf_2gram_test_ds = preprocess_dataset(test_ds, text_vectorization_layer)

# TF-IDF 2-gram modeli oluşturuyor ve en iyi modeli checkpoint ile eğitiyor
model_tfidf_2gram = get_model()
# TF-IDF 2-gram modelini kaydetme
callbacks = [tf.keras.callbacks.ModelCheckpoint(model_save_path + "tfidf_2gram.keras", save_best_only=True)]
model_tfidf_2gram.fit(tfidf_2gram_train_ds.cache(), validation_data=tfidf_2gram_val_ds, epochs=10, callbacks=callbacks)
# Eğitilmiş model yüklenip test ediliyor
# TF-IDF 2-gram modelini yükleme
model_tfidf_2gram = tf.keras.models.load_model(model_save_path + "tfidf_2gram.keras")
print(f"Test acc: {model_tfidf_2gram.evaluate(tfidf_2gram_test_ds)[1]:.3f}")  # Test doğruluğu yazdırılıyor

# String giriş alacak şekilde bir giriş katmanı oluşturuyor
inputs = tf.keras.Input(shape=(1,), dtype="string")

# 2-gram multi-hot vektör üretimi için TextVectorization katmanı ekleniyor
text_vectorization_export_layer = tf.keras.layers.TextVectorization(max_tokens=20000, output_mode="multi_hot", ngrams=2)
# Eğitim verisine adapte ediliyor - BU SATIR EKLENDİ
text_vectorization_export_layer.adapt(text_only_train_ds)

# Girişten metin vektörizasyon katmanı geçiyor
x = text_vectorization_export_layer(inputs)
# TF-IDF modeli kullanılarak çıktı üretiliyor
outputs = model_tfidf_2gram(x)
# Girişten çıkışa kadar model oluşturuluyor
inference_model = tf.keras.Model(inputs, outputs)

# Örnek bir metin tensor'e dönüştürülüyor
text_data = tf.convert_to_tensor(["This movie is great. I liked it."])
# Modelle tahmin yapılıyor
predictions = inference_model(text_data)
# Tahmin sonucu yüzde cinsinden yazdırılıyor
print(f"{float(predictions[0] * 100):.2f} percent positive")

# Bu kodun amacı, IMDb film incelemelerini analiz eden bir metin sınıflandırma modelini eğitmek ve değerlendirmektir.
# Metin sınıflandırma problemleri için, kullanıcı incelemelerinin pozitif mi yoksa negatif mi olduğunu tahmin eden derin öğrenme modelleri oluşturulmaktadır.
#
# Bu kodda, metin verilerini sayısal bir formata dönüştürüp, bu verilerle bir model eğitiliyor ve performansı test ediliyor.
# Burada farklı vektörleştirme yöntemleri  kullanılarak metinler sayısal hale getiriliyor ve aynı yapıdaki modelle eğitilip performansları karşılaştırılıyor.
# Farklı vektörleştirme yöntemleri: (multi-hot, 2-gram, TF-IDF)

# Veri Hazırlama:
# Metin verilerini işlemek ve model için uygun formata getirmek (vektörleştirme).
# IMDb incelemeleri gibi doğal dil verileri üzerinde çalışıyorsun. Veriyi sayısal hale getirip modele nasıl verileceğini öğrenmelisin.

# TextVectorization Katmanı:
# Metinleri model için uygun hale getirmek için TextVectorization katmanının nasıl kullanıldığını öğrenmelisin.
# multi-hot, n-gram, ve TF-IDF gibi yöntemlerin nasıl uygulandığını anlamalısın.

# Model Eğitimi:
# Derin öğrenme modelini nasıl oluşturup, derleyip eğiteceğini öğrenmelisin.
# Girdi (metin vektörü) ve çıktı (pozitif/negatif sınıflar) arasındaki ilişkiyi modelleme sürecini anlamalısın.

# Modelin Performansını Değerlendirme:
# Modelin doğrulama ve test setleri üzerinde performansını ölçmek için doğruluk (accuracy) gibi metriklerin nasıl kullanıldığını öğrenmelisin.

# Callbacks ve Model Kaydetme:
# Eğitim sırasında en iyi modelin nasıl kaydedildiğini (ModelCheckpoint) ve bu modelin nasıl yüklendiğini öğrenmelisin.
