import tensorflow as tf  # TensorFlow kütüphanesini içe aktarıyoruz, makine öğrenmesi modelleri için gerekli.
from tensorflow.keras.layers import TextVectorization  # Keras'taki TextVectorization katmanını içe aktarıyoruz, metinleri sayısal hale getirmek için kullanılıyor.
import re  # Düzenli ifadeler (regular expressions) ile çalışmak için gerekli olan kütüphane.
import string  # Karakterler (özellikle noktalama işaretleri) ile çalışmak için kullanılan kütüphane.

# TextVectorization sınıfı, metin verilerini sayısal hale getirir. Adapt metoduyla veriyi işler ve bir kelime dizini oluşturur.
text_vectorization = TextVectorization()

# Veriyi oluşturuyoruz, bu veri metinlerden oluşuyor.
data = ["Bugün hava çok güzel",  # İlk cümle
        "Ali, Efe ve Ece çay içecek",  # İkinci cümle
        "Selam söyle"]  # Üçüncü cümle

# Veriyi TextVectorization nesnesine adapte ediyoruz, böylece veri işlenmeye hazır hale gelir.
text_vectorization.adapt(data)

# Kelime sözlüğünü elde ediyoruz, adapt işleminden sonra kelimelerin listesi.
sonuc1 = text_vectorization.get_vocabulary()
print(sonuc1)  # ['[UNK]', '', 'ali', 'bugün', 'çay', 'efe', 'ece', 'güzel', 'hava', ...]

# Veriyi vektörleştiriyoruz, her cümle sayısal bir forma çevriliyor.
sonuc2 = vectorized_text = text_vectorization(data)
print(sonuc2)  # Her kelimeye bir sayı atanmış, ve bu sayılar her cümlede sırayla yer alıyor.


# Metin verilerini standart hale getirmek için bir fonksiyon tanımlıyoruz.
def standardization_fn(string_tensor):
    lowercase = tf.strings.lower(string_tensor)  # Tüm metni küçük harfe çeviriyoruz.
    return tf.strings.regex_replace(  # Noktalama işaretlerini düzenli ifade kullanarak temizliyoruz.
        lowercase, f"[{re.escape(string.punctuation)}]", ""  # Noktalama işaretlerini boşluk ile değiştiriyoruz.
    )


# Cümleyi kelimelere ayırmak için bir fonksiyon tanımlıyoruz.
def split_fn(string_tensor):
    return tf.strings.split(string_tensor)  # Cümleyi kelimelere bölüyoruz.


# TextVectorization nesnesini oluştururken, metinleri küçük harfe çeviren ve kelimelere bölen fonksiyonları belirliyoruz.
text_vectorization = TextVectorization(
    standardize=standardization_fn,  # Standart hale getirme fonksiyonunu belirtiyoruz.
    split=split_fn  # Kelimelere ayırma fonksiyonunu belirtiyoruz.
)

# Veriyi yeniden adapte ediyoruz, bu sefer standart hale getirilmiş ve bölünmüş veriyi işlemek için.
text_vectorization.adapt(data)

# Yeni bir metin verisi tanımlıyoruz.
text = "bugün ece çok güzel"
# Bu metni vektörleştiriyoruz, kelimeler sayısal değerlere çevriliyor.
sonuc3 = text_vectorization(text)
print(sonuc3)  # Bu çıktı her kelimeyi bir numaraya dönüştürür.

# TensorFlow veri kümesi oluşturuyoruz, bu veri kümesi metinlerden oluşuyor.
text_dataset = tf.data.Dataset.from_tensor_slices(["kedi", "aslan", "yunus"])

# TextVectorization katmanını oluşturuyoruz.
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=5000,  # Maksimum 5000 farklı kelimeyi destekleyecek.
    output_sequence_length=4  # Çıktıdaki her metin maksimum 4 kelime uzunluğunda olacak.
)

# Veri kümesini vektörleştirme katmanına adapte ediyoruz.
vectorize_layer.adapt(text_dataset.batch(64))  # 64'lük batch'ler halinde veriyi işliyoruz.

# Kelime sözlüğünü elde ediyoruz.
sonuc4 = vectorize_layer.get_vocabulary()
print(sonuc4)  # ['[UNK]', 'aslan', 'kedi', 'yunus'] gibi bir çıktı verir, kelime sırası.

# Modeli tanımlıyoruz, metin verisi alacak bir Sequential model.
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),  # Girdi bir metin dizisi olacak.
    vectorize_layer  # Metni sayısal hale getirecek TextVectorization katmanını modele ekliyoruz.
])

# Girdi verilerini tanımlıyoruz.
input_data = tf.constant([["kedi kartal aslan"], ["fok yunus"]])
# Model ile tahmin yapıyoruz, her kelime sayısal değerlere dönüştürülüyor.
sonuc5 = model.predict(input_data)
print(sonuc5)  # Her kelime sırasıyla vektörlere dönüşecek.

# Açıklama:
# Bu kodda TextVectorization ile metinler vektör haline getirilerek model için işlenebilir hale getiriliyor.
# Özelleştirilmiş fonksiyonlar ile metin temizleniyor ve kelimelere ayrılıyor.
# Ayrıca Keras modeli oluşturulup, metin vektörleri üzerinden işlemler yapılıyor.
