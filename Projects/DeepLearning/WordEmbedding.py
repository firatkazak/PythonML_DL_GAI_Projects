import tensorflow as tf  # TensorFlow kütüphanesini içe aktarıyor
import numpy as np  # NumPy kütüphanesini içe aktarıyor, rastgele sayılar ve veri manipülasyonu için kullanılıyor

# Embedding katmanı oluşturuluyor
embedding_layer = tf.keras.layers.Embedding(input_dim=1000, output_dim=5)
# input_dim: Giriş veri kümesindeki farklı kelime sayısı (örneğin, 1000 farklı kelime)
# output_dim: Her kelimeyi temsil edecek olan vektörün boyutu (örneğin, 5 boyutlu vektör)

# tf.constant ile oluşturulan bir tensör embedding katmanına veriliyor
result = embedding_layer(tf.constant([1, 2, 3]))
# Bu işlem, 1, 2 ve 3 indeksli kelimelerin 5 boyutlu vektör temsillerini döndürüyor

# Embedding sonuçlarını yazdırıyor (1, 2 ve 3 indeksli kelimelerin 5 boyutlu temsilleri)
print(result.numpy())

# Bir Sequential model oluşturuluyor
model = tf.keras.Sequential()
# Sequential model, katmanların sırayla eklendiği bir model türüdür

# Modelin ilk katmanı olarak Embedding katmanı ekleniyor
model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
# input_dim: Girişteki toplam kelime sayısı yine 1000 olarak belirtiliyor
# output_dim: Her kelimenin embedding vektörü bu sefer 64 boyutlu olacak şekilde tanımlanıyor

# 32 satırlık, her biri 10 kelimelik rastgele bir integer array oluşturuluyor
# 1000 farklı kelime (0-999 arası) arasından rastgele kelimeler seçiliyor
input_array = np.random.randint(1000, size=(32, 10))
# input_array: 32 örnekten oluşan ve her örnekte 10 kelime içeren bir 2 boyutlu array

# Model 'rmsprop' optimizasyon algoritması ve 'mse' (Mean Squared Error) kayıp fonksiyonu ile derleniyor
model.compile("rmsprop", "mse")
# "rmsprop": Eğitim sırasında modeli optimize etmek için kullanılan optimizasyon algoritması
# "mse": Mean Squared Error (ortalama kare hata), kayıp fonksiyonu olarak kullanılıyor

# Modelden tahmin alınıyor, input_array modeli tahmin etmek için kullanılıyor
output_array = model.predict(input_array)

# Tahmin sonuçları (output_array) yazdırılıyor
print(output_array)

# Tahmin edilen çıktının boyutu yazdırılıyor
print(output_array.shape)
# Boyut, (32, 10, 64) olacak. 32 örnek, her biri 10 kelime içeriyor ve her kelimenin 64 boyutlu bir embedding'i var

# Giriş array'inin ilk örneği yazdırılıyor (ilk satır)
print(input_array[:1])

# İlk tahmin edilen çıktının boyutu yazdırılıyor
print(output_array[:1].shape)
# Boyut, (1, 10, 64) olacak, bu da ilk örneğin tahmin edilmiş embedding'leri anlamına geliyor

# Açıklama:
# Bu kod, Embedding katmanını kullanarak kelimeleri vektör temsillerine dönüştürmeyi ve bu vektör temsillerini bir sinir ağı modeline vermeyi gösteriyor.

# Öğrenilenler:
# Embedding Katmanı: Kelimeleri sabit boyutlu vektörlerle temsil etmek için kullanılır.
# Sequential Model: Bir modelde katmanların nasıl sırayla eklendiğini anlamalısın.
# Model Eğitimi ve Tahmin: Bir modelin nasıl derlendiği, çalıştırıldığı ve tahmin ettiği hakkında bilgi edinebilirsin.
