import tensorflow as tf  # TensorFlow, derin öğrenme modelleri için bir kütüphane.
import numpy as np  # NumPy, sayısal işlemler ve diziler üzerinde çalışmak için kullanılıyor.
import keras

# Loading data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/mnist.npz")  # MNIST veri setini yükler, el yazısı rakamların görüntüleri ve etiketleri.
print(x_train.shape, y_train.shape)  # Eğitim verisinin şekli (60000, 28, 28) ve etiketlerin şekli (60000,) yazdırılır.
print(x_test.shape, y_test.shape)  # Test verisinin şekli (10000, 28, 28) ve etiketlerin şekli (10000,) yazdırılır.

tf.random.set_seed(42)  # Rastgele işlemler için sabit seed, aynı sonuçları almak için kullanılır.

# Modeling
model = tf.keras.models.Sequential([  # Keras Sequential modeli, katmanların sıralı bir modelini oluşturur.
    tf.keras.layers.Input(shape=(28, 28)),  # Giriş katmanı, her resmin 28x28 boyutunda olduğunu belirler.
    tf.keras.layers.Flatten(),  # Resmi 1D vektöre dönüştürür (28x28 = 784 özellik).
    tf.keras.layers.Dense(units=512, activation='relu'),  # 512 nöronlu bir tam bağlı katman, ReLU aktivasyon fonksiyonu kullanır.
    tf.keras.layers.Dense(units=512, activation='relu'),  # Aynı şekilde, ikinci gizli katman.
    tf.keras.layers.Dense(units=10, activation='softmax')  # Çıkış katmanı, 10 sınıf için softmax ile sınıflandırma yapılır.
])

# Modeli derleme (compile) adımı
model.compile(loss='sparse_categorical_crossentropy',  # Kategorik çapraz entropi kaybı fonksiyonu, her sınıf için doğru tahmin yapmayı hedefler.
              optimizer='adam',  # Adam optimizasyon algoritması, öğrenme hızını otomatik olarak ayarlar.
              metrics=['accuracy']  # Eğitim süresince doğruluk metriği kullanılarak izlenir.
              )

# Training
model.fit(x_train,  # Eğitim verisi (resimler).
          y_train,  # Eğitim etiketleri (doğru rakam sınıfı).
          epochs=10  # 10 dönem boyunca model eğitilecek.
          )

# Normalization (Normalizasyon, veriyi 0-1 aralığına çeker)
x_train = (x_train / 255.0).astype(np.float32)  # Eğitim verisini normalize eder, piksel değerleri 0-255 arasındadır ve 0-1 aralığına çekilir.
x_test = (x_test / 255.0).astype(np.float32)  # Test verisi de aynı şekilde normalize edilir.

# Aynı modeli yeniden oluşturuyoruz, normalizasyon sonrası aynı yapıyı tekrar eğitim için kullanacağız.
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),  # Giriş boyutu yine 28x28 resimler.
    tf.keras.layers.Flatten(),  # Resimleri düzleştirir.
    tf.keras.layers.Dense(units=512, activation='relu'),  # 512 nöronlu tam bağlı katman.
    tf.keras.layers.Dense(units=512, activation='relu'),  # İkinci 512 nöronlu tam bağlı katman.
    tf.keras.layers.Dense(units=10, activation='softmax')  # 10 sınıf çıkışlı softmax katman (0-9 rakamları için).
])

# Modeli tekrar derliyoruz.
model.compile(loss='sparse_categorical_crossentropy',  # Kategorik çapraz entropi kaybı.
              optimizer='adam',  # Adam optimizasyon algoritması.
              metrics=['accuracy'])  # Doğruluk metriği.

# Modeli yeniden eğitiyoruz.
model.fit(x_train,  # Normalleştirilmiş eğitim verisi.
          y_train,  # Etiketler.
          epochs=10  # 10 dönem.
          )

# Test verisi ile modelin performansını değerlendiriyoruz.
model.evaluate(x_test, y_test)  # Test setindeki doğruluğu ve kaybı döndürür.

# Standartization (Standartizasyon, veriyi ortalamadan sapma ile normalize eder)
# Loading data sets again for standartization (Veri setini yeniden yüklüyoruz, bu kez standartizasyon için.)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # MNIST veri setini tekrar yükleriz.
mean = np.mean(x_train)  # Eğitim verisinin ortalamasını hesaplar.
std = np.std(x_train)  # Eğitim verisinin standart sapmasını hesaplar.
x_train = ((x_train - mean) / std).astype(np.float32)  # Her piksel değerini ortalamadan çıkarır ve standart sapmaya böler.

# Aynı modeli standartizasyon işlemi sonrası tekrar oluşturuyoruz.
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),  # Giriş boyutu 28x28.
    tf.keras.layers.Flatten(),  # Resmi düzleştirir.
    tf.keras.layers.Dense(units=512, activation='relu'),  # 512 nöronlu tam bağlı katman.
    tf.keras.layers.Dense(units=512, activation='relu'),  # İkinci gizli katman.
    tf.keras.layers.Dense(units=10, activation='softmax')  # 10 sınıf için softmax çıkış katmanı.
])

# Modeli derliyoruz.
model.compile(loss='sparse_categorical_crossentropy',  # Kategorik çapraz entropi kaybı.
              optimizer='adam',  # Adam optimizasyon algoritması.
              metrics=['accuracy'])  # Doğruluk metriği.

# Modeli standartize edilmiş veriyle yeniden eğitiyoruz.
model.fit(x_train,  # Standartize edilmiş eğitim verisi.
          y_train,  # Etiketler.
          epochs=10  # 10 dönem boyunca.
          )

# Kodun Özeti:
# MNIST Veri Seti: El yazısı rakamlardan oluşan bir veri seti (28x28 piksel boyutunda siyah-beyaz görüntüler).
# Model Yapısı: Bu model, 3 tam bağlı (Dense) katmandan oluşan basit bir sinir ağı (İki gizli katman, biri çıkış katmanı).
# Çıkış katmanında softmax kullanılıyor çünkü model rakamları sınıflandırmak için eğitiliyor (0-9 arası 10 sınıf).
# Normalizasyon ve Standartizasyon: Piksel değerleri normalde 0-255 aralığında. Normalizasyon işlemi ile bunlar 0-1 aralığına çekiliyor.
# Standartizasyon ise veriyi ortalama ve standart sapma ile normalize ediyor.
# Eğitim ve Değerlendirme: Model önce eğitiliyor (fit) ve ardından test verisi ile değerlendiriliyor (evaluate).
