import tensorflow as tf  # TensorFlow kütüphanesini içe aktarır.
import keras  # Keras'ı TensorFlow üzerinden içe aktarır (yüksek seviyeli API).

# Loading the Dataset (MNIST veri setini yükleme)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# MNIST veri setini yükler. 'train_images' eğitim resimlerini, 'train_labels' ise bu resimlerin etiketlerini içerir.
# 'test_images' ve 'test_labels' test verisi ve etiketlerdir.

# Data Preprocessing (Veri ön işleme)
train_labels = train_labels[:1000]  # Eğitim setinden ilk 1000 etiket seçilir.
test_labels = test_labels[:1000]  # Test setinden ilk 1000 etiket seçilir.

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
# Eğitim resimleri de ilk 1000'den seçilir. reshape(-1, 28 * 28) ile her 28x28 resim, 784 özellikli tek bir vektöre dönüştürülür.
# / 255.0 ile piksel değerleri 0-255 aralığından 0-1 aralığına normalize edilir.

test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# Aynı işlemler test resimleri için de yapılır: 1000 görüntü, düzleştirilir ve normalize edilir.

# Modeling (Model oluşturma)
def create_model():
    model = tf.keras.Sequential([  # Sequential modeli katmanların sıralı olarak eklenmesini sağlar.
        keras.layers.Input(shape=(784,)),  # Giriş katmanı, 784 boyutlu (28*28 piksellik düzleştirilmiş görüntüler).
        keras.layers.Dense(512, activation='relu'),  # İlk tam bağlı (dense) katman, 512 nöron içerir. Aktivasyon fonksiyonu ReLU.
        keras.layers.Dropout(0.2),  # Dropout katmanı, overfitting'i önlemek için rastgele %20 nöronu devre dışı bırakır.
        keras.layers.Dense(10)  # Çıkış katmanı, 10 nöron içerir (0-9 arası rakamları tahmin etmek için).
    ])

    # Modeli derleme (compile)
    model.compile(optimizer='adam',  # Adam optimizasyon algoritması kullanılır.
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    # Kaybı fonksiyonu SparseCategoricalCrossentropy: her sınıf için kaybı hesaplar (logits formatında çıktı).
    return model  # Modeli döndürür.


# Modeli oluştur
sonuc = create_model()
sonuc.summary()  # Modelin özetini (katmanları, parametreleri vb.) yazdırır.

# Parametrelerin Açıklamaları:
# train_images[:1000]: Eğitim verisinin sadece ilk 1000 örneğini alır.
# .reshape(-1, 28 * 28): Resimleri 28x28'den 784 boyutlu vektörlere dönüştürür. -1 ifadesi, bilinmeyen bir boyutun otomatik olarak hesaplanmasını sağlar.
# / 255.0: Piksel değerlerini 0-255 aralığından 0-1 aralığına normalize eder.
# keras.layers.Input(shape=(784,)): Giriş katmanı, düzleştirilmiş 784 boyutlu veriyi kabul eder.
# keras.layers.Dense(512, activation='relu'): 512 nöronlu tam bağlı bir katman, ReLU aktivasyon fonksiyonu ile kullanılır.
# keras.layers.Dropout(0.2): Dropout katmanı, overfitting'i önlemek için %20 oranında nöronları rastgele devre dışı bırakır.
# keras.layers.Dense(10): 10 nöronlu çıkış katmanı (0-9 arası rakamları tahmin etmek için).
# optimizer='adam': Adam optimizasyon algoritması kullanılır, genellikle iyi performans gösterir.
# loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True): Kategorik çapraz entropi kaybı fonksiyonu, logits formatında çıktılar için kullanılır.
