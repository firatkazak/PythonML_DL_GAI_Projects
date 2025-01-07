from sklearn.datasets import load_sample_images  # sklearn'den örnek resim verileri yüklemek için kullanılır.
import tensorflow as tf  # TensorFlow kütüphanesi sinir ağları ve makine öğrenmesi işlemleri için kullanılır.

# Loading images
images = load_sample_images()["images"]  # Örnek resim verisetini yükleyip 'images' anahtarından resimleri alır.
print(images[0].shape)  # İlk resmin boyutunu ekrana yazdırır (yükseklik, genişlik, kanal sayısı).

# Data Preprocessing (Veri Ön İşleme)
images = tf.keras.layers.CenterCrop(height=80, width=120)(images)  # Resimlerin ortasını 80x120 boyutlarında kırpar.
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)  # Resim değerlerini 0-1 aralığına ölçekler (normalizasyon).
print(images.shape)  # Kırpılmış ve ölçeklenmiş resimlerin boyutunu ekrana yazdırır.

# Applying filters (Filtreler uygulama)
conv_layer = tf.keras.layers.Conv2D(  # 2D evrişim katmanı oluşturur.
    filters=32,  # 32 tane evrişim filtresi kullanır.
    kernel_size=7  # 7x7 boyutunda çekirdek (kernel) kullanır.
)
fmaps = conv_layer(images)  # Resimlere evrişim katmanını uygular ve çıktı özelliğini alır.
print(fmaps.shape)  # Özellik haritalarının boyutunu yazdırır.

# Adding the padding (Dolgu ekleme)
conv_layer = tf.keras.layers.Conv2D(  # 2D evrişim katmanı, dolgu ekleyerek yeniden tanımlanıyor.
    filters=32,  # Yine 32 evrişim filtresi kullanır.
    kernel_size=7,  # 7x7 boyutunda çekirdek kullanır.
    padding="same"  # 'same' doldurma ile, çıktı resmin boyutu giriş resmininkiyle aynı kalır.
)
fmaps = conv_layer(images)  # Resimlere yeni tanımlanan evrişim katmanını uygular.
print(fmaps.shape)  # Yeni özellik haritalarının boyutunu yazdırır.

# Exporing kernels and biases (Çekirdek ve kayma terimlerini inceleme)
kernels, biases = conv_layer.get_weights()  # Evrişim katmanındaki çekirdek ağırlıkları ve kayma (bias) değerlerini alır.
print(kernels.shape)  # Çekirdeklerin boyutunu yazdırır (örneğin 7x7xkanal_sayısıxfiltre_sayısı).
print(biases.shape)  # Kayma terimlerinin boyutunu yazdırır (32, yani her filtre için bir bias).

# Applying the Pooling (Havuzlama uygulama)
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)  # 2x2 boyutunda maksimum havuzlama katmanı oluşturur.
output = max_pool(images)  # Havuzlama işlemini resimlere uygular (özellik haritalarındaki bilgileri sıkıştırır).
print(output.shape)  # Havuzlanmış çıktı boyutunu yazdırır.

# Global Average Pooling (Küresel Ortalama Havuzlama)
global_avg_pool = tf.keras.layers.GlobalAvgPool2D()  # 2D küresel ortalama havuzlama katmanı oluşturur.
print(global_avg_pool(images))  # Resimler üzerinde küresel ortalama havuzlama uygular ve son çıktıyı yazdırır.

# Bu kodda TensorFlow kullanılarak bir dizi evrişim (convolution) ve havuzlama (pooling) işlemi yapılıyor.
# Veriler sklearn'den alınan örnek resimlerdir ve evrişim filtreleri uygulanarak özellik haritaları çıkarılıyor.
# Son olarak da havuzlama işlemleri yapılarak boyutlar küçültülüyor ve çıktı özellikleri elde ediliyor.
