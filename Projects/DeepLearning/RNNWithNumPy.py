import numpy as np  # NumPy kütüphanesini içe aktar

# Zaman adımları, giriş ve çıkış özelliklerinin boyutlarını tanımla
timesteps = 100  # Zaman adımı sayısı (giriş dizisindeki toplam zaman adımı)
input_features = 32  # Giriş özellik sayısı (her zaman adımında kaç özellik var)
output_features = 64  # Çıkış özellik sayısı (her zaman adımında üretilen çıkış özellikleri)

# Rastgele giriş verilerini oluştur
inputs = np.random.random((timesteps, input_features))  # (timesteps, input_features) boyutlarında rastgele giriş verisi oluştur

# Başlangıç durumu (state) ve ağırlıkları tanımla
state_t = np.zeros((output_features,))  # Önceki durum için sıfırdan oluşan bir dizi oluştur (çıkış özellik boyutunda)

# Ağırlıkları ve bias'ı rastgele başlat
W = np.random.random((output_features, input_features))  # Girişten çıkışa olan ağırlıklar (output_features x input_features boyutunda)
U = np.random.random((output_features, output_features))  # Önceki durumdan çıkışa olan ağırlıklar (output_features x output_features boyutunda)
b = np.random.random((output_features,))  # Çıkış için bias terimi (output_features boyutunda)

# Çıkışların saklanacağı liste
successive_outputs = []  # Zaman adımlarında üretilen çıkışları saklamak için boş bir liste oluştur

# Her giriş için hesaplama yap
for input_t in inputs:  # Giriş verisindeki her zaman adımını dolaş
    # Yeni durumu hesapla
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    # Ağırlıklar (W ve U) ile giriş (input_t) ve önceki durum (state_t) arasında matris çarpımı yap
    # Sonuçlara bias (b) ekle ve ardından tanh aktivasyon fonksiyonunu uygula

    successive_outputs.append(output_t)  # Hesaplanan çıkışı listeye ekle
    state_t = output_t  # Mevcut durumu güncelle (sonuç çıkışı önceki duruma atandı)

# Çıkış dizilerini birleştir
final_output_sequence = np.stack(successive_outputs, axis=0)  # Çıkış listesini bir numpy dizisi haline getir (zaman adımı boyutunda)

print(final_output_sequence.shape)  # Sonuç dizisinin boyutunu yazdır

# Kodun İşlevi;

# Veri Oluşturma: 100 zaman adımı ve 32 giriş özelliği ile rastgele giriş verileri oluşturuluyor. Her bir zaman adımında, 64 çıkış özelliği hesaplanıyor.

# Hesaplama: Her bir giriş için W ve U matrisleri kullanılarak bir çıkış hesaplanıyor.
# Bu, RNN (Recurrent Neural Network) mantığını yansıtır, çünkü önceki durumu (state_t) kullanarak mevcut durumu hesaplıyor.

# Sonuçları Saklama: Her hesaplanan çıkış, successive_outputs listesine ekleniyor ve sonunda bu liste bir numpy dizisi haline getiriliyor.

# Sonuç Yazdırma: Sonuç dizisinin boyutu yazdırılıyor, bu da (100, 64) şeklinde olmalıdır, çünkü 100 zaman adımı ve her zaman adımında 64 çıkış özelliği var.

# Öğrenilecek Kavramlar
# Zaman Serisi Verileri: Zaman adımlarını ve giriş/çıkış özelliklerini tanımlamak.
# Matris Çarpımları: Ağırlık matrisleri ile giriş verileri ve önceki durumların çarpımını anlamak.
# Aktivasyon Fonksiyonları: Çıkışları hesaplamak için kullanılan tanh fonksiyonunun işlevi.
# RNN Temelleri: Tekrarlayan sinir ağları (RNN) mantığı ve durum güncelleme süreçleri.
