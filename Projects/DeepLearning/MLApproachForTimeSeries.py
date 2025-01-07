import tensorflow as tf  # TensorFlow, makine öğrenmesi kütüphanesi
import numpy as np  # NumPy, bilimsel hesaplamalar için kullanılır
import matplotlib.pyplot as plt  # Matplotlib, veri görselleştirme için kullanılır

# TensorFlow dataset oluşturma, bir aralık belirlenir (0'dan 10'a kadar)
dataset = tf.data.Dataset.range(10)  # 0'dan 9'a kadar bir veri seti oluşturur
# Dataset, pencereleme (windowing) işlemi yapar; 5 elemanlık pencereler oluşturur
dataset = dataset.window(5, shift=1, drop_remainder=True)  # Her bir pencere 5 elemandan oluşur, 1 birim kaydırılır, eksik eleman olursa atılır
# flat_map, her bir pencereyi alır ve bir dizi (batch) halinde geri döner
dataset = dataset.flat_map(lambda window: window.batch(5))  # Her pencereyi bir batch'e (dizi) çevirir

# Veri setindeki her pencereyi numpy dizisi olarak yazdırır
for window in dataset:
    print(window.numpy())  # Her pencereyi yazdırır

# Etiketleri oluşturma (labeling)
dataset = tf.data.Dataset.range(10)  # 0'dan 9'a kadar bir veri seti oluşturur
dataset = dataset.window(5, shift=1, drop_remainder=True)  # 5 elemanlık pencereler oluşturur
dataset = dataset.flat_map(lambda window: window.batch(5))  # Pencereleri batch'lere dönüştürür
# Her bir pencerenin son elemanını etiket (y), geri kalanını girdi (x) olarak ayırır
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))  # Son eleman y (hedef), ilk 4 eleman x (girdi)

# Girdi (x) ve etiket (y) değerlerini numpy dizisi olarak yazdırır
for x, y in dataset:
    print(x.numpy(), y.numpy())

# Veri setini optimize etme (shuffle, batch ve prefetch işlemleri)
dataset = dataset.shuffle(buffer_size=10)  # Veriyi karıştırır, buffer_size ile bellek kullanımı optimize edilir
dataset = dataset.batch(2).prefetch(1)  # Verileri 2'lik batch'ler halinde alır ve 1 batch önceden yükler (prefetch)
for x, y in dataset:
    print("x = ", x.numpy())  # x giriş verilerini yazdırır
    print("y = ", y.numpy())  # y etiket verilerini yazdırır

# Yapay bir zaman serisi oluşturma

# Zaman verisi için bir eğilim (trend) fonksiyonu
def trend(time, slope=0):  # time: Zaman dizisi, slope: eğim değeri
    return slope * time  # Eğilim (trend) çizgisi oluşturur

# Mevsimsel desen oluşturma fonksiyonu
def seasonal_pattern(season_time):  # season_time: Bir mevsim içindeki zaman
    return np.where(season_time < 0.4,  # Eğer mevsim zamanı 0.4'ten küçükse
                    np.cos(season_time * 2 * np.pi),  # Kosinüs fonksiyonu kullan
                    1 / np.exp(3 * season_time))  # Değilse azalan bir fonksiyon kullan

# Mevsimsel etkileri ekleyen fonksiyon
def seasonality(time, period, amplitude=1, phase=0):  # time: Zaman dizisi, period: Mevsim periyodu, amplitude: dalga yüksekliği, phase: faz kayması
    season_time = ((time + phase) % period) / period  # Mevsimsel zaman dilimi hesaplanır
    return amplitude * seasonal_pattern(season_time)  # Mevsimsel desen ve amplitüd çarpımı

# Gürültü ekleyen fonksiyon
def noise(time, noise_level=1, seed=None):  # time: Zaman dizisi, noise_level: Gürültü seviyesi, seed: Rastgele tohum değeri
    rnd = np.random.RandomState(seed)  # Rastgele sayı üretici oluşturur
    return rnd.randn(len(time)) * noise_level  # Zaman serisine rastgele gürültü ekler

# 4 yıllık zaman dizisi oluşturuluyor (365 gün x 4)
time = np.arange(4 * 365 + 1, dtype="float32")

baseline = 10  # Başlangıç değeri
series = trend(time, .05)  # Zaman serisine küçük bir eğilim eklenir
amplitude = 15  # Mevsimsel genlik (dalga yüksekliği)
slope = 0.09  # Eğilim eğimi
noise_level = 6  # Gürültü seviyesi

# Zaman serisine eğilim, mevsimsellik ve gürültü eklenir
series = baseline + trend(time, slope) + seasonality(time, period=365,
                                                     amplitude=amplitude)
series += noise(time, noise_level, seed=42)

# Zaman serisi verisini pencereleme (windowing) fonksiyonu
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):  # series: Zaman serisi, window_size: pencere boyutu, batch_size: Batch boyutu, shuffle_buffer: Karıştırma için buffer boyutu
    dataset = tf.data.Dataset.from_tensor_slices(series)  # Veriyi TensorFlow Dataset'e dönüştürür
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)  # window_size + 1 boyutunda pencere oluşturur, her seferinde 1 birim kaydırır
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))  # Pencereleri batch'e dönüştürür
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))  # Veriyi karıştırır ve son elemanı etiket (y), diğerlerini girdi (x) olarak ayırır
    dataset = dataset.batch(batch_size).prefetch(1)  # Verileri batch'ler halinde alır ve önceden yükler
    return dataset  # Dataset'i döndürür

# Dataset'i bölme işlemi
split_time = 1000  # Veriyi bölme noktası
time_train = time[:split_time]  # Eğitim zamanı
x_train = series[:split_time]  # Eğitim verisi
time_valid = time[split_time:]  # Doğrulama zamanı
x_valid = series[split_time:]  # Doğrulama verisi

# Zaman serisi çizdirme fonksiyonu
def plot_series(time, series, format="-", start=0, end=None):  # time: Zaman dizisi, series: Değerler, format: çizim formatı, start: Başlangıç noktası, end: Bitiş noktası
    plt.plot(time[start:end], series[start:end], format)  # Zaman serisi çizdirilir
    plt.xlabel("Time")  # X ekseni etiketi
    plt.ylabel("Value")  # Y ekseni etiketi
    plt.grid(True)  # Izgaralı görünüm

# Zaman serisi grafiği
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)

window_size = 20  # Pencere boyutu
batch_size = 32  # Batch boyutu
shuffle_buffer_size = 1000  # Karıştırma için buffer boyutu

# Eğitim verisini pencereleme (windowed_dataset) ile işliyoruz
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# Dataset'teki bir batch'i yazdırıyoruz
for feature, label in dataset.take(1):  # 1 batch alır
    print(feature[:1])  # İlk girdiyi yazdırır
    print(label[:1])  # İlk etiketi yazdırır

# Model oluşturma
model = tf.keras.models.Sequential([  # Sequential model, katmanlar sıralı şekilde eklenir
    tf.keras.layers.Input(shape=[window_size]),  # Girdi boyutu pencere boyutuna eşit
    tf.keras.layers.Dense(10, activation="relu"),  # 10 nöronlu, ReLU aktivasyon fonksiyonlu tam bağlı katman
    tf.keras.layers.Dense(10, activation="relu"),  # İkinci tam bağlı katman
    tf.keras.layers.Dense(1)  # Çıkış katmanı, 1 nöron (tek değer tahmini)
])

# Model derleme
model.compile(loss="mse",  # Kayıp fonksiyonu: Ortalama Kare Hata (MSE)
              optimizer=tf.keras.optimizers.Adam())  # Optimizer: Adam optimizasyon algoritması

# Modeli eğitiyoruz
dataset = dataset.repeat()  # Dataset'i sürekli tekrarlar
model.fit(dataset, epochs=100, steps_per_epoch=200)  # Modeli 100 epoch boyunca, her epoch'ta 200 adım olacak şekilde eğitiyoruz

# Bir tahmin yapıyoruz
print(series[1020])  # 1020. zaman anındaki gerçek değeri yazdırır
print(model.predict(series[1000:1020][np.newaxis]))  # 1000-1020 aralığındaki değerleri kullanarak tahmin yapar

# Tüm zaman serisi için tahmin yapma
forecast = []
for time in range(len(series) - window_size):  # Zaman serisinin her bir penceresi için
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))  # Model tahmini yapar ve listeye ekler

len(forecast)  # Tahmin sayısı

# Doğrulama setindeki tahminleri alıyoruz
forecast = forecast[split_time - window_size:]  # Eğitim setinden sonrasını alıyoruz
results = np.array(forecast)[:, 0, 0]  # Tahminleri numpy dizisine çeviriyoruz

# Tahmin ve gerçek değerleri çizdiriyoruz
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)  # Gerçek değerler
plot_series(time_valid, results)  # Tahmin edilen değerler

# Ortalama Kare Hata (MSE) hesaplama
mse = tf.reduce_mean(tf.square(x_valid - results))  # MSE hesaplanır
print("Mean Squared Error:", mse.numpy())  # MSE değeri yazdırılır

# Ortalama Mutlak Hata (MAE) hesaplama
mae = tf.reduce_mean(tf.abs(x_valid - results))  # MAE hesaplanır
print("Mean Absolute Error:", mae.numpy())  # MAE değeri yazdırılır

# AÇIKLAMA: Bu kod parçası, TensorFlow ve NumPy kullanarak zaman serisi tahmini yapar.
