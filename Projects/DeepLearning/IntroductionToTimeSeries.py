import numpy as np  # NumPy, bilimsel hesaplamalar için kullanılır
import tensorflow as tf  # TensorFlow, makine öğrenmesi için bir kütüphane
import matplotlib.pyplot as plt  # Matplotlib, veri görselleştirme kütüphanesi

# Zaman verisi için bir eğilim fonksiyonu
def trend(time, slope=0):  # time: Zaman dizisi, slope: eğim değeri
    return slope * time  # Eğilim (trend) çizgisi oluşturur

# Mevsimsel bir desen oluşturma fonksiyonu
def seasonal_pattern(season_time):  # season_time: bir mevsim içindeki zaman
    return np.where(season_time < 0.4,  # 0.4'ten küçükse cos fonksiyonu kullan
                    np.cos(season_time * 2 * np.pi),  # Zamanın kosinüs değeri
                    1 / np.exp(3 * season_time))  # Değilse exp ile bir azalan fonksiyon kullan

# Zaman verisinde mevsimsellik ekleme fonksiyonu
def seasonality(time, period, amplitude=1, phase=0):  # time: Zaman, period: mevsim periyodu, amplitude: dalga yüksekliği, phase: faz kayması
    season_time = ((time + phase) % period) / period  # Mevsimsel zaman dilimi hesaplanır
    return amplitude * seasonal_pattern(season_time)  # Amplitüd ile mevsimsel desen çarpılır

# Gürültü ekleme fonksiyonu
def noise(time, noise_level=1, seed=None):  # time: Zaman, noise_level: gürültü seviyesi, seed: rastgelelik için sabit değer
    rnd = np.random.RandomState(seed)  # Belirli bir seed ile rastgele sayı üretici oluşturulur
    return rnd.randn(len(time)) * noise_level  # Zaman dizisine rastgele gürültü eklenir

time = np.arange(4 * 365 + 1, dtype="float32")  # 4 yıllık zaman verisi (365 gün) oluşturulur

baseline = 10  # Başlangıç seviyesindeki sabit değer
series = trend(time, slope=.05)  # Zaman dizisine küçük bir eğilim eklenir
amplitude = 15  # Mevsimsel dalgaların yüksekliği
slope = 0.09  # Eğilim eğimi
noise_level = 6  # Gürültü seviyesi

# Series verisine önce eğilim, sonra mevsimsellik eklenir
series = baseline + trend(time, slope) + seasonality(time,
                                                     period=365,  # Mevsim 365 günlük bir döngü
                                                     amplitude=amplitude  # Mevsimsel dalgaların yüksekliği
                                                     )
# Gürültü eklenir
series += noise(time,
                noise_level,  # Gürültünün miktarı
                seed=42  # Tekrarlanabilir rastgelelik için sabit seed
                )

# Zaman serisini çizdirme fonksiyonu
def plot_series(time, series, format="-", start=0, end=None):  # time: Zaman, series: Değerler, format: çizim formatı
    plt.plot(time[start:end], series[start:end], format)  # Zaman serisini çizdirir
    plt.xlabel("Time")  # X ekseni etiketlenir
    plt.ylabel("Value")  # Y ekseni etiketlenir
    plt.grid(True)  # Grafik ızgarası açılır

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time, series)  # Zaman serisi grafiği çizilir
plt.show()  # Grafik gösterilir

split_time = 1000  # Eğitim ve doğrulama seti için veri bölme noktası
time_train = time[:split_time]  # Eğitim verisi için zaman dilimi
x_train = series[:split_time]  # Eğitim verisi

time_valid = time[split_time:]  # Doğrulama verisi için zaman dilimi
x_valid = series[split_time:]  # Doğrulama verisi

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_train, x_train)  # Eğitim seti grafiği
plt.show()  # Grafik gösterilir

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_valid, x_valid)  # Doğrulama seti grafiği
plt.show()  # Grafik gösterilir

# Naive tahmin: Bir önceki zaman adımının değeri tahmin olarak kullanılır
naive_forecast = series[split_time - 1:-1]  # Naive tahmin, doğrulama setine uyarlanır

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_valid, x_valid)  # Doğrulama seti grafiği
plot_series(time_valid, naive_forecast)  # Naive tahmin grafiği
plt.show()  # Grafik gösterilir

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_valid, x_valid, start=0, end=150)  # İlk 150 adım
plot_series(time_valid, naive_forecast, start=1, end=151)  # Naive tahmin grafiği
plt.show()  # Grafik gösterilir

# Ortalama Kare Hata (MSE) hesaplanır
mse = tf.reduce_mean(tf.square(x_valid - naive_forecast))  # MSE: (gerçek - tahmin)² ortalaması
# Ortalama Mutlak Hata (MAE) hesaplanır
mae = tf.reduce_mean(tf.abs(x_valid - naive_forecast))  # MAE: |gerçek - tahmin| ortalaması

print("Mean Squared Error:", mse.numpy())  # MSE değeri yazdırılır
print("Mean Absolute Error:", mae.numpy())  # MAE değeri yazdırılır

# Hareketli ortalama tahmin fonksiyonu
def moving_average_forecast(series, window_size):  # series: zaman serisi, window_size: hareketli ortalama penceresi
    forecast = []  # Tahmin sonuçları için boş liste
    for time in range(len(series) - window_size):  # Pencere boyunca ilerlenir
        forecast.append(series[time:time + window_size].mean())  # Pencere içindeki ortalama alınır
    return np.array(forecast)  # Tahminler numpy dizisine dönüştürülür

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]  # Hareketli ortalama ile tahmin yapılır

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_valid, x_valid)  # Doğrulama seti grafiği
plot_series(time_valid, moving_avg)  # Hareketli ortalama tahmin grafiği
plt.show()  # Grafik gösterilir

# MSE ve MAE hesaplanır
mse = tf.reduce_mean(tf.square(x_valid - moving_avg))  # MSE hesaplanır
mae = tf.reduce_mean(tf.abs(x_valid - moving_avg))  # MAE hesaplanır

print("Mean Squared Error:", mse.numpy())  # MSE değeri yazdırılır
print("Mean Absolute Error:", mae.numpy())  # MAE değeri yazdırılır

# 365 günlük fark serisi hesaplanır
diff_series = (series[365:] - series[:-365])  # Mevsimsel fark serisi oluşturulur
diff_time = time[365:]  # Fark serisine karşılık gelen zaman dilimi

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(diff_time, diff_series)  # Fark serisi grafiği
plt.show()  # Grafik gösterilir

diff_moving_avg = moving_average_forecast(diff_series, window_size=50)[split_time - 365 - 50:]  # Fark serisine hareketli ortalama uygulanır

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_valid, diff_series[split_time - 365:])  # Doğrulama seti fark serisi grafiği
plot_series(time_valid, diff_moving_avg)  # Hareketli ortalama tahmin grafiği
plt.show()  # Grafik gösterilir

# Geçmiş verilere dayalı hareketli ortalama tahmin
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg  # Geçmiş verilerle birlikte tahmin yapılır

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_valid, x_valid)  # Doğrulama seti grafiği
plot_series(time_valid, diff_moving_avg_plus_past)  # Tahmin grafiği
plt.show()  # Grafik gösterilir

# MSE ve MAE hesaplanır
mse = tf.reduce_mean(tf.square(x_valid - diff_moving_avg_plus_past))  # MSE hesaplanır
mae = tf.reduce_mean(tf.abs(x_valid - diff_moving_avg_plus_past))  # MAE hesaplanır

print("Mean Squared Error:", mse.numpy())  # MSE değeri yazdırılır
print("Mean Absolute Error:", mae.numpy())  # MAE değeri yazdırılır

# Daha düzgün geçmiş verilerle tahmin
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], window_size=10) + diff_moving_avg  # Düzgünleştirilmiş geçmiş veriyle tahmin

plt.figure(figsize=(10, 6))  # Grafik boyutu belirlenir
plot_series(time_valid, x_valid)  # Doğrulama seti grafiği
plot_series(time_valid, diff_moving_avg_plus_smooth_past)  # Tahmin grafiği
plt.show()  # Grafik gösterilir

# MSE ve MAE hesaplanır
mse = tf.reduce_mean(tf.square(x_valid - diff_moving_avg_plus_smooth_past))  # MSE hesaplanır
mae = tf.reduce_mean(tf.abs(x_valid - diff_moving_avg_plus_smooth_past))  # MAE hesaplanır

print("Mean Squared Error:", mse.numpy())  # MSE değeri yazdırılır
print("Mean Absolute Error:", mae.numpy())  # MAE değeri yazdırılır

# AÇIKLAMA: Bu kod parçası zaman serisi tahmini yapıyor.
