import tensorflow as tf  # TensorFlow kütüphanesini içe aktar
import pandas as pd  # Pandas kütüphanesini içe aktar
from pathlib import Path  # Dosya yolu işlemleri için Path sınıfını içe aktar
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib kütüphanesini içe aktar
import numpy as np  # Numpy kütüphanesini içe aktar
from tensorflow.keras.layers import Input, SimpleRNN, Dense, LSTM, GRU  # Keras katmanlarını içe aktar
from tensorflow.keras.models import Sequential  # Sequential model yapısını içe aktar

# Dataset'i yükleme
tf.keras.utils.get_file(
    "ridership.tgz",  # İndirilecek dosya adı
    "https://github.com/TirendazAcademy/Deep-Learning-with-TensorFlow/raw/main/Data/ridership.tgz",  # Dosya URL'si
    cache_dir="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler",  # Önbellek dizini
    extract=True  # Dosyanın çıkarılmasını sağla
)

# Veri ön işleme
path = Path("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/datasets/ridership_extracted/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")  # Veri dosyasının yolu
df = pd.read_csv(path, parse_dates=["service_date"])  # CSV dosyasını oku ve "service_date" sütununu tarih olarak ayrıştır
df.columns = ["date", "day_type", "bus", "rail", "total"]  # Sütun isimlerini yeniden adlandır
df = df.sort_values("date").set_index("date")  # Tarihe göre sıralayıp indeks olarak ayarla
df = df.drop("total", axis=1)  # "total" sütununu çıkar
df = df.drop_duplicates()  # Tekrar eden satırları çıkar

print(df.head())  # İlk 5 satırı yazdır

# Veri görselleştirme
df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))  # Belirtilen tarihlerde veri noktalarını göster
plt.show()  # Grafiği göster

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]  # 7 gün farkını al ve belirtilen tarihlerdeki verileri seç

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))  # 2 satırlı, 1 sütunlu alt grafik oluştur
df.plot(ax=axs[0], grid=True, marker=".")  # İlk grafikte verileri göster
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")  # 7 gün kaydırılmış verileri göster
diff_7.plot(ax=axs[1], grid=True, marker=".")  # İkinci grafikte 7 günlük farkı göster
plt.show()  # Grafiği göster

sonuc1 = list(df.loc["2019-05-25":"2019-05-27"]["day_type"])  # Belirtilen tarihlerdeki "day_type" değerlerini listele
print(sonuc1)  # Sonucu yazdır

# Model 1: Naif yaklaşım
print(diff_7.abs().mean())  # 7 günlük farkın mutlak değerlerinin ortalamasını yazdır

targets = df[["bus", "rail"]]["2019-03":"2019-05"]  # Hedef değişken olarak "bus" ve "rail" sütunlarını belirle
print(targets)  # Hedef değişkenleri yazdır

period = slice("2001", "2019")  # 2001'den 2019'a kadar olan dönem dilimini tanımla
print(period)  # Dönem dilimini yazdır

# Veri türlerini kontrol et
print(df.dtypes)  # DataFrame'in veri türlerini yazdır

# Sayısal sütunları seç
numeric_df = df.select_dtypes(include=[float, int])  # Sadece sayısal sütunları seç

# Sayısal olmayan değerleri NaN olarak işaretle ve çıkar
numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')  # Sayısal olmayan değerleri NaN yap

# Temizlenmiş verilerle işlemlere devam et
df_monthly = numeric_df.resample("M").mean()  # Aylık ortalamaları hesapla

rolling_average_12_months = df_monthly[period].rolling(window=12).mean()  # 12 aylık kaydırmalı ortalamayı hesapla

fig, ax = plt.subplots(figsize=(8, 4))  # Grafik oluştur
df_monthly[period].plot(ax=ax, marker=".")  # Aylık verileri grafikte göster
rolling_average_12_months.plot(ax=ax, grid=True, legend=False)  # Kaydırmalı ortalamayı grafikte göster
plt.show()  # Grafiği göster

df_monthly.diff(12)[period].plot(grid=True, marker=".", figsize=(8, 3))  # 12 aylık farkları grafikte göster
plt.show()  # Grafiği göster

rail_train = df["rail"]["2016-01":"2018-12"] / 1e6  # Eğitim seti için "rail" verisini al ve milyonla böl
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6  # Doğrulama seti için "rail" verisini al ve milyonla böl
rail_test = df["rail"]["2019-05":] / 1e6  # Test seti için "rail" verisini al ve milyonla böl

# Keras ile veri yükleme
seq_length = 56  # Girdi dizisinin uzunluğunu belirle
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),  # Eğitim verilerini numpy dizisine dönüştür
    targets=rail_train[seq_length:],  # Hedef değişkenleri belirle
    sequence_length=seq_length,  # Girdi dizisinin uzunluğu
    batch_size=32,  # Batch boyutu
    shuffle=True,  # Verileri karıştır
    seed=42  # Rastgelelik için tohum değeri
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_valid.to_numpy(),  # Doğrulama verilerini numpy dizisine dönüştür
    targets=rail_valid[seq_length:],  # Hedef değişkenleri belirle
    sequence_length=seq_length,  # Girdi dizisinin uzunluğu
    batch_size=32  # Batch boyutu
)

# Model 2: Regresyon
tf.random.set_seed(42)  # Rastgelelik için tohum değeri
model = tf.keras.Sequential([  # Sekansiyel bir model oluştur
    Input(shape=[seq_length]),  # Giriş katmanı, şekli belirle
    Dense(1)  # Çıkış katmanı, 1 nöronlu yoğun katman
])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(  # Erken durdurma geri çağırma fonksiyonu
    monitor="val_mae",  # İzlenecek metrik: doğrulama MAE
    patience=30,  # 30 dönem boyunca iyileşme olmazsa durdur
    restore_best_weights=True  # En iyi ağırlıkları geri yükle
)

opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)  # Stokastik gradyan inişi optimizasyonunu tanımla

model.compile(loss=tf.keras.losses.Huber(),  # Modelin kaybı olarak Huber kaybını kullan
              optimizer=opt,  # Optimizer olarak belirlediğimiz opt'i kullan
              metrics=["mae"])  # Performans metriği olarak MAE'yi kullan

history = model.fit(train_ds,  # Eğitim verisi ile modeli eğit
                    validation_data=valid_ds,  # Doğrulama verisini ekle
                    epochs=100,  # Toplam eğitim dönemi
                    callbacks=[early_stopping_cb])  # Erken durdurma geri çağırmasını ekle

valid_loss, valid_mae = model.evaluate(valid_ds)  # Modeli doğrulama verisi ile değerlendir
print(valid_mae * 1e6)  # Doğrulama MAE değerini yazdır (milyonla çarp)

# Model 3: Basit RNN
tf.random.set_seed(42)  # Rastgelelik için tohum değeri
univar_model = Sequential([  # Sekansiyel bir model oluştur
    Input(shape=[None, 1]),  # Giriş katmanı, şekli belirle; burada her zaman 'None' dinamik zaman adımını temsil eder
    SimpleRNN(32),  # 32 nöronlu basit RNN katmanı ekle
    Dense(1)  # Çıkış katmanı, 1 nöronlu yoğun katman
])


# Modeli eğitme ve değerlendirme fonksiyonu
def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=100):  # Modeli eğitme ve değerlendirme fonksiyonu
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(  # Erken durdurma geri çağırma fonksiyonu
        monitor="val_mae",  # İzlenecek metrik: doğrulama MAE
        patience=30,  # 30 dönem boyunca iyileşme olmazsa durdur
        restore_best_weights=True)  # En iyi ağırlıkları geri yükle
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)  # Stokastik gradyan inişi optimizasyonunu tanımla
    model.compile(loss=tf.keras.losses.Huber(),  # Modelin kaybı olarak Huber kaybını kullan
                  optimizer=opt,  # Optimizer olarak belirlediğimiz opt'i kullan
                  metrics=["mae"])  # Performans metriği olarak MAE'yi kullan
    history = model.fit(train_set,  # Eğitim verisi ile modeli eğit
                        validation_data=valid_set,  # Doğrulama verisini ekle
                        epochs=epochs,  # Toplam eğitim dönemi
                        callbacks=[early_stopping_cb])  # Erken durdurma geri çağırmasını ekle
    valid_loss, valid_mae = model.evaluate(valid_set)  # Modeli doğrulama verisi ile değerlendir
    return valid_mae * 1e6  # Doğrulama MAE değerini döndür (milyonla çarp)


fit_and_evaluate(univar_model, train_ds, valid_ds, learning_rate=0.05)  # Modeli eğit ve değerlendir

# Model 4: Derin RNN'ler
tf.random.set_seed(42)  # Rastgelelik için tohum değeri
deep_model = Sequential([  # Sekansiyel bir model oluştur
    Input(shape=[None, 1]),  # Giriş katmanı, şekli belirle
    SimpleRNN(32, return_sequences=True),  # 32 nöronlu RNN katmanı ekle, çıktı dizisi döndür
    SimpleRNN(32, return_sequences=True),  # İkinci 32 nöronlu RNN katmanı ekle
    SimpleRNN(32, return_sequences=True),  # Üçüncü 32 nöronlu RNN katmanı ekle
    Dense(1)  # Çıkış katmanı, 1 nöronlu yoğun katman
])

fit_and_evaluate(deep_model, train_ds, valid_ds, learning_rate=0.01)  # Derin modeli eğit ve değerlendir

# Model Tahmini
X = rail_valid.to_numpy()[np.newaxis, :seq_length, np.newaxis]  # Doğrulama verisini dizilere dönüştür ve şekil değiştir
print(X.shape)  # X'in şeklini yazdır

for step_ahead in range(14):  # 14 adım ileriye tahmin yap
    y_pred_one = univar_model.predict(X)  # Modelle tahmin et
    X = np.concatenate([X, y_pred_one.reshape(1, 1, 1)], axis=1)  # Tahmin sonucunu X'e ekle

Y_pred = pd.Series(X[0, -14:, 0],  # Tahmin sonuçlarını pandas serisi olarak oluştur
                   index=pd.date_range("2019-02-26", "2019-03-11"))  # Tarih aralığı ver

fig, ax = plt.subplots(figsize=(8, 3.5))  # Grafik oluştur
(rail_valid * 1e6)["2019-02-01":"2019-03-11"].plot(label=True, marker=".", ax=ax)  # Gerçek verileri grafikte göster
(Y_pred * 1e6).plot(label="Predictions", grid=True, marker="x", color="r", ax=ax)  # Tahminleri grafikte göster
ax.set_ylim([200_000, 800_000])  # Y ekseninin sınırlarını ayarla
plt.legend()  # Efsaneyi göster
plt.show()  # Grafiği göster

lstm_model = Sequential([  # LSTM modeli oluştur
    Input(shape=[None, 1]),  # Giriş katmanı, şekli belirle
    LSTM(32, return_sequences=True),  # 32 nöronlu LSTM katmanı ekle, çıktı dizisi döndür
    Dense(1)  # Çıkış katmanı, 1 nöronlu yoğun katman
])

fit_and_evaluate(lstm_model, train_ds, valid_ds, learning_rate=0.01, epochs=20)  # LSTM modelini eğit ve değerlendir

# Model 6: GRU
gru_model = Sequential([  # GRU modeli oluştur
    Input(shape=[None, 1]),  # Giriş katmanı, şekli belirle
    GRU(32, return_sequences=True),  # 32 nöronlu GRU katmanı ekle, çıktı dizisi döndür
    Dense(1)  # Çıkış katmanı, 1 nöronlu yoğun katman
])

fit_and_evaluate(gru_model, train_ds, valid_ds, learning_rate=0.01, epochs=20)  # GRU modelini eğit ve değerlendir

# AÇIKLAMA
# Bu kod, zaman serisi verilerini kullanarak makine öğrenimi modelleri oluşturmayı ve değerlendirmeyi amaçlıyor.

# Kodun Amacı;

# Veri Yükleme ve Ön İşleme:
# Dataset'i indirir ve yükler.
# Tarih sütununu işler, eksik ve gereksiz verileri temizler.
# Verileri analiz etmek ve görselleştirmek için hazır hale getirir.

# Görselleştirme:
# Verilerin zaman içindeki trendlerini ve değişikliklerini grafiklerle gösterir.
# Farklar ve kaydırmalarla ilgili grafikler oluşturur.

# Modelleme:
# Regresyon Modelleri: Basit bir regresyon modeli
# ve daha karmaşık RNN (Recurrent Neural Network), LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) modelleri oluşturur.
# Eğitim, doğrulama ve test setleri ile model performansını değerlendirir.

# Tahmin:
# Son modellerle gelecekteki verileri tahmin eder ve sonuçları görselleştirir.
# Öğrenmen Gereken Ana Konular

# Zaman Serisi Analizi:
# Zaman serisi verilerini anlama, dönemsellik ve trend analizi gibi kavramlar.
# pandas kütüphanesi ile veri çerçeveleri oluşturma ve manipülasyon.

# Veri Görselleştirme:
# matplotlib veya seaborn gibi kütüphanelerle veri görselleştirme teknikleri.

# Makine Öğrenimi Modelleri:
# Regresyon analizi, RNN, LSTM ve GRU gibi tekrar eden sinir ağı yapıları hakkında bilgi.
# Modelin eğitilmesi, değerlendirilmesi ve hiperparametre ayarlamaları.

# Keras ile Modelleme:
# TensorFlow ve Keras kütüphanelerini kullanarak model oluşturma ve eğitme.
# Keras'taki katmanlar (Input, Dense, SimpleRNN, LSTM, GRU) ve geri çağırma fonksiyonlarının kullanımı.

# Model Değerlendirme:
# Modelin performansını değerlendirmenin yolları, metrikler (MAE, Huber kaybı vb.) ve erken durdurma teknikleri.
