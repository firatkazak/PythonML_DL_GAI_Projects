import tensorflow as tf  # TensorFlow kütüphanesini içe aktarır
import os  # Dosya ve dizin işlemleri için os modülünü içe aktarır
import numpy as np  # NumPy kütüphanesini içe aktarır, bilimsel hesaplamalar için kullanılır
from tensorflow import keras  # Keras'ı TensorFlow içinden içe aktarır
from tensorflow.keras import layers  # Keras katmanlarını içe aktarır
import matplotlib.pyplot as plt  # Matplotlib, veri görselleştirme için kullanılır

# csv dosyamızın pathi;
csv_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/jena_climate_2009_2016.csv"

# CSV dosyasını okur
with open(csv_path) as f:  # CSV dosyasını okuma modunda açar
    data = f.read()  # Dosyadaki verileri okur

# Satırları ayırır
lines = data.split("\n")  # Veriyi satır satır böler
header = lines[0].split(",")  # İlk satırı başlık olarak ayırır (sütun adları)
lines = lines[1:]  # İlk satırı atlar
print(header)  # Başlıkları yazdırır
print(len(lines))  # Satır sayısını yazdırır

# Isı verilerini ve ham verileri depolamak için diziler oluşturur
temperature = np.zeros((len(lines),))  # Isı verileri için bir dizi oluşturur
raw_data = np.zeros((len(lines), len(header) - 1))  # Ham veriler için bir dizi oluşturur

# Ham verileri okur
for i, line in enumerate(lines):  # Her satırı ve indeksini alır
    values = [float(x) for x in line.split(",")[1:]]  # Satırı ayırır ve değerleri float'a çevirir, ilk değeri atlar
    temperature[i] = values[1]  # Isı verisini diziye atar
    raw_data[i, :] = values[:]  # Tüm değerleri ham veriye atar

# Isı verilerini çizer
plt.plot(range(len(temperature)), temperature)  # İlgili aralıkta sıcaklık değerlerini çizer

# Veri kümesini böler
num_train_samples = int(0.5 * len(raw_data))  # Eğitim verileri için örnek sayısını belirler
num_val_samples = int(0.25 * len(raw_data))  # Doğrulama verileri için örnek sayısını belirler
num_test_samples = len(raw_data) - num_train_samples - num_val_samples  # Test verileri için örnek sayısını hesaplar
print("num_train_samples:", num_train_samples)  # Eğitim örnek sayısını yazdırır
print("num_val_samples:", num_val_samples)  # Doğrulama örnek sayısını yazdırır
print("num_test_samples:", num_test_samples)  # Test örnek sayısını yazdırır

# Verileri normalize eder
mean = raw_data[:num_train_samples].mean(axis=0)  # Eğitim setinin ortalamasını hesaplar
raw_data -= mean  # Ham verilerden ortalamayı çıkarır
std = raw_data[:num_train_samples].std(axis=0)  # Eğitim setinin standart sapmasını hesaplar
raw_data /= std  # Ham verileri standart sapmaya bölerek normalize eder

# Zaman serisi için parametreleri ayarlar
sampling_rate = 6  # Örnekleme oranı, her 6. değeri alır
sequence_length = 120  # Her bir dizinin uzunluğu
delay = sampling_rate * (sequence_length + 24 - 1)  # Gecikme hesaplaması
batch_size = 256  # Batch boyutu

# Eğitim veri setini oluşturur
train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],  # Gecikmeli ham veriler (son 'delay' kadar hariç)
    targets=temperature[delay:],  # Hedef sıcaklık değerleri
    sampling_rate=sampling_rate,  # Örnekleme oranı
    sequence_length=sequence_length,  # Dizinin uzunluğu
    shuffle=True,  # Veriyi karıştırma
    batch_size=batch_size,  # Batch boyutu
    start_index=0,  # Başlangıç indeksi
    end_index=num_train_samples  # Eğitim setinin son indeksi
)

# Doğrulama veri setini oluşturur
val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],  # Gecikmeli ham veriler (son 'delay' kadar hariç)
    targets=temperature[delay:],  # Hedef sıcaklık değerleri
    sampling_rate=sampling_rate,  # Örnekleme oranı
    sequence_length=sequence_length,  # Dizinin uzunluğu
    shuffle=True,  # Veriyi karıştırma
    batch_size=batch_size,  # Batch boyutu
    start_index=num_train_samples,  # Doğrulama setinin başlangıç indeksi
    end_index=num_train_samples + num_val_samples)  # Doğrulama setinin son indeksi

# Test veri setini oluşturur
test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],  # Gecikmeli ham veriler (son 'delay' kadar hariç)
    targets=temperature[delay:],  # Hedef sıcaklık değerleri
    sampling_rate=sampling_rate,  # Örnekleme oranı
    sequence_length=sequence_length,  # Dizinin uzunluğu
    shuffle=True,  # Veriyi karıştırma
    batch_size=batch_size,  # Batch boyutu
    start_index=num_train_samples + num_val_samples)  # Test setinin başlangıç indeksi

# Eğitim veri setindeki örneklerin şekillerini yazdırır
for samples, targets in train_dataset:  # Eğitim veri setinden örnekleri alır
    print("samples shape:", samples.shape)  # Girdi örneklerinin boyutunu yazdırır
    print("targets shape:", targets.shape)  # Hedef değerlerin boyutunu yazdırır
    break  # Sadece bir batch için yazdırma


# Basit bir yöntemle değerlendirme yapma
def evaluate_naive_method(dataset):  # Dataset parametre olarak alınır
    total_abs_err = 0.  # Toplam mutlak hata başlatılır
    samples_seen = 0  # Görülen örnek sayısı başlatılır
    for samples, targets in dataset:  # Dataset'teki örnekleri döngüye alır
        preds = samples[:, -1, 1] * std[1] + mean[1]  # Tahminleri oluşturur (son zaman dilimindeki ikinci özellik)
        total_abs_err += np.sum(np.abs(preds - targets))  # Tahmin hatalarını toplar
        samples_seen += samples.shape[0]  # Görülen örnek sayısını günceller
    return total_abs_err / samples_seen  # Ortalama mutlak hatayı döndürür


# Doğrulama ve test setleri için değerlendirme yapar
print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")  # Doğrulama seti için MAE'yi yazdırır
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")  # Test seti için MAE'yi yazdırır

# Model oluşturma
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))  # Girdi katmanı, dizinin uzunluğu ve özellik sayısı
x = layers.GlobalAveragePooling1D()(inputs)  # 1D global ortalama havuzlama katmanı
x = layers.Dense(16, activation="relu")(x)  # 16 nöronlu, ReLU aktivasyon fonksiyonlu tam bağlı katman
outputs = layers.Dense(1)(x)  # Çıkış katmanı, tek nöron (tek değer tahmini)
model = keras.Model(inputs, outputs)  # Modeli oluşturur

# Modeli kaydetmek için callback ayarları
callbacks = [keras.callbacks.ModelCheckpoint("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/jena_dense.keras", save_best_only=True)]  # En iyi modeli kaydeder

# Modeli derleme
model.compile(optimizer="rmsprop",  # Optimizer: RMSprop
              loss="mse",  # Kayıp fonksiyonu: Ortalama Kare Hata (MSE)
              metrics=["mae"]  # Kullanılacak metrik: Ortalama Mutlak Hata (MAE)
              )

# Modeli eğitme
history = model.fit(train_dataset,  # Eğitim verisi
                    epochs=10,  # Eğitim döngüsü sayısı
                    validation_data=val_dataset,  # Doğrulama verisi
                    callbacks=callbacks)  # Callback fonksiyonları

# Eğitilmiş modeli yükleme
model = keras.models.load_model("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/jena_dense.keras")  # En iyi modeli yükler
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")  # Test setindeki MAE'yi yazdırır

# Eğitim ve doğrulama kaybını çizer
loss = history.history["mae"]  # Eğitim kaybını alır
val_loss = history.history["val_mae"]  # Doğrulama kaybını alır
epochs = range(1, len(loss) + 1)  # Dönem sayısını ayarlar
plt.figure()  # Yeni bir figür oluşturur

plt.plot(epochs,  # Dönem sayısını x eksenine koyar
         loss,  # Eğitim kaybını y eksenine koyar
         "bo",  # Mavi daire işareti ile çizer
         label="Training MAE"  # Eğitimi etiketler
         )

plt.plot(epochs,  # Dönem sayısını x eksenine koyar
         val_loss,  # Doğrulama kaybını y eksenine koyar
         "b",  # Mavi çizgi ile çizer
         label="Validation MAE"  # Doğrulamayı etiketler
         )

plt.title("Training and validation MAE")  # Başlık ekler
plt.legend()  # Legend ekler
plt.show()  # Grafiği gösterir

# Yeni bir model oluşturma (CNN)
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))  # Girdi katmanı
x = layers.Conv1D(8, 24, activation="relu")(inputs)  # 1D Konvolüsyon katmanı
x = layers.MaxPooling1D(2)(x)  # 1D Maksimum havuzlama katmanı
x = layers.Conv1D(8, 12, activation="relu")(x)  # Başka bir 1D Konvolüsyon katmanı
x = layers.MaxPooling1D(2)(x)  # Başka bir 1D Maksimum havuzlama katmanı
x = layers.Conv1D(8, 6, activation="relu")(x)  # Başka bir 1D Konvolüsyon katmanı
x = layers.GlobalAveragePooling1D()(x)  # Global ortalama havuzlama katmanı
outputs = layers.Dense(1)(x)  # Çıkış katmanı
model = keras.Model(inputs, outputs)  # Modeli oluşturur

# Modeli kaydetmek için callback ayarları
callbacks = [keras.callbacks.ModelCheckpoint("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/jena_conv.keras", save_best_only=True)]  # En iyi modeli kaydeder

# Modeli derleme
model.compile(optimizer="rmsprop",  # Optimizer: RMSprop
              loss="mse",  # Kayıp fonksiyonu: Ortalama Kare Hata (MSE)
              metrics=["mae"]  # Kullanılacak metrik: Ortalama Mutlak Hata (MAE)
              )

# Modeli eğitme
history = model.fit(train_dataset,  # Eğitim verisi
                    epochs=10,  # Eğitim döngüsü sayısı
                    validation_data=val_dataset,  # Doğrulama verisi
                    callbacks=callbacks)  # Callback fonksiyonları

# Eğitilmiş modeli yükleme
model = keras.models.load_model("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/jena_conv.keras")  # En iyi modeli yükler
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")  # Test setindeki MAE'yi yazdırır

# Eğitim ve doğrulama kaybını çizer
loss = history.history["mae"]  # Eğitim kaybını alır
val_loss = history.history["val_mae"]  # Doğrulama kaybını alır
epochs = range(1, len(loss) + 1)  # Dönem sayısını ayarlar
plt.figure()  # Yeni bir figür oluşturur

plt.plot(epochs,  # Dönem sayısını x eksenine koyar
         loss,  # Eğitim kaybını y eksenine koyar
         "bo",  # Mavi daire işareti ile çizer
         label="Training MAE"  # Eğitimi etiketler
         )

plt.plot(epochs,  # Dönem sayısını x eksenine koyar
         val_loss,  # Doğrulama kaybını y eksenine koyar
         "b",  # Mavi çizgi ile çizer
         label="Validation MAE"  # Doğrulamayı etiketler
         )

plt.title("Training and validation MAE")  # Başlık ekler
plt.legend()  # Legend ekler
plt.show()  # Grafiği gösterir

# Yeni bir model oluşturma (LSTM)
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))  # Girdi katmanı
x = layers.LSTM(16)(inputs)  # LSTM katmanı
outputs = layers.Dense(1)(x)  # Çıkış katmanı
model = keras.Model(inputs, outputs)  # Modeli oluşturur

# Modeli kaydetmek için callback ayarları
callbacks = [keras.callbacks.ModelCheckpoint("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/jena_lstm.keras", save_best_only=True)]  # En iyi modeli kaydeder

# Modeli derleme
model.compile(optimizer="rmsprop",  # Optimizer: RMSprop
              loss="mse",  # Kayıp fonksiyonu: Ortalama Kare Hata (MSE)
              metrics=["mae"]  # Kullanılacak metrik: Ortalama Mutlak Hata (MAE)
              )

# Modeli eğitme
history = model.fit(train_dataset,  # Eğitim verisi
                    epochs=10,  # Eğitim döngüsü sayısı
                    validation_data=val_dataset,  # Doğrulama verisi
                    callbacks=callbacks  # Callback fonksiyonları
                    )

# Eğitilmiş modeli yükleme
model = keras.models.load_model("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/jena_lstm.keras")  # En iyi modeli yükler
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")  # Test setindeki MAE'yi yazdırır

# Eğitim ve doğrulama kaybını çizer
loss = history.history["mae"]  # Eğitim kaybını alır
val_loss = history.history["val_mae"]  # Doğrulama kaybını alır
epochs = range(1, len(loss) + 1)  # Dönem sayısını ayarlar
plt.figure()  # Yeni bir figür oluşturur

plt.plot(epochs,  # Dönem sayısını x eksenine koyar
         loss,  # Eğitim kaybını y eksenine koyar
         "bo",  # Mavi daire işareti ile çizer
         label="Training MAE"  # Eğitimi etiketler
         )

plt.plot(epochs,  # Dönem sayısını x eksenine koyar
         val_loss,  # Doğrulama kaybını y eksenine koyar
         "b",  # Mavi çizgi ile çizer
         label="Validation MAE"  # Doğrulamayı etiketler
         )

plt.title("Training and validation MAE")  # Başlık ekler
plt.legend()  # Legend ekler
plt.show()  # Grafiği gösterir

# sequence_length: Girdi dizisinin uzunluğu, modelin hangi uzunluktaki veriyi işleyeceğini belirler.
# sampling_rate: Veri setinden ne sıklıkla örnek alınacağını belirler (örneğin, her 6. veriyi alır).
# batch_size: Her eğitim döngüsünde işlenecek veri miktarını belirtir.
# mean ve std: Normalizasyon için kullanılır, verilerin ortalamasını ve standart sapmasını hesaplar.

# AÇIKLAMA: Bu kod, zaman serisi verilerini kullanarak makine öğrenimi modeli oluşturmayı, eğitmeyi ve değerlendirmeyi amaçlayan bir örnek uygulamadır.

# Kodun Temel İşlevleri;

# Veri İndirme ve Ön İşleme:
# Bir iklim veri kümesi (sıcaklık gibi) indirilir ve CSV formatında açılır.
# Veriler satırlarına ayrılır ve önemli değerler (sıcaklık ve diğer ham veriler) numpy dizilerine depolanır.
# Veriler normalizasyon işlemi ile standart hale getirilir, bu sayede modelin daha iyi öğrenmesi sağlanır.

# Veri Kümesinin Bölünmesi:
# Veri seti, eğitim, doğrulama ve test setlerine bölünür. Bu, modelin eğitimi sırasında overfitting (aşırı uyum) riskini azaltmak için önemlidir.

# Zaman Serisi Veri Seti Oluşturma:
# tf.keras.utils.timeseries_dataset_from_array fonksiyonu kullanılarak zaman serisi veri setleri oluşturulur.
# Bu, geçmiş verilerin kullanılarak gelecekteki sıcaklık tahminlerinin yapılabilmesi için gereklidir.

# Model Oluşturma:
# Farklı türlerde (Dense, CNN ve LSTM) modeller oluşturulur. Her model, zaman serisi verilerini işlemek için uygun mimarilere sahiptir.
# Her modelin katmanları, keras kütüphanesindeki çeşitli katman türlerini (örneğin, Conv1D, LSTM) kullanarak tanımlanır.

# Modelin Eğitimi:
# Eğitim seti üzerinde model eğitilir ve doğrulama seti ile performansı izlenir.
# Model, kayıp fonksiyonu olarak MSE (Ortalama Kare Hata) kullanır ve optimizasyon için RMSprop algoritması seçilir.

# Model Değerlendirme:
# Eğitim ve test verileri üzerinde modelin tahmin performansı değerlendirilir.
# Ortalama Mutlak Hata (MAE) hesaplanarak modelin başarısı ölçülür.

# Sonuçların Görselleştirilmesi:
# Eğitim ve doğrulama kayıpları (MAE) grafikle gösterilerek modelin öğrenme süreci görselleştirilir.

# NE ÖĞRENDİM?

# Zaman Serisi Analizi: Zaman serisi verilerini nasıl işleyebileceğinizi ve bu tür verilerle makine öğrenimi modeli oluşturmayı öğrenebilirsiniz.

# Veri Ön İşleme: Verilerin nasıl normalleştirileceği, bölüneceği ve işleneceği hakkında bilgi sahibi olursunuz.

# Model Mimarisinin Oluşturulması: Farklı türdeki model mimarilerinin nasıl oluşturulacağını ve bunların hangi durumlarda kullanıldığını anlayabilirsiniz.

# Model Eğitimi ve Değerlendirilmesi: Bir modelin nasıl eğitileceğini, doğrulama ile aşırı uyumu önlemek için stratejiler geliştirmeyi
# ve sonuçların nasıl değerlendirileceğini öğrendik.

# Görselleştirme: Modelin performansını anlamak için sonuçların nasıl grafikle gösterileceğini öğrenirsiniz.

# Sonuç: Bu kod, makine öğrenimi uygulamalarında karşılaşabileceğiniz birçok temel kavramı kapsar. Zaman serisi verileriyle çalışmak,
# veri ön işleme teknikleri, model tasarımı, eğitim süreci ve model değerlendirmesi gibi konular hakkında derinlemesine bilgi edinmenize yardımcı olur.
