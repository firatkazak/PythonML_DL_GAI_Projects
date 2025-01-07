import tensorflow as tf  # TensorFlow kütüphanesini yüklüyor, derin öğrenme modellerini oluşturmak ve eğitmek için kullanılır.
import matplotlib.pyplot as plt  # Grafik çizimi için kullanılan kütüphane, genellikle verilerin görselleştirilmesinde kullanılır.
import pandas as pd  # Veri manipülasyonu ve analizinde kullanılan güçlü bir kütüphane.
import numpy as np  # Sayısal hesaplamalar için kullanılan kütüphane, çok boyutlu diziler üzerinde işlem yapmayı sağlar.

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()  # Fashion MNIST veri setini indiriyor ve veri setini eğitim/test olarak yüklüyor.

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
# Veriyi iki gruba ayırıyor: X_train_full ve y_train_full (eğitim verileri), X_test ve y_test (test verileri).
# X_train_full: Eğitim veri setinin girişleri (görüntüler).
# y_train_full: Eğitim veri setinin hedefleri (etiketler).
# X_test, y_test: Test veri seti, modelin başarısını değerlendirmek için kullanılır.

X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
# Eğitim verilerinden 5000 örneği doğrulama seti olarak ayırıyor, geriye kalanları eğitim verisi olarak kullanıyor.
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
# Son 5000 örneği doğrulama seti olarak kullanıyor.

print(X_train.shape)  # X_train'in boyutlarını yazdırır. (55,000, 28, 28) gibi olabilir.
print(X_train.dtype)  # X_train'in veri tipini yazdırır, genellikle uint8 (8 bitlik tam sayı).

X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.
# Görüntü verilerini 0-255 arasında değerlerden 0-1 arasına normalize ediyor. Modelin daha iyi öğrenmesi için bu gereklidir.

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Her bir sınıfa karşılık gelen isimleri tanımlıyor. Toplamda 10 sınıf var, her biri bir giysi türüne karşılık gelir.

print(class_names[y_train[0]])  # Eğitim setindeki ilk görüntünün sınıf ismini yazdırıyor. Örneğin "Coat" gibi.

tf.random.set_seed(42)  # Rastgele sayı üretimini sabitliyor, böylece sonuçlar her seferinde aynı oluyor.

# Model oluşturma
model = tf.keras.Sequential()  # Sıralı bir model oluşturuyor. Katmanlar sırayla eklenir.
model.add(tf.keras.layers.Input(shape=[28, 28]))  # Giriş katmanını tanımlıyor. Her bir görüntü 28x28 piksel boyutunda.
model.add(tf.keras.layers.Flatten())  # Görüntüyü 28x28 matristen (2D) düz bir vektöre (1D) dönüştürüyor.
model.add(tf.keras.layers.Dense(units=300, activation="relu"))  # İlk tam bağlantılı (dense) katman, 300 nöron içerir ve aktivasyon fonksiyonu olarak ReLU kullanır.
model.add(tf.keras.layers.Dense(units=100, activation="relu"))  # İkinci dense katman, 100 nöron içerir ve ReLU kullanır.
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
# Çıkış katmanı, 10 sınıfı temsil eden 10 nörondan oluşur ve softmax aktivasyon fonksiyonu kullanılır. Bu fonksiyon, her sınıfa bir olasılık atar.

# Alternatif model oluşturma yöntemi, aynı yapı:
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[28, 28]),  # Giriş katmanı (28x28 görüntüler için).
    tf.keras.layers.Flatten(),  # Görüntüyü 1D vektöre dönüştürüyor.
    tf.keras.layers.Dense(units=300, activation="relu"),  # İlk gizli katman, 300 nöron.
    tf.keras.layers.Dense(units=100, activation="relu"),  # İkinci gizli katman, 100 nöron.
    tf.keras.layers.Dense(units=10, activation="softmax")  # Çıkış katmanı, 10 sınıf için.
])

print(model.summary())  # Modelin özetini yazdırır, her katmandaki parametrelerin sayısını gösterir.
print(model.layers)  # Modeldeki katmanları listeler.

hidden1 = model.layers[1]  # Modelin ikinci katmanını (flatten katmanı) seçiyor.
weights, biases = hidden1.get_weights()  # Seçilen katmanın ağırlıklarını (weights) ve önyargılarını (biases) alıyor.

print(weights)  # Ağırlık matrisi yazdırılıyor.
print(biases)  # Önyargı vektörü yazdırılıyor.

# Modelin derlenmesi (compile edilmesi)
model.compile(
    loss="sparse_categorical_crossentropy",
    # Kayıp fonksiyonu olarak sparse_categorical_crossentropy kullanılıyor. Bu, sınıflandırma problemlerinde sık kullanılan bir fonksiyondur.
    optimizer="sgd",  # Optimizasyon algoritması olarak Stochastic Gradient Descent (SGD) kullanılıyor.
    metrics=["accuracy"]  # Modelin başarımını ölçmek için doğruluk (accuracy) metriği kullanılıyor.
)

# Modelin eğitilmesi
history = model.fit(X_train,  # Eğitim verileri (girişler).
                    y_train,  # Eğitim verileri (hedefler).
                    epochs=30,  # 30 epoch boyunca eğitim yapılacak.
                    validation_data=(X_valid, y_valid)  # Doğrulama verileri, modelin her epoch sonunda doğruluk oranını kontrol etmek için kullanılır.
                    )

# Eğitim geçmişinin grafiği
pd.DataFrame(history.history).plot(figsize=(8, 5),  # Eğitim geçmişini pandas DataFrame formatına çevirip grafiğini çiziyor.
                                   xlim=[0, 29],  # X ekseni sınırları (epoch sayısı).
                                   ylim=[0, 1],  # Y ekseni sınırları (başarım oranı 0-1).
                                   grid=True,  # Izgara çizgilerini gösteriyor.
                                   xlabel="Epoch",  # X ekseninin başlığı.
                                   style=["r--", "r--.", "b-", "b-*"])  # Grafiğin stilini ayarlıyor.
plt.show()  # Grafiği gösteriyor.

# Test veri seti üzerinde modelin performansını değerlendirme
sonuc1 = model.evaluate(X_test, y_test)  # Test verilerini kullanarak modelin doğruluk ve kayıp değerini hesaplar.
print(sonuc1)  # Test sonuçlarını yazdırıyor.

# Yeni verilerde tahmin yapma
X_new = X_test[:3]  # Test setinden 3 yeni örnek alınıyor.
y_proba = model.predict(X_new)  # Model bu yeni örnekler için olasılık dağılımları tahmin ediyor.
y_proba.round(2)  # Olasılıkları iki ondalık basamağa yuvarlıyor.
y_pred = y_proba.argmax(axis=-1)  # En yüksek olasılığa sahip sınıfı seçiyor (tahmin edilen sınıf).
print(y_pred)  # Tahmin edilen sınıf numaralarını yazdırıyor.

sonuc2 = np.array(class_names)[y_pred]  # Tahmin edilen sınıf numaralarını sınıf isimlerine dönüştürüyor.
print(sonuc2)  # Tahmin edilen sınıfları yazdırıyor. Örneğin ["Sneaker", "T-shirt/top", "Ankle boot"] gibi.

# AÇIKLAMALAR
# Fashion MNIST: 28x28 boyutunda gri tonlamalı giysi görüntülerinden oluşan bir veri setidir. 10 farklı sınıfı içerir.
# model.compile(): Modeli eğitmeden önce kayıp fonksiyonunu, optimizasyon algoritmasını ve başarı metriğini tanımlar.
# model.fit(): Modeli eğitim verileriyle eğitir ve doğrulama verisiyle performansı izler.
# model.evaluate(): Test verileri üzerinde modelin doğruluk ve kayıp oranlarını hesaplar.
# model.predict(): Yeni verilerde tahmin yapar ve her sınıfa olasılık atar.
