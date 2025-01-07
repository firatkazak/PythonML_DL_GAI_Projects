from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

# Building the Model for One Input & Output

# Loading dataset
housing = fetch_california_housing()  # California konut verilerini yüklüyoruz.

# Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=42)
# Verileri eğitim ve test olarak bölüyoruz. random_state=42, sonuçların tutarlılığı için sabit bir rastgelelik sağlar.
tf.random.set_seed(42)  # TensorFlow işlemleri için rastgeleliği kontrol eder. Sonuçların yeniden üretilebilir olması için.

# Modeling

# Creating the layers
normalization_layer = tf.keras.layers.Normalization()  # Giriş verilerini normalize etmek için katman.
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")  # 30 nöronlu, ReLU aktivasyonlu gizli katman.
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")  # 30 nöronlu, ReLU aktivasyonlu ikinci gizli katman.
concat_layer = tf.keras.layers.Concatenate()  # Girdi ve son gizli katmanı birleştirmek için kullanılan katman.
output_layer = tf.keras.layers.Dense(1)  # Tek bir çıkış nöronuna sahip katman, regresyon çıktısı (örneğin, konut fiyatı).

# Building the model
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])  # Giriş verisi için giriş katmanı. shape=X_train.shape[1:] ile giriş boyutunu belirleriz.
normalized = normalization_layer(input_)  # Giriş verileri normalizasyon katmanından geçiyor.
hidden1 = hidden_layer1(normalized)  # İlk gizli katmana normalleştirilmiş girişler veriliyor.
hidden2 = hidden_layer2(hidden1)  # İkinci gizli katman, birinci katmanın çıktısını alıyor.
concat = concat_layer([normalized, hidden2])  # Normalleştirilmiş giriş ve ikinci gizli katman birleştiriliyor.
output = output_layer(concat)  # Çıktı katmanı birleştirilmiş verileri alıyor.
model = tf.keras.Model(inputs=[input_], outputs=[output])  # Model oluşturuluyor, giriş katmanı input_, çıkış katmanı output.
model.summary()  # Modelin özetini görüntüler.

# Model Training
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Adam optimizasyon algoritması, öğrenme oranı 0.001.
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])  # Modelin derlenmesi, kayıp fonksiyonu MSE, metrik RMSE.
normalization_layer.adapt(X_train)  # Normalizasyon katmanını eğitim verilerine göre adapte ediyoruz.
history1 = model.fit(X_train, y_train, epochs=20, validation_split=0.2)
# Modeli eğitiyoruz, eğitim verilerinin %20'si doğrulama için kullanılıyor, 20 epoch boyunca eğitim.

# Model Evaluation
mse_test = model.evaluate(X_test, y_test)  # Test verileri ile modelin değerlendirilmesi. MSE ve RMSE değerlerini döner.
print(mse_test)

# Prediction
X_new = X_test[:3]  # Test verilerinden ilk 3 örnek alınır.
sonuc1 = pd.DataFrame(X_new)  # İlk 3 örnekten bir DataFrame oluşturulur.
print(sonuc1)

y_pred = model.predict(X_new)  # İlk 3 örnek için tahmin yapılır.
sonuc2 = pd.DataFrame(y_pred)  # Tahminler DataFrame'e dönüştürülür.
print(sonuc2)

sonuc3 = pd.DataFrame(y_test[:3])  # Test setinden gerçek değerlerin ilk 3 örneği DataFrame olarak yazdırılır.
print(sonuc3)

# Building the Model with Two Inputs & One Output
# Creating the layers
# Input layers
input_wide = tf.keras.layers.Input(shape=[5])  # Geniş giriş verisi için (5 özellikli) giriş katmanı.
input_deep = tf.keras.layers.Input(shape=[6])  # Derin giriş verisi için (6 özellikli) giriş katmanı.

# Normalization layers
norm_layer_wide = tf.keras.layers.Normalization()  # Geniş veri için normalizasyon katmanı.
norm_layer_deep = tf.keras.layers.Normalization()  # Derin veri için normalizasyon katmanı.

# Modeling
norm_wide = norm_layer_wide(input_wide)  # Geniş veri normalizasyonu.
norm_deep = norm_layer_deep(input_deep)  # Derin veri normalizasyonu.

# Hidden layers
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)  # Derin veriye uygulanan ilk gizli katman.
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)  # İkinci gizli katman, ilk gizli katmanın çıktısını alıyor.

# Concatenation layer
concat = tf.keras.layers.concatenate([norm_wide, hidden2])  # Geniş ve derin veriler birleştiriliyor.

# Output layer
output = tf.keras.layers.Dense(1)(concat)  # Tek bir çıktıya sahip katman, birleştirilmiş veriler üzerine uygulanıyor.

# Model building
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])  # Model iki girişe (geniş ve derin) sahip.

# Model Compiling
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Adam optimizasyonu.
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])  # Model derleniyor, kayıp fonksiyonu MSE.
X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]  # Eğitim verileri geniş (ilk 5 özellik) ve derin (sonraki 6 özellik) olarak bölünüyor.
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]  # Test verileri geniş ve derin olarak bölünüyor.

norm_layer_wide.adapt(X_train_wide)  # Geniş veri normalizasyonu eğitim verilerine göre adapte ediliyor.
norm_layer_deep.adapt(X_train_deep)  # Derin veri normalizasyonu eğitim verilerine göre adapte ediliyor.

history2 = model.fit((X_train_wide, X_train_deep), y_train, epochs=20, validation_split=0.2)
# Model geniş ve derin girişler için eğitiliyor, 20 epoch boyunca.

# Model Evaluation
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)  # Test seti üzerinde model değerlendirilmesi.
print(mse_test)

X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]  # Test setinden geniş ve derin veri için ilk 3 örnek alınıyor.
y_pred = model.predict((X_new_wide, X_new_deep))  # İlk 3 örnek için tahmin yapılır.
sonuc4 = pd.DataFrame(y_pred)  # Tahminler DataFrame'e dönüştürülür.
print(sonuc4)

# Building the Model with Two Inputs & Two Outputs
# Model Building
norm_wide = norm_layer_wide(input_wide)  # Geniş veri normalizasyonu.
norm_deep = norm_layer_deep(input_deep)  # Derin veri normalizasyonu.

# Hidden layers
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)  # Derin veriye ilk gizli katman.
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)  # Derin veriye ikinci gizli katman.

# Concatenation layer
concat = tf.keras.layers.concatenate([norm_wide, hidden2])  # Geniş ve derin veriler birleştiriliyor.

# Output layers
output = tf.keras.layers.Dense(1)(concat)  # Birleştirilmiş veriler üzerine ana çıkış katmanı.
aux_output = tf.keras.layers.Dense(1)(hidden2)  # İkinci çıkış katmanı, ikinci gizli katmandan alınan ek bir çıkış.

model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output, aux_output])
# İki giriş ve iki çıkışa sahip model oluşturuluyor.

# Model Compiling
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Adam optimizasyonu.
model.compile(
    loss=("mse", "mse"),  # İki çıkış için kayıp fonksiyonu MSE.
    loss_weights=(0.9, 0.1),  # Ana çıkışa %90, ek çıkışa %10 ağırlık veriliyor.
    optimizer=optimizer,
    metrics=[["RootMeanSquaredError"], ["RootMeanSquaredError"]]  # İki çıkış için RMSE metriği.
)

# Model Training
norm_layer_wide.adapt(X_train_wide)  # Geniş veri normalizasyonu.
norm_layer_deep.adapt(X_train_deep)  # Derin veri normalizasyonu.
history3 = model.fit((X_train_wide, X_train_deep), (y_train, y_train), epochs=20, validation_split=0.2)
# Model iki giriş ve iki çıkış için eğitiliyor.

# Model Evaluation
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))  # Test verileri üzerinde modelin değerlendirilmesi.
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results  # Değerlendirme sonuçları.

# Model Prediction
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))  # Ana ve ek çıkışlar için tahminler yapılır.

print(y_pred_main)  # Ana çıkış tahminleri yazdırılır.
print(y_pred_aux)  # Ek çıkış tahminleri yazdırılır.

y_pred_tuple = model.predict((X_new_wide, X_new_deep))  # İki çıkış için tahminler alınır.
y_pred = dict(zip(model.output_names, y_pred_tuple))  # Tahminler bir sözlük olarak anahtar (çıkış isimleri) ve değer (tahminler) şeklinde eşleştirilir.
print(y_pred)

# Bu kod, yapay sinir ağlarıyla konut fiyatı tahmini üzerine bir uygulama yapıyor.
# Özellikle derin öğrenme kullanarak, Kaliforniya konut verileri üzerinde bir regresyon modeli oluşturuluyor.
# Kodda, TensorFlow ve Keras kullanılarak farklı giriş verileriyle model eğitimi, değerlendirme ve tahmin işlemleri gerçekleştiriliyor.

# Öne çıkan konular:
# 1. Veri yükleme ve işleme:
#    - `fetch_california_housing()` ile Kaliforniya konut verileri yükleniyor.
#    - Veriler, `train_test_split()` ile eğitim ve test setlerine ayrılıyor.

# 2. Model yapısı ve katmanlar:
#    - Girdilerde normalizasyon yapılıyor.
#    - İki gizli katman oluşturuluyor (her biri 30 nöronlu, `relu` aktivasyon fonksiyonu kullanıyor).
#    - Çıkışta ise tek bir değer üretiliyor (konut fiyatını tahmin etmek için).

# 3. Eğitim ve değerlendirme:
#    - Model `Adam` optimizasyon algoritması ile derleniyor.
#    - Kayıp fonksiyonu olarak **MSE** (Mean Squared Error, ortalama kare hata) kullanılıyor.
#    - Model, eğitim verisi üzerinde eğitilip, test verisi üzerinde değerlendirme yapılıyor.

# 4. İki Girişli Model:
#    - Modelde iki farklı girdi kullanılıyor (geniş ve derin girişler), bunlar farklı özellik kümeleri olabilir.
#    - Bu model, birden fazla veri türü (özellik) kullanarak tahmin yapmaya yönelik daha karmaşık bir yapı oluşturuyor.

# 5. İki Çıkışlı Model:
#    - Son model, iki farklı çıkışa sahip: ana çıkış (asıl tahmin) ve ek çıkış (yardımcı tahmin).
#    - Bu, modelin hem ana hedef (konut fiyatı) hem de başka bir hedef (örneğin, farklı bir özellik) için tahmin yapabilmesine olanak tanır.

# Neyi öğrenmelisin?
# - TensorFlow ve Keras: Derin öğrenme modelleri oluşturmak için bu kütüphaneleri nasıl kullanacağını öğrenmen faydalı olur.
# - Sinir Ağı Katmanları: `Dense`, `Normalization`, `Concatenate` gibi katmanlar ne işe yarar, nasıl kullanılır?
# - Regresyon Problemleri: Bu model bir regresyon problemi çözüyor (sürekli bir değer tahmini).
# - NOT: Regresyon modellerinin nasıl eğitildiği ve değerlendirildiği konusunu kavra.
# - Veri Normalizasyonu: Girdi verilerinin daha iyi performans göstermesi için nasıl normalleştirildiğini öğren.
# - Optimizasyon: `Adam` gibi optimizasyon algoritmalarının nasıl çalıştığını ve **MSE** gibi kayıp fonksiyonlarını öğren.
# Bu kod, **yapay sinir ağlarıyla regresyon** problemleri çözmeyi öğrenmek için bir başlangıç noktasıdır.
