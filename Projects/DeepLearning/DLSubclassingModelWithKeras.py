from sklearn.datasets import fetch_california_housing  # California konut veri setini yüklemek için kullanılıyor.
from sklearn.model_selection import train_test_split  # Veriyi eğitim, doğrulama ve test setlerine ayırmak için.
import tensorflow as tf  # TensorFlow ile derin öğrenme modelleri inşa etmek için.

# Data Loading
housing = fetch_california_housing()  # California konut veri setini yüklüyor, 'data' ve 'target' içerir.

for i in housing:  # housing veri setinin içeriklerini yazdırmak için bir döngü.
    print(i)  # Veri setindeki anahtarları ('data', 'target', vb.) yazdırır.

# Data preprocessing
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,  # housing verisini bölüyoruz.
                                                              housing.target,  # Konut fiyatları hedef (target).
                                                              random_state=42  # Sonuçları tekrar edilebilir kılmak için sabit bir seed.
                                                              )

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,  # Eğitim verisini daha fazla bölüyoruz.
                                                      y_train_full,  # Doğrulama seti oluşturmak için.
                                                      random_state=42  # Tekrar edilebilirliği sağlamak için seed.
                                                      )


# Model Building
class WideAndDeepModel(tf.keras.Model):  # TensorFlow'un Model sınıfından türetilmiş özel bir model oluşturuyoruz.
    def __init__(self, units=30, activation="relu", **kwargs):  # Modelin katmanlarını tanımlıyoruz.
        super().__init__(**kwargs)  # Üst sınıfın başlatıcı metodunu çağırıyoruz (örneğin model adı gibi ek parametreleri desteklemek için).
        self.norm_layer_wide = tf.keras.layers.Normalization()  # Geniş giriş için normalizasyon katmanı.
        self.norm_layer_deep = tf.keras.layers.Normalization()  # Derin giriş için normalizasyon katmanı.
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)  # Birinci gizli katman (units: nöron sayısı, activation: ReLU fonksiyonu).
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)  # İkinci gizli katman (aynı parametrelerle).
        self.main_output = tf.keras.layers.Dense(1)  # Çıkış katmanı (fiyatı tahmin etmek için 1 nöron).

    def call(self, inputs):  # İleri besleme sırasında kullanılacak metod.
        input_wide, input_deep = inputs  # Geniş ve derin girişleri alıyor.
        norm_wide = self.norm_layer_wide(input_wide)  # Geniş girişi normalizasyon katmanından geçiriyor.
        norm_deep = self.norm_layer_deep(input_deep)  # Derin girişi normalizasyon katmanından geçiriyor.
        hidden1 = self.hidden1(norm_deep)  # Derin giriş üzerinden birinci gizli katmanı çalıştırıyor.
        hidden2 = self.hidden2(hidden1)  # Birinci gizli katmandan gelen çıkışı ikinci gizli katmana gönderiyor.
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])  # Geniş ve ikinci gizli katmandan gelen çıkışı birleştiriyor.
        output = self.main_output(concat)  # Birleştirilmiş veriyi çıkış katmanına gönderiyor.
        return output  # Modelin tahminini döndürüyor.


tf.random.set_seed(42)  # Rastgele işlemlerin tekrarlanabilir olmasını sağlamak için sabit bir seed belirliyoruz.
model = WideAndDeepModel(units=30,  # 30 nöronlu gizli katmanlar.
                         activation="relu",  # ReLU aktivasyon fonksiyonu.
                         name="my_model"  # Modelin ismi.
                         )

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Adam optimizasyon algoritmasını kullanıyoruz (öğrenme hızı: 0.001).
model.compile(loss="mse",  # Kayıp fonksiyonu: ortalama kare hata (Mean Squared Error).
              optimizer=optimizer,  # Adam optimizatörü.
              metrics=["RootMeanSquaredError"]  # Ek olarak kök ortalama kare hata (RMSE) metriği kullanılıyor.
              )

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]  # Eğitim verisini geniş ve derin girdiler olarak ayırıyoruz (ilk 5 ve son 2 özellik).
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]  # Doğrulama seti için de aynı şekilde geniş ve derin girdiler ayırıyoruz.
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]  # Test verisi için de geniş ve derin girdiler ayırıyoruz.

# Model Training
model.norm_layer_wide.adapt(X_train_wide)  # Normalizasyon katmanı, geniş eğitim verisine adapte oluyor.
model.norm_layer_deep.adapt(X_train_deep)  # Normalizasyon katmanı, derin eğitim verisine adapte oluyor.

history = model.fit((X_train_wide, X_train_deep),  # Geniş ve derin eğitim verileri.
                    y_train,  # Eğitim hedefleri (fiyatlar).
                    validation_data=((X_valid_wide, X_valid_deep), y_valid),  # Doğrulama verileri.
                    epochs=10  # 10 dönem boyunca eğitilecek.
                    )

# Model Evaluation
eval_results = model.evaluate((X_test_wide, X_test_deep), y_test)  # Test verisi ile modeli değerlendiriyoruz.

print(eval_results)  # Sonuçları yazdırıyoruz (MSE ve RMSE).

# Prediction
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]  # Test verisinden 3 örnek seçiyoruz.

y_pred = model.predict((X_new_wide, X_new_deep))  # Bu örnekler için tahmin yapıyoruz.
print(y_pred)  # Tahmin edilen fiyatlar yazdırılıyor.
print(y_test[:3])  # Gerçek fiyatlar yazdırılıyor.

# Özet
# Bu kod, geniş (wide) ve derin (deep) bir modelin nasıl oluşturulacağını gösteriyor. Model iki girişten oluşuyor: geniş ve derin girdi.
# Geniş giriş, daha az özelliğe sahip basit verilerden oluşurken, derin giriş daha fazla özelliği işleyen daha kompleks bir yapıdır.

# WideAndDeepModel: İki girişten gelen verileri normalleştiriyor, gizli katmanlardan geçiriyor ve çıkışta tahmin yapıyor.
# Model Training: Model eğitim verileri ile eğitiliyor ve doğrulama seti ile performansı kontrol ediliyor.
# Model Evaluation ve Prediction: Test verileriyle modelin performansı değerlendiriliyor ve yeni veriler için tahmin yapılıyor.

# Öğrenmen Gerekenler
# Wide & Deep Learning Model Yapısı: Geniş ve derin girişlerle çalışan modeller.
# Özelleştirilmiş Keras Modeli Oluşturma: tf.keras.Model sınıfından türeyen özel bir model nasıl oluşturulur.
# Normalizasyon Katmanı: Verileri daha iyi öğrenilebilir hale getirmek için normalizasyon kullanımı.
# Model Eğitim Süreci: Eğitim ve doğrulama setleri ile modelin eğitilmesi ve değerlendirilmesi.
