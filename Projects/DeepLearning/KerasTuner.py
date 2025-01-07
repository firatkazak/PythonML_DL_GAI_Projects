import keras_tuner as kt  # Keras Tuner, model hiperparametrelerini optimize etmek için kullanılan bir kütüphane.
import tensorflow as tf  # TensorFlow kütüphanesi.
import keras  # Keras'ı TensorFlow üzerinden içe aktarır.
from pathlib import Path  # Dosya yolları ile çalışmak için Path kütüphanesi.

# Loading the Dataset (Fashion MNIST veri setini yükleme)
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
# Fashion MNIST veri setini yükler. Eğitim ve test setleri görüntü ve etiket olarak ayrılır.

print(img_train.shape)  # Eğitim setinin boyutunu ekrana yazdırır.

img_train = img_train.astype('float32') / 255.0  # Eğitim görüntülerini 0-255 aralığından 0-1 aralığına normalleştirir.
img_test = img_test.astype('float32') / 255.0  # Aynı şekilde test görüntüleri de normalleştirilir.


# Model Building (Model oluşturma)
def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    # 'n_hidden', 0 ile 8 arasında gizli katman sayısını hiperparametre olarak belirler.
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    # 'n_neurons', her katmandaki nöron sayısını 16 ile 256 arasında hiperparametre olarak belirler.
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    # 'learning_rate', öğrenme hızını 1e-4 ile 1e-2 arasında logaritmik aralıkta ayarlar.
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    # 'optimizer', SGD (stochastic gradient descent) veya Adam optimizasyon yöntemlerinden birini seçer.

    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        # SGD seçildiyse, belirlenen öğrenme hızıyla SGD optimizasyonu uygulanır.
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Aksi durumda Adam optimizasyon algoritması kullanılır.

    model = tf.keras.Sequential()  # Sıralı (Sequential) model başlatılır.
    model.add(tf.keras.layers.Flatten())  # Giriş verisi düzleştirilir (2D'den 1D'ye geçiş yapılır).

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
        # Hiperparametrelerde belirtilen sayıda gizli katman eklenir. Her katman ReLU aktivasyon fonksiyonu kullanır.

    model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
    # Son katman 10 birimli bir Dense katmanıdır. Bu, 10 sınıf tahmini yapar ve softmax ile sınıflandırır.

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    # Modelin kayıp fonksiyonu (loss) olarak 'sparse_categorical_crossentropy' kullanılır. Optimizasyon algoritması seçilen optimizera göre belirlenir.

    return model  # Oluşturulan modeli döndürür.


# MyClassificationHyperModel sınıfı, HyperModel'den miras alarak özel bir sınıf tanımlanır.
class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)  # Hiperparametre aramaları için build_model fonksiyonunu kullanır.


# Random Search Tuner (Rastgele Arama Yöntemi ile Tuner)
random_search_tuner = kt.RandomSearch(build_model,  # Modeli oluşturur.
                                      objective="val_accuracy",  # Doğrulama doğruluğu optimize edilecek hedef olarak seçilir.
                                      max_trials=5,  # 5 farklı deneme yapılacaktır.
                                      overwrite=True,  # Daha önceki arama sonuçları üzerine yazılır.
                                      directory="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/my_fashion_mnist",  # Arama sonuçlarının kaydedileceği dizin.
                                      project_name="my_rnd_search",  # Proje ismi.
                                      seed=42  # Rastgelelik için sabit bir tohum değeri (tekrar edilebilirlik sağlar).
                                      )

random_search_tuner.search(img_train,  # Eğitim görüntüleri kullanılır.
                           label_train,  # Eğitim etiketleri.
                           epochs=10,  # Her model için 10 epoch eğitimi yapılır.
                           validation_split=0.2  # Verinin %20'si doğrulama için ayrılır.
                           )

# Taking a look at the best model (En iyi modeli inceleme)
top3_models = random_search_tuner.get_best_models(num_models=3)  # En iyi 3 modeli döndürür.
best_model = top3_models[0]  # En iyi model seçilir.
print(best_model)  # En iyi modeli yazdırır.

top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
# En iyi 3 hiperparametre kombinasyonunu döndürür.
print(top3_params[0].values)  # En iyi hiperparametre değerlerini yazdırır.

best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
# En iyi deneme sonucunu alır.
best_trial.summary()  # Bu denemenin özetini yazdırır.

# Evaluate the best model (En iyi modeli değerlendirme)
best_model.fit(img_train, label_train, epochs=5)  # En iyi model 5 epoch boyunca eğitimden geçirilir.
test_loss, test_accuracy = best_model.evaluate(img_test, label_test)  # Test verisi ile değerlendirme yapılır.
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")  # Test kaybı ve doğruluğunu yazdırır.

# Hyperband Tuner (Hyperband Yöntemi ile Tuner)
hyperband_tuner = kt.Hyperband(MyClassificationHyperModel(),  # Model için MyClassificationHyperModel kullanılır.
                               objective="val_accuracy",  # Amaç doğrulama doğruluğunu optimize etmektir.
                               seed=42,  # Sabit tohum değeri.
                               max_epochs=10,  # Maksimum 10 epoch boyunca eğitim yapılır.
                               factor=3,  # Hyperband için faktör değeri (her aşamada kaynakları 3'e böler).
                               hyperband_iterations=2,  # Hyperband algoritması 2 iterasyon yapacaktır.
                               overwrite=True,  # Daha önceki sonuçlar üzerine yazılır.
                               directory="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/my_fashion_mnist",  # Kayıt dizini.
                               project_name="hyperband"  # Proje ismi.
                               )

root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
# TensorBoard logları için dizin ayarlanır.
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(root_logdir))
# TensorBoard callback tanımlanır (eğitim sırasında log tutar).
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
# Early Stopping callback tanımlanır (eğitim 2 epoch boyunca iyileşme olmazsa durdurulur).

hyperband_tuner.search(img_train,  # Eğitim verisi kullanılır.
                       label_train,  # Eğitim etiketleri.
                       epochs=10,  # 10 epoch boyunca eğitim yapılır.
                       validation_split=0.2,  # Verinin %20'si doğrulama için ayrılır.
                       callbacks=[early_stopping_cb, tensorboard_cb]  # Callback'ler kullanılır.
                       )

# Bayesian Optimization Tuner (Bayes Optimizasyonu ile Tuner)
bayesian_opt_tuner = kt.BayesianOptimization(MyClassificationHyperModel(),  # Model için MyClassificationHyperModel kullanılır.
                                             objective="val_accuracy",  # Amaç doğrulama doğruluğunu optimize etmektir.
                                             seed=42,  # Sabit tohum değeri.
                                             max_trials=10,  # 10 deneme yapılacaktır.
                                             alpha=1e-4,  # Bayes optimizasyonu parametresi (hassasiyeti belirler).
                                             beta=2.6,  # Bayes optimizasyon parametresi (bilgi keşfi/keşfetme oranı).
                                             overwrite=True,  # Sonuçlar üzerine yazılır.
                                             directory="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/my_fashion_mnist",  # Dizin.
                                             project_name="bayesian_opt"  # Proje ismi.
                                             )

bayesian_opt_tuner.search(img_train,  # Eğitim verisi.
                          label_train,  # Eğitim etiketleri.
                          epochs=10,  # 10 epoch boyunca eğitim yapılır.
                          validation_split=0.2,  # Verinin %20'si doğrulama için ayrılır.
                          callbacks=[early_stopping_cb]  # Early Stopping callback kullanılır.
                          )

# Parametrelerin ve Yöntemlerin Açıklaması:

# kt.RandomSearch:
# Amaç: Rastgele arama yöntemiyle en iyi hiperparametre kombinasyonlarını bulmak.
# Parametreler:
# build_model: Optimizasyon için kullanılacak modelin fonksiyonu.
# objective: Optimizasyon hedefi (örn. doğrulama doğruluğu).
# max_trials: Yapılacak toplam deneme sayısını belirler.
# overwrite: Var olan sonuçların üzerine yazılıp yazılmayacağını belirtir.
# directory: Sonuçların kaydedileceği dizin.
# project_name: Proje adı.
# seed: Rastgelelik için sabit bir tohum değeri (tekrar edilebilirlik sağlar).

# kt.Hyperband:
# Amaç: Hyperband algoritmasını kullanarak hızlı hiperparametre araması yapmak.
# Parametreler:
# MyClassificationHyperModel(): Kullanılacak modelin sınıfı.
# objective: Optimizasyon hedefi (örn. doğrulama doğruluğu).
# seed: Rastgelelik için sabit bir tohum değeri.
# max_epochs: Her model için maksimum epoch sayısı.
# factor: Her aşamada kaynakların nasıl bölüneceğini belirler (kaynakları ne kadar hızlı azaltacağı).
# hyperband_iterations: Hyperband algoritmasının kaç iterasyon yapacağını belirtir.
# overwrite: Var olan sonuçların üzerine yazılıp yazılmayacağını belirtir.

# kt.BayesianOptimization:
# Amaç: Bayes optimizasyonu ile hiperparametreleri optimize etmek.
# Parametreler:
# MyClassificationHyperModel(): Kullanılacak modelin sınıfı.
# objective: Optimizasyon hedefi (örn. doğrulama doğruluğu).
# seed: Rastgelelik için sabit bir tohum değeri.
# max_trials: Yapılacak toplam deneme sayısını belirler.
# alpha: Bayes optimizasyonu sürecinde kullanılan bir parametre (hassasiyeti belirler).
# beta: Bayes optimizasyonu sürecinde kullanılan bir parametre (bilgi keşfi/keşfetme oranı).
# overwrite: Var olan sonuçların üzerine yazılıp yazılmayacağını belirtir.
# directory: Sonuçların kaydedileceği dizin.
# project_name: Proje adı.

# validation_split: Modelin eğitim sürecinde, eğitim verisinin belirli bir yüzdesinin (örneğin %20) doğrulama için ayrılmasını sağlar.
# Bu, modelin doğruluğunu değerlendirmek için kullanılır.

# early_stopping_cb: Eğitim sırasında modelin doğrulama kaybı veya doğrulama doğruluğu belirli bir süre boyunca iyileşmezse eğitim sürecini durdurur.
# Bu, aşırı uyum (overfitting) riskini azaltmaya yardımcı olur.

# tensorboard_cb: TensorBoard ile modelin eğitim sürecini görselleştirmek için kullanılan callback.
# Eğitim sırasında çeşitli metrikleri takip etmek için log dosyaları oluşturur.
