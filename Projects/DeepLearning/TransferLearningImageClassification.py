import tensorflow_datasets as tfds  # TensorFlow veri setlerini yüklemek için tensorflow_datasets kütüphanesi kullanılıyor.
import tensorflow as tf  # TensorFlow kütüphanesi makine öğrenmesi modelleri için kullanılıyor.

# Data Loading (Veri Yükleme)
dataset, info = tfds.load(name="tf_flowers", as_supervised=True,
                          with_info=True)  # "tf_flowers" veri seti yükleniyor. as_supervised=True, veri giriş ve çıkış çiftleri olarak döndürüyor.
# with_info=True, veri seti hakkında bilgi döndürüyor (örneğin sınıf isimleri, veri seti büyüklüğü vs.).

# Data Preprocessing (Veri Ön İşleme)
dataset_size = info.splits["train"].num_examples  # Eğitim verisi örneklerinin sayısını alıyor.
class_names = info.features["label"].names  # Sınıf isimlerini (etiketlerini) alıyor.
n_classes = info.features["label"].num_classes  # Veri setindeki sınıf sayısını alıyor.
test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    name="tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],  # Veri seti üç parçaya bölünüyor: %10 test, %15 validasyon, %75 eğitim.
    as_supervised=True  # Giriş ve etiket çiftleri olarak döndürülüyor.
)

batch_size = 32  # Eğitim sırasında verilerin toplu olarak işleneceği batch büyüklüğü.
preprocess = tf.keras.Sequential([  # Resim verileri için ön işleme adımları bir sıralı modelde tanımlanıyor.
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),  # Görselleri 224x224 boyutuna getiriyor.
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)  # Xception modeline uygun şekilde giriş verisini ön işliyor (örneğin normalizasyon).
])

train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))  # Eğitim verisi her bir resim için ön işlemden geçiriliyor.
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)  # Veriler karıştırılıyor, batch'ler halinde yükleniyor ve performans için önceden yükleniyor.

valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)  # Validasyon verisi de aynı şekilde işleniyor.
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)  # Test verisi de aynı şekilde işleniyor.

data_augmentation = tf.keras.Sequential([  # Veri artırma için bir sıralı model tanımlanıyor (data augmentation).
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),  # Yatay olarak rastgele çeviriyor.
    tf.keras.layers.RandomContrast(factor=0.2, seed=42),  # Rastgele kontrast değişiklikleri yapıyor.
    tf.keras.layers.RandomRotation(factor=0.05, seed=42)  # Rastgele hafif döndürme işlemi yapıyor.
])

# Model Building (Model Kurulumu)
tf.random.set_seed(42)  # Rastgele işlemler için tekrarlanabilirlik sağlamak amacıyla sabit bir rastgelelik tohum değeri belirleniyor.
base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)  # Xception modeli önceden eğitilmiş ImageNet ağırlıkları ile yükleniyor.
# include_top=False, modelin tepe katmanının (son sınıflandırma katmanı) dahil edilmediği anlamına gelir, bu şekilde son katman yeniden eğitilebilir.
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)  # Sonuçları düzleştiren ve ortalama havuzlama yapan katman ekleniyor.
output = tf.keras.layers.Dense(n_classes, activation="softmax")(
    x)  # n_classes (sınıf sayısı) kadar softmax aktivasyon fonksiyonuna sahip bir tam bağlantılı (dense) katman ekleniyor.
model = tf.keras.Model(inputs=base_model.input, outputs=output)  # Girişler ve çıkışlar tanımlanarak yeni model oluşturuluyor.

for layer in base_model.layers:  # Xception modelindeki katmanlar eğitim sırasında sabitleniyor (donduruluyor).
    layer.trainable = False  # Önceden eğitilmiş ağırlıkların güncellenmemesi için katmanlar sabitleniyor.

model.compile(loss="sparse_categorical_crossentropy",  # Sparse kategorik çapraz entropi kaybı kullanılıyor (etiketler tamsayı şeklinde olduğunda kullanılır).
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
              # Stochastic Gradient Descent (SGD) algoritması kullanılıyor, öğrenme oranı 0.1, momentum 0.9.
              metrics=["accuracy"]  # Başarı ölçütü olarak doğruluk (accuracy) izleniyor.
              )

# Model Training (Model Eğitimi)
history = model.fit(train_set, validation_data=valid_set, epochs=3)  # Model 3 dönem (epoch) boyunca eğitiliyor, validasyon verisiyle doğruluk kontrol ediliyor.

for indices in zip(range(33), range(33, 66), range(66, 99), range(99, 132)):  # 132 katmanlı Xception modelinin katman isimlerini gruplar halinde ekrana yazdırıyor.
    for idx in indices:
        print(f"{idx:3}: {base_model.layers[idx].name:22}", end="")  # Katman indeksini ve adını yazdırıyor.
    print()

for layer in base_model.layers[56:]:  # Modelin son kısmındaki bazı katmanlar serbest bırakılıyor (dondurulmuş katmanlar eğitilebilir hale geliyor).
    layer.trainable = True  # Bu katmanlar artık eğitilebilir (trainable).

model.compile(loss="sparse_categorical_crossentropy",  # Modelin kaybı ve optimizasyonu tekrar tanımlanıyor (son katmanlar açıldığı için).
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),  # Aynı optimizer ve loss fonksiyonu ile derleniyor.
              metrics=["accuracy"])

history = model.fit(train_set, validation_data=valid_set, epochs=10)  # Model tekrar eğitiliyor, bu kez 10 dönem boyunca.

# AÇIKLAMA

# Bu kod, bir derin öğrenme modeli kullanarak görüntü sınıflandırma görevini gerçekleştiren bir yapıyı içeriyor. Özetle, aşağıdaki adımları gerçekleştiriyor:

# Veri Setini Yükleme ve Ön İşleme:

# tf_flowers adlı çiçek sınıflandırma veri setini TensorFlow veri setleri aracılığıyla yüklüyor.
# Görselleri modelin girdi boyutuna (224x224 piksel) uyacak şekilde yeniden boyutlandırıyor ve Xception modeline uygun şekilde normalizasyon yapıyor.
# Veriyi eğitim, doğrulama (validasyon) ve test setlerine bölüyor.
# Veri seti için veri artırma (data augmentation) işlemleri tanımlanıyor (döndürme, çevirme vb.).

# Transfer Learning (Aktarım Öğrenmesi) Kullanarak Modeli Kurma:

# Daha önce ImageNet veri seti ile eğitilmiş Xception modeli kullanılıyor.
# Bu modelin son katmanları çıkarılıyor ve yerine çiçek sınıflarını sınıflandıracak yeni katmanlar ekleniyor.
# Önceden eğitilmiş modelin ağırlıkları sabitleniyor, yani modelin bu katmanları yeniden eğitilmiyor.

# Modeli Eğitme:

# Model, veri seti üzerinde 3 dönem (epoch) boyunca eğitiliyor.
# Eğitimin ardından modelin bazı katmanları açılıyor (tekrar eğitilebilir hale getiriliyor) ve bu sefer 10 dönem boyunca daha fazla eğitim yapılıyor.

# Bu Koddan Öğrendiklerim;

# TensorFlow ile Veri Yükleme ve Ön İşleme:

# tensorflow_datasets kullanarak veri seti yüklemeyi ve görüntü verisi üzerinde yeniden boyutlandırma, normalizasyon gibi işlemleri nasıl yapacağını öğrenmelisin.
# Veriyi eğitim, doğrulama ve test setlerine nasıl bölebileceğini anlamalısın.

# Transfer Learning (Aktarım Öğrenmesi):

# Daha önce eğitilmiş bir modelin (örneğin Xception) nasıl kullanılacağını ve belirli katmanları sabitleyerek kendi veri setine uyarlayarak nasıl yeniden eğitebileceğini.
# Yeni bir görev için modelin son katmanlarını değiştirme ve yeniden eğitme mantığı.

# Model Eğitimi ve Optimizasyon:
#
# Modeli nasıl eğiteceğini ve optimizasyon yöntemlerini öğrenmelisin.
# Özellikle SGD gibi optimizasyon algoritmalarını nasıl kullanacağını ve learning_rate, momentum gibi parametrelerin ne işe yaradığını anlamalısın.
# loss fonksiyonlarını (burada sparse_categorical_crossentropy) öğrenmelisin, çünkü bunlar sınıflandırma problemlerinde kullanılır.

# Veri Artırma (Data Augmentation):

# Veri setindeki çeşitliliği artırmak için kullanılan rastgele dönüşümler (örneğin, döndürme, kontrast artırma) gibi teknikleri öğrenmelisin.
# Bu kodu tam anlamıyla kavramak için TensorFlow, derin öğrenme, görüntü sınıflandırma ve transfer learning konularına odaklanman gerekiyor.
# Özellikle veri seti ile çalışmayı, model eğitimi süreçlerini ve önceden eğitilmiş modelleri (pretrained models) yeniden kullanmayı anlamak önemli.
