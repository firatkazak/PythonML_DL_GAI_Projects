import os  # İşletim sistemi ile ilgili dosya ve dizin işlemleri için kullanılıyor.
import random  # Rastgele işlemler yapmak için kullanılıyor (örneğin rastgele dosya seçimi).
import matplotlib.pyplot as plt  # Grafik çizimleri ve görselleştirme için kullanılıyor.
import tensorflow as tf  # TensorFlow kütüphanesi, derin öğrenme için kullanılıyor.
from keras import layers  # Keras katmanları, model oluştururken kullanılıyor.
import numpy as np  # NumPy kütüphanesi, bilimsel hesaplamalar ve veri işlemleri için kullanılıyor.

# Understanding the Dataset (Veri Setini Anlama)
base_dir = 'C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/rock-paper-scissors/rps-cv-images'  # Veri setinin bulunduğu temel dizin.
paper_dir = os.path.join(base_dir, 'paper')  # Paper (kağıt) sınıfının bulunduğu klasör yolu.
rock_dir = os.path.join(base_dir, 'rock')  # Rock (taş) sınıfının bulunduğu klasör yolu.
scissors_dir = os.path.join(base_dir, 'scissors')  # Scissors (makas) sınıfının bulunduğu klasör yolu.

print("Rock : ", len(os.listdir(rock_dir)))  # Rock sınıfındaki resimlerin sayısını ekrana yazdırıyor.
print("Paper : ", len(os.listdir(paper_dir)))  # Paper sınıfındaki resimlerin sayısını ekrana yazdırıyor.
print("Scissors : ", len(os.listdir(scissors_dir)))  # Scissors sınıfındaki resimlerin sayısını ekrana yazdırıyor.

random_image = random.sample(os.listdir(paper_dir), 1)  # Paper dizininden rastgele bir resim seçiyor.
img = tf.keras.utils.load_img(f"{paper_dir}/{random_image[0]}")  # Seçilen resmi yüklüyor.
print(img)  # Resmi ekrana yazdırıyor (resmin bilgilerini gösterir).

img = tf.keras.utils.img_to_array(img)  # Resmi bir NumPy dizisine çeviriyor.
print(img.shape)  # Resmin boyutlarını (şeklini) yazdırıyor.

# Data Preprocessing (Veri Ön İşleme)
train_ds = tf.keras.utils.image_dataset_from_directory(  # Eğitim veri setini oluşturuyor.
    base_dir,  # Veri setinin bulunduğu dizin.
    validation_split=0.2,  # Verinin %20'si validasyon (doğrulama) için ayrılıyor.
    subset="training",  # Bu veri seti "training" (eğitim) olarak kullanılacak.
    image_size=(180, 180),  # Resimler 180x180 piksel olarak yeniden boyutlandırılıyor.
    batch_size=32,  # Veriler 32'lik gruplar halinde işleniyor.
    seed=42)  # Rastgelelik için sabit bir tohum değeri (aynı sonuçlar elde edilsin diye).

val_ds = tf.keras.utils.image_dataset_from_directory(  # Doğrulama veri setini oluşturuyor.
    base_dir,  # Veri setinin bulunduğu dizin.
    validation_split=0.2,  # Verinin %20'si validasyon için ayrılıyor.
    subset="validation",  # Bu veri seti "validation" (doğrulama) olarak kullanılacak.
    image_size=(180, 180),  # Resimler 180x180 piksel olarak yeniden boyutlandırılıyor.
    batch_size=32,  # Veriler 32'lik gruplar halinde işleniyor.
    seed=42)  # Rastgelelik için sabit bir tohum değeri.

class_names = train_ds.class_names  # Veri setindeki sınıf isimlerini alıyor (örneğin "rock", "paper", "scissors").
print(class_names)  # Sınıf isimlerini yazdırıyor.

for image_batch, labels_batch in train_ds:  # Eğitim veri setindeki ilk batch'i alıyor.
    print(image_batch.shape)  # Resim batch'inin boyutlarını yazdırıyor.
    print(labels_batch.shape)  # Etiketlerin boyutlarını yazdırıyor.
    break  # Sadece ilk batch'i göstermek için döngüyü durduruyor.

# Verileri görselleştiriyor
plt.figure(figsize=(10, 10))  # 10x10 boyutunda bir grafik oluşturuyor.
for images, labels in train_ds.take(1):  # Eğitim setinden bir batch alıyor.
    for i in range(9):  # İlk 9 resmi görselleştiriyor.
        ax = plt.subplot(3, 3, i + 1)  # 3x3 bir grid oluşturuyor.
        plt.imshow(images[i].numpy().astype("uint8"))  # Resmi gösteriyor.
        plt.title(class_names[labels[i]])  # Resmin sınıf ismini başlık olarak ekliyor.
        plt.axis("off")  # Ekseni kapatıyor.
plt.show()  # Grafiği gösteriyor.

# Configuring the Dataset for Performance (Veri Setini Performans için Yapılandırma)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)  # Veriyi önceden yükleme, karıştırma ve performans için optimize etme.
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)  # Doğrulama verisi de performans için optimize ediliyor.

# Data Augmentation (Veri Artırma)
data_augmentation = tf.keras.Sequential([  # Veri artırma işlemleri için bir model tanımlanıyor.
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),  # Resimleri rastgele yatay ve dikey olarak çeviriyor.
    tf.keras.layers.RandomRotation(0.1, seed=42),  # Resimleri rastgele %10 oranında döndürüyor.
])

plt.figure(figsize=(10, 10))  # 10x10 boyutunda bir grafik oluşturuyor.
for images, _ in train_ds.take(1):  # Eğitim setinden bir batch alıyor.
    for i in range(9):  # İlk 9 resmi veri artırma uygulayarak görselleştiriyor.
        augmented_images = data_augmentation(images)  # Veri artırmayı resimlere uyguluyor.
        ax = plt.subplot(3, 3, i + 1)  # 3x3 bir grid oluşturuyor.
        plt.imshow(augmented_images[0].numpy().astype("uint8"))  # Artırılmış resmi gösteriyor.
        plt.axis("off")  # Ekseni kapatıyor.
plt.show()  # Grafiği gösteriyor.

# Building the Model (Modeli Kurma)
NUM_CLASSES = len(class_names)  # Sınıf sayısını alıyor (örneğin 3 sınıf: rock, paper, scissors).

model = tf.keras.Sequential([  # Sıralı bir model oluşturuyor.
    layers.Input(shape=(180, 180, 3)),  # Giriş katmanı, 180x180 boyutunda 3 kanallı (RGB) resimleri alıyor.
    tf.keras.layers.Rescaling(1. / 255),  # Resimleri [0, 1] aralığına ölçeklendiriyor.
    data_augmentation,  # Veri artırma işlemlerini modele dahil ediyor.
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    # 128 filtreli bir evrişim katmanı, 3x3 filtre boyutu ile. 'same' padding kullanarak çıkış boyutunu koruyor.
    tf.keras.layers.MaxPooling2D(),  # 2x2 boyutunda maksimum havuzlama katmanı, özellik haritasını küçültüyor.
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),  # İkinci evrişim katmanı.
    tf.keras.layers.MaxPooling2D(),  # Maksimum havuzlama katmanı.
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),  # Üçüncü evrişim katmanı.
    tf.keras.layers.MaxPooling2D(),  # Maksimum havuzlama katmanı.
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),  # Dördüncü evrişim katmanı.
    tf.keras.layers.MaxPooling2D(),  # Maksimum havuzlama katmanı.
    tf.keras.layers.Flatten(),  # Özellik haritalarını tek boyutlu bir diziye düzleştiriyor.
    tf.keras.layers.Dense(512, activation='relu'),  # 512 nöronlu tam bağlantılı katman, ReLU aktivasyonu kullanıyor.
    tf.keras.layers.Dense(NUM_CLASSES)  # Sınıflandırma katmanı, sınıf sayısı kadar çıkış (örneğin 3 sınıf).
])
model.summary()  # Modelin özetini ekrana yazdırıyor (katmanların yapısı ve parametre sayıları).

model.compile(optimizer='adam',  # Adam optimizasyon algoritması kullanılıyor.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Kategorik çapraz entropi kaybı, etiketler sparse (tek numara ile temsil ediliyor).
              metrics=['accuracy'])  # Modelin başarım ölçütü olarak doğruluk kullanılıyor.

# Modeli eğitme
history = model.fit(train_ds, validation_data=val_ds, epochs=15)  # Modeli 15 epoch boyunca eğitim ve doğrulama setlerinde eğitiyor.

# Visualizing Training Results (Eğitim Sonuçlarını Görselleştirme)
acc = history.history['accuracy']  # Eğitim doğruluğunu alıyor.
val_acc = history.history['val_accuracy']  # Doğrulama doğruluğunu alıyor.

loss = history.history['loss']  # Eğitim kaybını alıyor.
val_loss = history.history['val_loss']  # Doğrulama kaybını alıyor.

plt.figure(figsize=(10, 10))  # 10x10 boyutunda bir grafik oluşturuyor.
plt.subplot(1, 2, 1)  # 1. grafiği ekliyor.
plt.plot(acc, label='Training Accuracy')  # Eğitim doğruluğunu grafikte çiziyor.
plt.plot(val_acc, label='Validation Accuracy')  # Doğrulama doğruluğunu grafikte çiziyor.
plt.legend(loc='lower right')  # Grafiğin açıklama kısmı.
plt.title('Training and Validation Accuracy')  # Grafik başlığı.

plt.subplot(1, 2, 2)  # 2. grafiği ekliyor.
plt.plot(loss, label='Training Loss')  # Eğitim kaybını grafikte çiziyor.
plt.plot(val_loss, label='Validation Loss')  # Doğrulama kaybını grafikte çiziyor.
plt.legend(loc='upper right')  # Grafiğin açıklama kısmı.
plt.title('Training and Validation Loss')  # Grafik başlığı.
plt.show()  # Grafikleri gösteriyor.

# Predicting on New Data (Yeni Veride Tahmin Yapma)
random_image = random.sample(os.listdir(paper_dir), 1)  # Paper dizininden rastgele bir resim seçiyor.
img = tf.keras.utils.load_img(f"{paper_dir}/{random_image[0]}", target_size=(180, 180))  # Resmi 180x180 boyutuna yeniden boyutlandırarak yüklüyor.

img_array = tf.keras.utils.img_to_array(img)  # Resmi NumPy dizisine çeviriyor.
img_array = tf.expand_dims(img_array, 0)  # 4 boyutlu bir tensör oluşturmak için batch boyutu ekliyor.

predictions = model.predict(img_array)  # Modelle tahmin yapıyor.
score = tf.nn.softmax(predictions[0])  # Tahmin edilen değerleri softmax fonksiyonu ile olasılığa çeviriyor.

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))  # En olası sınıfı ve güven yüzdesini yazdırıyor.
)

# Veri Ön İşleme: TensorFlow ile görüntüleri nasıl yükleyeceğin, yeniden boyutlandırma ve normalizasyon gibi işlemleri nasıl yapacağını öğrenmelisin.
# Veri Artırma (Data Augmentation): Eğitim verilerini artırma tekniklerini (döndürme, çevirme vb.) nasıl uygulayacağını öğrenmelisin.
# Model Kurulumu: Keras kullanarak CNN tabanlı bir görüntü sınıflandırma modelini nasıl oluşturacağını anlamalısın.
# Eğitim ve Doğrulama: Modelin eğitimi, doğrulama verileriyle nasıl test edileceğini ve eğitim sonuçlarını nasıl analiz edebileceğini öğrenmelisin.
# Tahmin Yapma: Eğitilmiş modeli yeni veriler üzerinde nasıl kullanacağını ve tahminlerin nasıl yorumlanacağını kavramalısın.
