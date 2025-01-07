import tensorflow as tf  # TensorFlow ana kütüphanesi.
from keras import layers  # Sinir ağı katmanlarını oluşturmak için kullanılır.
import os  # Dosya ve dizin işlemleri için kullanılır.
import numpy as np  # Sayısal işlemler ve matris hesaplamaları için.
import matplotlib.pyplot as plt  # Grafik çizmek için kullanılır.
import matplotlib.image as implt  # Görüntü işlemek ve göstermek için.
import seaborn as sns  # İstatistiksel veri görselleştirme için.
import pandas as pd  # Veri manipülasyonu ve analizi için.

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Görüntü verilerini ön işlemek ve augmentasyon (veri genişletme) yapmak için kullanılır.
from tensorflow.keras.preprocessing import image  # Görüntü işleme yardımcı fonksiyonlarını sağlar.
from tensorflow.keras.optimizers import RMSprop  # RMSprop optimizasyon algoritması.

# Data Paths; # Eğitim ve doğrulama verilerinin dosya yolları.
train_path = 'C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/horse-or-human/train'  # Eğitim verileri yolu.
val_path = 'C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/horse-or-human/validation'  # Doğrulama verileri yolu.
train_horses = 'C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/horse-or-human/train/horses'  # Eğitimdeki at resimleri yolu.
train_humans = 'C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/horse-or-human/train/humans'  # Eğitimdeki insan resimleri yolu.

# Grafik ayarları
sns.set_style("whitegrid")  # Grafiklerde beyaz grid stili kullanılır.
colors = {'humans': 'blue', 'horses': 'orange'}  # İnsanlar ve atlar için renk tanımları.

# Understanding the data
category_names = os.listdir(train_path)  # Eğitim klasöründeki kategorileri listeler (['humans', 'horses']).
nb_categories = len(category_names)  # Kategori sayısını alır (2 kategori: humans ve horses).

# Train images
train_images = []  # Eğitim verilerinin sayısını tutacak liste.
for category in category_names:  # Her kategori için.
    folder = os.path.join(train_path, category)  # Kategori klasörünün yolunu oluşturur.
    num_images = len(os.listdir(folder))  # Klasördeki görüntü sayısını alır.
    train_images.append({'category': category, 'num_images': num_images})  # Kategori ve resim sayısını listeye ekler.

df_train = pd.DataFrame(train_images)  # Eğitim verilerini bir DataFrame'e çevirir.

sns.barplot(x='num_images',
            y='category',
            data=df_train,
            hue='category',
            palette=colors
            )  # Eğitim verilerinin kategorilere göre dağılımını gösteren bir çubuk grafik çizer.

plt.title("Number Of Training Images Per Category")  # Grafik başlığı.
plt.show()  # Grafiği gösterir.

# Validation images
val_images = []  # Doğrulama verilerinin sayısını tutacak liste.
for category in category_names:  # Her kategori için.
    folder = os.path.join(val_path, category)  # Kategori klasörünün yolunu oluşturur.
    num_images = len(os.listdir(folder))  # Klasördeki görüntü sayısını alır.
    val_images.append({'category': category, 'num_images': num_images})  # Kategori ve resim sayısını listeye ekler.

df_val = pd.DataFrame(val_images)  # Doğrulama verilerini bir DataFrame'e çevirir.

sns.barplot(x='num_images',
            y='category',
            data=df_val,
            hue='category',
            palette=colors
            )  # Doğrulama verilerinin kategorilere göre dağılımını gösteren bir çubuk grafik çizer.

plt.title("Number Of Validation Images Per Category")  # Grafik başlığı.
plt.show()  # Grafiği gösterir.

# Resim gösterimi
img1 = implt.imread(train_horses + "/horse02-0.png")  # At resmi yüklenir.
img2 = implt.imread(train_humans + "/human02-00.png")  # İnsan resmi yüklenir.

plt.subplot(1, 2, 1)  # 1x2'lik alt grafiklerin ilkine geçilir.
plt.title('horse')  # Başlık atılır.
plt.imshow(img1)  # At resmi gösterilir.
plt.subplot(1, 2, 2)  # 1x2'lik alt grafiklerin ikincisine geçilir.
plt.title('human')  # Başlık atılır.
plt.imshow(img2)  # İnsan resmi gösterilir.
plt.show()  # Grafikleri gösterir.

# Data Preprocessing (Veri Ön İşleme)
train_datagen = ImageDataGenerator(rescale=1. / 255,  # Tüm pikselleri [0, 1] aralığına ölçekler.
                                   rotation_range=40,  # Resimlerin rastgele döndürülmesi için derece aralığı.
                                   width_shift_range=0.2,  # Rastgele genişlik kaydırma aralığı.
                                   height_shift_range=0.2,  # Rastgele yükseklik kaydırma aralığı.
                                   shear_range=0.2,  # Rastgele kayma işlemi (shearing).
                                   zoom_range=0.2,  # Rastgele zoom işlemi.
                                   horizontal_flip=True,  # Resimleri yatayda rastgele çevirir.
                                   fill_mode='nearest'  # Yeni pikselleri en yakın değerle doldurur.
                                   )

train_generator = train_datagen.flow_from_directory(train_path,  # Eğitim veri yolundan veri yükler.
                                                    target_size=(300, 300),  # Resimleri 300x300 boyutunda yeniden boyutlandırır.
                                                    batch_size=128,  # Her seferde işlenecek görüntü sayısı.
                                                    class_mode='binary'  # İkili sınıflandırma için sınıf modu.
                                                    )

validation_datagen = ImageDataGenerator(rescale=1 / 255)  # Doğrulama verileri için sadece yeniden ölçekleme yapılır.

validation_generator = train_datagen.flow_from_directory(val_path,  # Doğrulama veri yolundan veri yükler.
                                                         target_size=(300, 300),  # Resimleri 300x300 boyutunda yeniden boyutlandırır.
                                                         class_mode='binary'  # İkili sınıflandırma için sınıf modu.
                                                         )

# Model Building (Model Oluşturma)
cnn_model = tf.keras.models.Sequential([  # Sequential model oluşturuluyor (katmanlar sırayla ekleniyor).
    layers.Input(shape=(300, 300, 3)),  # Giriş katmanı, 300x300 boyutunda ve 3 kanallı (renkli) resim kabul eder.
    layers.Conv2D(16, 3, activation='relu'),  # 16 filtreli, 3x3 boyutunda konvolüsyon katmanı, ReLU aktivasyonu ile.
    layers.MaxPooling2D(2, 2),  # 2x2 boyutunda max pooling katmanı (boyut azaltma).
    layers.Conv2D(32, 3, activation='relu'),  # 32 filtreli ikinci konvolüsyon katmanı.
    layers.MaxPooling2D(2, 2),  # 2x2 boyutunda pooling.
    layers.Conv2D(64, 3, activation='relu'),  # 64 filtreli üçüncü konvolüsyon katmanı.
    layers.MaxPooling2D(2, 2),  # Pooling işlemi.
    layers.Conv2D(64, 3, activation='relu'),  # 64 filtreli dördüncü konvolüsyon katmanı.
    layers.MaxPooling2D(2, 2),  # Pooling işlemi.
    layers.Conv2D(64, 3, activation='relu'),  # 64 filtreli beşinci konvolüsyon katmanı.
    layers.MaxPooling2D(2, 2),  # Pooling işlemi.
    layers.Flatten(),  # Veriyi düzleştirir (dense katmanlar için).
    layers.Dense(units=512, activation='relu'),  # 512 nöronlu tam bağlantılı katman.
    layers.Dense(units=1, activation='sigmoid')  # 1 nöronlu çıktı katmanı, sigmoid aktivasyonu (ikili sınıflandırma için).
])
cnn_model.summary()  # Modelin özetini gösterir.

cnn_model.compile(loss='binary_crossentropy',  # İkili sınıflandırma için uygun kayıp fonksiyonu.
                  optimizer=RMSprop(learning_rate=0.001),  # RMSprop optimizasyon algoritması, öğrenme hızı 0.001.
                  metrics=['acc']  # Doğruluk (accuracy) metriği ile değerlendirilir.
                  )

# Model Training (Model Eğitimi)
history = cnn_model.fit(train_generator,  # Eğitim verisi jeneratörü.
                        steps_per_epoch=8,  # Her epoch'ta işlenecek adım sayısı.
                        epochs=15,  # Eğitim için toplam epoch sayısı.
                        verbose=1,  # Eğitim süreci ile ilgili çıktıların gösterilmesi.
                        validation_data=validation_generator  # Doğrulama verisi jeneratörü.
                        )

plt.figure(figsize=(12, 8))  # Grafik boyutu ayarlanır.

# Plotting the Training and Validation Loss (Eğitim ve Doğrulama Kaybı Grafiği)
plt.plot(history.history['loss'], label='train loss')  # Eğitim kaybı grafiği.
plt.plot(history.history['val_loss'], label='val loss')  # Doğrulama kaybı grafiği.
plt.legend()  # Grafikteki etiketler.
plt.xlabel("Epochs")  # X ekseni etiketi.
plt.ylabel("Loss")  # Y ekseni etiketi.

# Plotting the Training and Validation Accuracy (Eğitim ve Doğrulama Doğruluğu Grafiği)
plt.figure(figsize=(12, 8))  # Grafik boyutu ayarlanır.
plt.plot(history.history['acc'], label='train acc')  # Eğitim doğruluğu grafiği.
plt.plot(history.history['val_acc'], label='val acc')  # Doğrulama doğruluğu grafiği.
plt.legend()  # Grafikteki etiketler.
plt.xlabel("Epochs")  # X ekseni etiketi.
plt.ylabel("Accuracy")  # Y ekseni etiketi.

# Prediction (Tahmin)
# Creating a path variable: (Resim yolunu oluşturma)
path = os.path.join(val_path, 'humans', 'valhuman01-01.png')  # Tahmin edilecek resmin yolunu oluşturur.

# Loading the image and resizing (Resmi yükleme ve yeniden boyutlandırma)
img = image.load_img(path, target_size=(300, 300))  # Resmi yükler ve 300x300 boyutuna yeniden boyutlandırır.
x = image.img_to_array(img)  # Resmi bir numpy dizisine çevirir.
x = np.expand_dims(x, axis=0)  # Diziyi bir batch boyutunda genişletir (CNN giriş formatına uygun hale getirir).
plt.imshow(img)  # Resmi gösterir.
plt.axis('off')  # Eksenleri gizler.
plt.show()  # Grafiği gösterir.

classes = cnn_model.predict(x)  # Model, resim üzerinde tahmin yapar.
if classes[0] > 0.5:  # Tahmin sonucuna göre çıktı.
    print("Image is a human")  # Sonuç insan ise yazdırılır.
else:
    print("Image is a horse")  # Sonuç at ise yazdırılır.

# AÇIKLAMA:
# Bu kod, derin öğrenme ve görüntü işleme alanında temel bir ikili sınıflandırma modelinin oluşturulması, eğitilmesi ve değerlendirilmesi için kullanılır.
# Özellikle, at ve insan görüntülerini ayırt etmek için bir konvolüsyonel sinir ağı (CNN) modelinin nasıl kurulduğunu gösterir.

# 1. Veri Hazırlama;
# Görüntü Verisi: Kod, iki farklı sınıfta (at ve insan) görüntüleri içeren bir veri kümesi kullanarak çalışır.
# ImageDataGenerator sınıfı ile veri augmentasyonu (veri genişletme) ve ön işleme yapılır.
# Klasör Yapısı: Eğitim ve doğrulama verilerinin klasör yapısının nasıl düzenlendiği ve bu verilerin nasıl yüklendiği hakkında bilgi verir.

# 2. Görselleştirme: Eğitim ve doğrulama verilerinin sayısını çubuk grafiklerle görselleştirme yeteneğini öğrendik.

# 3. Model Oluşturma;
# Konvolüsyonel Sinir Ağı (CNN): Modelin nasıl kurulduğunu ve katmanların (konvolüsyon, max pooling, dense) nasıl yapılandırıldığını öğrenirsiniz.
# Model, görüntü verilerini analiz etmek ve sınıflandırmak için kullanılır.
# Aktivasyon Fonksiyonları: ReLU ve sigmoid gibi aktivasyon fonksiyonlarının ne işe yaradığını öğrenirsiniz.

# 4. Model Eğitimi:
# Modelin nasıl eğitildiğini ve eğitim sırasında kayıp (loss) ve doğruluk (accuracy) gibi metriklerin nasıl izlendiğini görürsünüz.
# fit metodu ile modelin nasıl eğitildiği hakkında bilgi alırsınız.

# 5. Model Değerlendirme:
# Eğitim ve doğrulama kaybı ile doğruluk grafiklerinin nasıl çizildiğini ve bu grafiklerin modelin performansını nasıl değerlendirdiğini öğrenirsiniz.

# 6. Tahmin Yapma:
# Eğitilen modelin yeni bir görüntü üzerinde nasıl tahmin yaptığı, görüntünün nasıl yüklendiği ve ön işleme tabi tutulduğunu öğrenirsiniz.

# 7. Pratik Uygulama:
# Kendi veri kümenizi kullanarak benzer bir sınıflandırma problemi üzerinde nasıl çalışabileceğinizi öğrenirsiniz.

# Amaç:
# Kodun genel amacı, bir görüntü sınıflandırma problemi olan "at" ve "insan" görüntüleri arasında ayırım yapabilen bir makine öğrenimi modelinin oluşturulmasıdır.
# Bu, derin öğrenmenin görüntü işleme alanında nasıl uygulandığını gösterir ve daha karmaşık projeler için bir temel sağlar.
