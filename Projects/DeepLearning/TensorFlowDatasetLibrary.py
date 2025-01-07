import tensorflow as tf  # TensorFlow kütüphanesini içe aktarıyor, makine öğrenmesi ve derin öğrenme için kullanılır.
import tensorflow_datasets as tfds  # TensorFlow Datasets kütüphanesini içe aktarıyor, önceden hazırlanmış veri setlerini kullanmak için.
import matplotlib.pyplot as plt  # Matplotlib kütüphanesinden grafik çizmeye yarayan pyplot modülünü içe aktarıyor.

# TensorFlow Datasets'te bulunan tüm hazır veri setlerinin listesini döndürüyor.
tfds.list_builders()  # Tüm kullanılabilir veri setlerinin isimlerini listeler.

# "fashion_mnist" veri setini yüklüyor ve sadece eğitim kısmını alıyor.
mnist_train = tfds.load(name="fashion_mnist", split="train")  # name: veri setinin ismi, split: hangi bölümü istediğimiz (train/test/validation).

# mnist_train nesnesinin tf.data.Dataset türünde olup olmadığını kontrol ediyor.
isinstance(mnist_train, tf.data.Dataset)  # Veri setinin doğru türde olup olmadığını kontrol eder.
print(mnist_train)  # Veri setinin genel yapısını yazdırır.

# Eğitim veri setindeki ilk öğeyi alıp anahtarlarını yazdırıyor.
for item in mnist_train.take(1):  # .take(1): Veri setinden 1 adet öğe alır.
    print(type(item))  # item nesnesinin türünü yazdırır.
    print(item.keys())  # Veri setindeki örnek öğenin anahtarlarını yazdırır (ör. "image", "label").

# İlk öğenin "image" ve "label" anahtarlarına karşılık gelen şekilleri (boyutlarını) yazdırıyor.
for item in mnist_train.take(1):
    print(item["image"].shape)  # Görüntünün (image) boyutlarını yazdırır.
    print(item["label"].shape)  # Etiketin (label) boyutunu yazdırır.

# "fashion_mnist" veri setini yüklüyor ve bilgi içeriğiyle birlikte döndürüyor.
mnist_test, info = tfds.load(name="fashion_mnist", with_info=True)  # with_info: Veri setiyle ilgili bilgileri de döndürür.
print(info)  # Veri seti hakkındaki bilgileri yazdırır (ör. sınıf sayısı, örnek sayısı vb.).

# TensorFlow'un dahili Fashion MNIST veri setini yükleyip eğitim ve test setlerini ayırıyor.
mnist = tf.keras.datasets.fashion_mnist  # TensorFlow'da yerleşik Fashion MNIST veri setini kullanır.
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()  # Eğitim ve test verilerini yükler.

ds = tfds.load(name="fashion_mnist", split="train", as_supervised=True)  # Veri setini etiketleriyle eşleşmiş halde yükler.

ds = ds.take(1)  # Veri setinden 1 örnek alır.
for image, label in tfds.as_numpy(ds):  # TensorFlow veri setini NumPy dizisine dönüştürür.
    print(type(image), type(label), label)  # Görüntü ve etiketin türünü ve etiketi yazdırır.

# Tüm test veri setini tek bir seferde yükleyip NumPy dizilerine dönüştürüyor.
image, label = tfds.as_numpy(
    tfds.load(
        name="fashion_mnist",  # Veri seti ismi.
        split="test",  # Test bölümü.
        batch_size=-1,  # Tüm veriyi tek bir batch olarak döndürür.
        as_supervised=True  # Görüntü ve etiketi eşleştirilmiş olarak döndürür.
    )
)

print(type(image), image.shape)  # Görüntülerin türünü ve boyutlarını yazdırır.

# Eğitim veri setini ve veri setiyle ilgili bilgileri döndürüyor.
dataset, dataset_info = tfds.load(name="fashion_mnist", split="train", with_info=True)  # split="train": Eğitim verilerini alır.


# Görselleri çizen bir fonksiyon tanımlıyor.
def plot_images(data_set, num_images=4):  # data_set: Görüntü ve etiket içeren veri seti, num_images: Çizilecek görüntü sayısı.
    plt.figure(figsize=(10, 10))  # Grafik boyutunu ayarlar.
    for i, data in enumerate(data_set.take(num_images)):  # Belirtilen sayıda örnek alır.
        img, lbl = data['image'], data['label']  # Görüntü ve etiketi alır.
        plt.subplot(1, num_images, i + 1)  # Alt grafik konumunu belirler.
        plt.imshow(img.numpy().squeeze(), cmap='gray')  # Görüntüyü siyah-beyaz olarak çizer.
        plt.title(f'Label: {lbl.numpy()}')  # Görüntünün etiketini başlık olarak yazar.
        plt.axis('off')  # Eksenleri kapatır.
    plt.show()  # Grafiği gösterir.


# Eğitim veri setindeki ilk birkaç görüntüyü çiziyor.
plot_images(dataset)  # Görüntüleri çizen fonksiyonu çağırır.
