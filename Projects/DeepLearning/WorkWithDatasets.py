import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt

xTensor = tf.range(5)  # # tf.range(5)  -> 0'dan 4'e kadar sayılardan oluşan bir tensor oluşturur.
dataset = tf.data.Dataset.from_tensor_slices(xTensor)  # # Tensorflow veri seti API'si ile tensor verilerini parça parça (slice) ederek bir veri seti oluşturur.

for item in dataset:
    print("Değerler: ", item.numpy())

print("Tensor Çıktısı: ", item)

print("numpy Metodu İle Tensor'ü Numpy'a Çevirme: ", item.numpy())

print("as_numpy_iterator Metodu İle Tensor'ü Numpy'a Çevirme: ", list(dataset.as_numpy_iterator()))

print("Elemanların Özellikleri: ", dataset.element_spec)

# dataset objesi oluşturma: Bu tensor rastgele [3, 5] boyutlarında (3 satır ve 5 sütun) elemanlar oluşturur.
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([3, 5]))
# dataset objesinin elemanları;
print("Dataset Objesinin Elemanları: ", dataset1.element_spec)  # TensorSpec(shape=(5,), dtype=tf.float32, name=None)
# 1 Boyutlu, 5 elemanlı float32 tipinde tensor oluşturulmuş.

weatherData = [5, 1, -3, -2, -4, 7, -10]
weatherTFdataset = tf.data.Dataset.from_tensor_slices(weatherData)  # Yukarıdaki weatherData listesi, bir veri setine çevriliyor.
# weatherTFdataset'deki elemanları numpy olarak yazdırır.
for item in weatherTFdataset:
    print("weatherTFdataset Elemanları(Numpy Türünde): ", item.numpy())

# weatherTFdataset'deki elemanların ilk 3 tanesini yazdırır.
for item in weatherTFdataset.take(3):
    print("weatherTFdataset'deki Elemanların İlk 3 Tanesi: ", item.numpy())

# 0'dan büyük olan elemanları yazdırır. print(item) tensor, print(item.numpy()) numpy olarak yazdırır.
for item in weatherTFdataset.filter(lambda x: x > 0):
    print("0'dan Büyük Olan Elemanlar: ", item)

# map(): Herhangi bir fonksiyonu elemanlara uygular. Aşağıda her bir elemanı 2 ile çarptık ve ekrana yazdırdık.
for item in weatherTFdataset.map(lambda x: x * 2):
    print("map Fonksiyonu: ", item.numpy())

# shuffle(): elemanları karıştırır. buffer_size=elemanlardan 3 tanesini belleğe alır karıştırmak için.
for item in weatherTFdataset.shuffle(buffer_size=3):
    print("shuffle Fonksiyonu: ", item.numpy())

# batch(): elemanları gruplar. Verileri sinir ağına bir anda vermek yerine gruplar halinde vermemizi sağlar.
for item in weatherTFdataset.batch(2):
    print("batch Fonksiyonu: ", item.numpy())

# Pipeline oluşturma: Yukarıdaki metotları tek bir şekilde yazabiliyoruz;
pipelineDataset = tf.data.Dataset.from_tensor_slices(weatherData)
pipelineDataset = dataset.filter(lambda x: x > 0).map(lambda y: y * 2).shuffle(3).batch(2)
for item in pipelineDataset:
    print(item.numpy())

flowers_root = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/flower_photos"

print("flower_photos Dizini: ", flowers_root)

flowers_root = pathlib.Path(flowers_root)  # Verilen değişkeni Path'e çeviriyor.
# pathlib.Path() -> Dosya yolunu platform bağımsız hale getirir, file path'i Path objesine çevirir.

print("Path'e çevrilmiş flower_photos:", flowers_root)

# Resimlerin bulunduğu klasördeki dosyaları listeler. str(flowers_root / "*/*") -> Alt klasörlerdeki dosyaları da dahil eder.
ds = tf.data.Dataset.list_files(str(flowers_root / "*/*"))

image_count = sum(1 for _ in ds)

print("Resim Sayısı: ", image_count)  # Veri setinde 3670 tane resim var.
# for döngüsü ile ilk 3 değeri ekrana yazdırıyoruz;
for file in ds.take(3):
    print("İlk 3 Değer: ", file.numpy())

# Shuffle ederek 3 tane veriyi getirdik;
ds = ds.shuffle(200)
for file in ds.take(3):
    print("Shuffle Edilip Verilen 3 Veri: ", file.numpy())

# Veriyi Train Test Olarak Ayırma;
train_size = int(image_count * 0.8)  # Verilerin %80'i train için ayrılıyor.
train_ds = ds.take(train_size)  # Eğitim için ilk %80 veriyi al.
test_ds = ds.skip(train_size)  # Geri kalan %20'yi test veri seti olarak al.

print("Eğitim Veri Seti: ", len(train_ds))  # Eğitim veri seti: 2936
print("Test Veri Seti: ", len(test_ds))  # Test veri seti: 734


# # Dizinden resimlerin etiketlerini elde etme fonksiyonu yazdık get_label ile.
def get_label(file_path):
    parts = tf.strings.split(file_path, os.sep)
    return parts[-2]


# Yukarıdaki metodun çalışıp çalışmadığını kontrol ediyoruz;
x = b'C:\\Users\\firat\\OneDrive\\Belgeler\\Projects\\PythonMLProjects\\Projects\\Gerekliler\\flower_photos\\tulips\\15275504998_ca9eb82998.jpg'

print(get_label(x))  # tf.Tensor(b'tulips', shape=(), dtype=string)
print(get_label(x).numpy())  # b'tulips'


# Veri Önişleme: Resimleri okuma, boyut değiştirme ve ölçeklendirme için veri ön işleme yapıyoruz.
# Resimleri okuma, boyutlandırma ve ölçeklendirme fonksiyonu yazdık process_image ile.
def process_image(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    img = img / 255
    return img, label


img, label = process_image(x)  # Bu satır, daha önce tanımlanan process_image fonksiyonunu kullanarak, resim dosyasını işler ve etiketi ile birlikte döndürür.

print("Image Shape: ", img.shape)
print("Image 3,3,1: ", img[:3, :3, :1])

plt.imshow(img.numpy())
plt.title(label.numpy())
plt.show()

# Eğitim ve test veri setlerinde resimleri işlemden geçiriyoruz.
train_ds = train_ds.map(process_image)
test_ds = test_ds.map(process_image)

# # train_ds'den bir örnek alıp görüntü ve etiketi yazdırıyoruz.
for image, label in train_ds.take(1):
    print("Resim: ", label.numpy())
    print("Etiket: ", image.numpy())

# Açıklama: Bu kod, TensorFlow kullanarak bir veri setini işler ve çeşitli veri ön işleme tekniklerini uygular.
# İlk olarak TensorFlow veri seti API'si ile basit bir tensor veri seti oluşturulur ve ardından veri üzerinde filtreleme, map, shuffle gibi işlemler yapılır.
# Daha sonra bir resim veri seti indirilip, resimler okunur, boyutlandırılır ve normalleştirilir. Eğitim ve test veri setleri oluşturulur.
# Resimlerin etiketlenmesi, etiketlerin dosya yollarından elde edilmesi ve veri setlerinin train-test olarak ayrılması sağlanır.

# Bu kod, özellikle makine öğrenmesi modelleri için veri ön işleme işlemlerini gösterir.
# Resimlerin boyutlandırılması, normalleştirilmesi ve etiketlenmesi gibi adımları içerir.
# Bu süreç, bir modelin eğitim aşaması öncesinde uygulanır ve modelin giriş verileri hazır hale getirilir.
