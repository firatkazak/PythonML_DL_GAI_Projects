import tensorflow as tf  # TensorFlow kütüphanesi içe aktarılıyor, derin öğrenme işlemleri için kullanılıyor
import tensorflow_datasets as tfds  # TensorFlow Datasets kütüphanesi, hazır veri kümeleri için kullanılıyor
import re  # Düzenli ifadeler (regex) ile metin işlemleri için kullanılıyor
import string  # Python'un string işlemleri için gerekli kütüphanesi
import io  # Dosya işlemleri için kullanılan bir kütüphane

# IMDb film inceleme verileri yükleniyor ve train, validation ve test setlerine bölünüyor
raw_train_ds, raw_val_ds, raw_test_ds = tfds.load(name="imdb_reviews", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True)
# name="imdb_reviews": Yüklenen veri IMDb incelemeleri.
# split: Verinin %90'ı eğitim, %10'u doğrulama, geri kalanı test seti olarak ayrılıyor.
# as_supervised=True: Veriler hem giriş (inceleme), hem de çıkış (etiket) olarak döndürülüyor.

# Eğitim verisinden 3 örnek alınıyor ve her bir incelemenin ve etiketin içeriği yazdırılıyor
for review, label in raw_train_ds.take(3):  # İlk 3 örnek alınıyor
    print(review.numpy().decode("utf-8"))  # İnceleme metni (byte formatından utf-8'e dönüştürülüyor)
    print("Label: ", label.numpy())  # Etiket (0=negatif, 1=pozitif)

# Rastgeleliği kontrol altına almak için bir sabit seed değeri belirleniyor
tf.random.set_seed(42)

AUTOTUNE = tf.data.AUTOTUNE  # TensorFlow otomatik ayarlamayı etkinleştiriyor, performans için veri işlemeyi optimize ediyor

# Eğitim verisi karıştırılıyor, batch'lere ayrılıyor ve otomatik olarak işleniyor
train_ds = raw_train_ds.shuffle(5000, seed=42).batch(32).prefetch(buffer_size=AUTOTUNE)
# shuffle(5000): Veriyi karıştırmak için 5000 örnek kullanıyor.
# batch(32): 32'lik küçük gruplar halinde veriler işlenecek.
# prefetch(AUTOTUNE): Eğitim sırasında bir sonraki veriyi hazır tutmak için prefetch yapılıyor.

# Doğrulama verisi de batch'lenip, prefetch işlemi ile optimize ediliyor
val_ds = raw_val_ds.batch(32).prefetch(buffer_size=AUTOTUNE)


# Özel bir metin standartlaştırma fonksiyonu tanımlanıyor (küçük harfe çevirme ve noktalama işaretlerini temizleme)
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)  # Metni küçük harfe çeviriyor
    stripped_html = tf.strings.regex_replace(lowercase, "", "")  # HTML etiketlerini temizliyor
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), "")  # Noktalama işaretlerini kaldırıyor


# TextVectorization katmanı oluşturuluyor, metinler sayısal dizilere çevrilecek
vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,  # Standartlaştırma işlemi için özel fonksiyon kullanılıyor
    max_tokens=10000,  # Toplamda en fazla 10.000 benzersiz kelime işlenecek
    output_mode="int",  # Çıkış modunda kelimeler tamsayılar olarak vektörleştirilecek
    output_sequence_length=100  # Her metin en fazla 100 kelime uzunluğunda olacak
)

# Eğitim verisindeki metinlerin çıkartıldığı bir veri seti oluşturuluyor (etiketler çıkarılıyor)
text_ds = train_ds.map(lambda x, y: x)
# Veriler yalnızca metin (x) olacak şekilde dönüştürülüyor

# TextVectorization katmanı veriye adapte ediliyor (kelimeler vektörleştiriliyor)
vectorize_layer.adapt(text_ds)

# Model oluşturuluyor
model = tf.keras.Sequential([  # Sıralı model oluşturuluyor
    vectorize_layer,  # İlk katman TextVectorization, metinleri vektöre dönüştürüyor
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16, name="embedding"),
    # input_dim=10000: Toplamda 10.000 benzersiz kelime olacak.
    # output_dim=16: Her kelime 16 boyutlu bir vektörle temsil edilecek.
    tf.keras.layers.GlobalAveragePooling1D(),
    # Embedding vektörlerini ortalayan bir havuzlama katmanı (boyut azaltma için kullanılır)
    tf.keras.layers.Dense(units=16, activation="relu"),
    # Tam bağlantılı katman, 16 nöron ve ReLU aktivasyon fonksiyonu kullanıyor
    tf.keras.layers.Dense(1)
    # Son katman, pozitif ya da negatif sınıfı tahmin etmek için tek bir çıktı nöronu var
])

# Modelin yapısı yazdırılıyor
model.summary()

# Model derleniyor, optimizasyon ve kayıp fonksiyonu belirleniyor
model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
# optimizer="adam": Eğitim sırasında parametreleri optimize eden algoritma
# loss="BinaryCrossentropy": İki sınıflı (pozitif/negatif) sınıflandırma problemi için kullanılan kayıp fonksiyonu
# from_logits=True: Modelin son çıktısı aktivasyon fonksiyonuna girmeden önce logits formatında olacak

# TensorBoard ile eğitim süreci izlenebilmesi için callback tanımlanıyor
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/logs")
# logs: TensorBoard ile gözlemlenebilecek eğitim metriklerinin kaydedileceği yerel dizin

# Model eğitiliyor
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[tensorboard_callback])
# train_ds: Eğitim verisi
# validation_data: Doğrulama verisi
# epochs=5: Eğitim 5 döngü boyunca yapılacak
# callbacks: TensorBoard callback'i eğitim sırasında kullanılacak

# Eğitilen modelin embedding ağırlıkları alınıyor
weights = model.get_layer("embedding").get_weights()[0]
# Embedding katmanının ağırlıkları (kelime vektörleri) alınıyor

# Kelime listesi alınıyor (vocabulary)
vocab = vectorize_layer.get_vocabulary()

# Embedding vektörleri TSV dosyasına yazılıyor
with io.open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/vectors.tsv", "w", encoding="utf-8") as out_v:
    for row in weights:
        out_v.write("\t".join([str(x) for x in row]) + "\n")

# Kelime listesi (vocabulary) TSV dosyasına yazılıyor
with io.open("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/metadata.tsv", "w", encoding="utf-8") as out_m:
    for word in vocab:
        out_m.write(word + "\n")

# Bilgi mesajı yazdırılıyor
print("vectors.tsv ve metadata.tsv dosyaları yerel dizine kaydedildi.")

# Bu kodun amacı IMDb film inceleme verilerini kullanarak bir metin sınıflandırma modeli eğitmektir.
# Embedding katmanı kullanarak kelimeleri sayısal temsillere çeviriyor ve bu temsillerle pozitif/negatif yorumları sınıflandıran bir model oluşturuluyor.
# Ayrıca, embedding vektörleri dosyaya yazdırılarak görselleştirilebilir hale getiriliyor.

# KODUN AÇIKLAMASI
# Veri Hazırlığı: IMDb film incelemeleri veri seti TensorFlow Datasets aracılığıyla yükleniyor. Eğitim, doğrulama ve test setlerine bölünüyor.
# Özelleştirilmiş Metin İşleme: Kelimeler küçük harfe çevrilip noktalama işaretlerinden arındırılıyor.
# Embedding: Her kelime bir embedding vektörüne dönüştürülüyor.
# Model Yapısı: Bir derin öğrenme modeli oluşturuluyor. Bu model, metin vektörlerini giriş alıp, incelemenin pozitif mi negatif mi olduğunu tahmin ediyor.
# Eğitim ve Değerlendirme: Model 5 dönem boyunca eğitim yapılıyor ve TensorBoard ile süreç izleniyor.
# Embedding Vektörlerinin Çıktısı: Embedding vektörleri ve kelime listesi dosyalara kaydediliyor, bu sayede daha sonra analiz edilebiliyor.

# Ne Öğrendik:
# TextVectorization ve Embedding katmanlarının nasıl çalıştığını anlamalısın.
# Sequential Model ve katmanların eklenmesi.
# Model eğitimi ve değerlendirme süreçlerini kavramalısın.
# TensorBoard ile eğitim metriklerini nasıl izleyebileceğini öğrenmelisin.
