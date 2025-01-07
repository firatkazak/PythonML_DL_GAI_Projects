import tensorflow as tf
import keras


# Custom Layer with Subclassing API
class Linear1(keras.layers.Layer):  # Linear1 adında bir katman oluşturuyoruz.
    def __init__(self, units=32, input_dim=64):  # units: çıkış nöron sayısı, input_dim: giriş özelliklerinin boyutu
        super(Linear1, self).__init__()  # Üst sınıfın (keras.layers.Layer) __init__ metodunu çağırıyoruz.

        w_init = tf.random_normal_initializer()  # Ağırlıkları rastgele normal dağılıma göre başlatıyoruz.
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),  # input_dim x units boyutunda ağırlık matrisi
            trainable=True  # Bu ağırlıklar eğitim sırasında güncellenecek.
        )
        b_init = tf.zeros_initializer()  # Bias değerlerini sıfırla başlatıyoruz.
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"),  # Bias vektörü units boyutunda
            trainable=True  # Bu bias değerleri de eğitim sırasında güncellenecek.
        )

    def call(self, inputs):  # Model ileri doğru geçtiğinde bu metod çağrılır.
        return tf.matmul(inputs, self.w) + self.b  # inputs ile w matris çarpımı + bias döner.


# Örnek girdi tensoru oluşturuyoruz.
x = tf.ones((2, 2))  # 2x2 boyutunda tüm elemanları 1 olan bir tensor
print(x)

# Linear1 katmanını oluşturuyoruz.
linear_layer = Linear1(4, 2)  # 4 nöronlu bir çıkış katmanı, giriş boyutu 2.
y = linear_layer(x)  # x girdisi üzerinden Linear1 katmanını çalıştırıyoruz.
print(y)  # Sonuçlar (y) yazdırılıyor.

# Linear1 katmanının ağırlıkları (w, b) yazdırılıyor.
print(linear_layer.weights)


# Custom Layer using add_weight
class Linear2(keras.layers.Layer):  # Yeni bir katman (Linear2) tanımlıyoruz.
    def __init__(self, units=32, input_dim=32):  # units: çıkış nöron sayısı, input_dim: giriş boyutu
        super(Linear2, self).__init__()  # Üst sınıfın (keras.layers.Layer) __init__ metodunu çağırıyoruz.
        self.w = self.add_weight(shape=(input_dim, units),  # input_dim x units boyutunda ağırlık matrisi
                                 initializer="random_normal",  # Ağırlıklar normal dağılımla başlatılıyor.
                                 trainable=True)  # Eğitim sırasında güncellenebilir.
        self.b = self.add_weight(shape=(units,),  # units boyutunda bias vektörü
                                 initializer="zeros",  # Bias sıfırlarla başlatılıyor.
                                 trainable=True)  # Eğitim sırasında güncellenebilir.

    def call(self, inputs):  # Model ileri doğru geçtiğinde bu metod çalışır.
        return tf.matmul(inputs, self.w) + self.b  # inputs ile w matris çarpımı + bias döner.


# Örnek girdi tensoru
x = tf.ones((2, 2))  # 2x2 boyutunda tüm elemanları 1 olan bir tensor
linear_layer = Linear2(4, 2)  # 4 nöronlu çıkış katmanı, giriş boyutu 2
y = linear_layer(x)  # x girdisi üzerinden Linear2 katmanını çalıştırıyoruz.
print(y)  # Sonuçlar yazdırılıyor.


# Custom Layer without the shape of the inputs
class Linear3(keras.layers.Layer):  # Giriş boyutu önceden bilinmeyen katman
    def __init__(self, units=32):  # units: çıkış nöron sayısı
        super(Linear3, self).__init__()  # Üst sınıfın (keras.layers.Layer) __init__ metodunu çağırıyoruz.
        self.units = units  # units değişkenini kaydediyoruz.
        self.w = None  # Ağırlık başlatılmamış (build metodunda başlatılacak)
        self.b = None  # Bias başlatılmamış (build metodunda başlatılacak)

    def build(self, input_shape):  # Giriş boyutunu alınca çağrılır (ilk kez model çağrıldığında çalışır)
        self.w = self.add_weight(shape=(input_shape[-1], self.units),  # input_shape[-1] x units boyutunda ağırlık
                                 initializer="random_normal",  # Ağırlıklar normal dağılımla başlatılıyor.
                                 trainable=True)  # Eğitim sırasında güncellenebilir.
        self.b = self.add_weight(shape=(self.units,),  # units boyutunda bias vektörü
                                 initializer="random_normal",  # Bias normal dağılımla başlatılıyor.
                                 trainable=True)  # Eğitim sırasında güncellenebilir.

    def call(self, inputs):  # İleri geçiş metodu
        return tf.matmul(inputs, self.w) + self.b  # inputs ile w matris çarpımı + bias döner.


linear_layer = Linear3(32)  # 32 nöronlu çıkış katmanı (giriş boyutu sonradan öğrenilecek)
y = linear_layer(x)  # x girdisi üzerinden Linear3 katmanını çalıştırıyoruz.
print(y)  # Sonuçlar yazdırılıyor.


# Layers are recursively composable
class MLPBlock(keras.layers.Layer):  # Çok katmanlı bir yapay sinir ağı bloğu
    def __init__(self):  # Başlatıcı metod
        super(MLPBlock, self).__init__()  # Üst sınıfın (keras.layers.Layer) __init__ metodunu çağırıyoruz.
        self.linear_1 = Linear1(32, 64)  # 64 girişli ve 32 nöronlu Linear1 katmanı
        self.linear_2 = Linear2(32, 32)  # 32 girişli ve 32 nöronlu Linear2 katmanı
        self.linear_3 = Linear3(1)  # 1 nöronlu çıkış katmanı (Linear3)

    def call(self, inputs):  # İleri geçiş metodu
        x1 = self.linear_1(inputs)  # İlk katmandan ileri geçiş
        x2 = tf.nn.relu(x1)  # ReLU aktivasyon fonksiyonu
        x3 = self.linear_2(x2)  # İkinci katmandan ileri geçiş
        x4 = tf.nn.relu(x3)  # ReLU aktivasyon fonksiyonu
        return self.linear_3(x4)  # Üçüncü (çıkış) katmanından ileri geçiş


mlp = MLPBlock()  # MLP bloğunu oluşturuyoruz
y = mlp(tf.ones(shape=(3, 64)))  # 3x64 boyutunda bir girdi üzerinden bloğu çalıştırıyoruz
print(mlp.weights)  # MLP bloğunun tüm ağırlıkları yazdırılıyor

# Özet
# Linear1, Linear2, Linear3: Bu katmanlar girişler üzerinde matris çarpımı ve bias eklemesi yapıyor.
#
# Linear1: TensorFlow değişkenleriyle ağırlıkları manuel olarak tanımlar.
# Linear2: add_weight metoduyla ağırlıkları tanımlar.
# Linear3: Giriş boyutunu build sırasında öğrenir ve ağırlıkları o zaman başlatır.
# MLPBlock: Birkaç katmanın bir arada kullanıldığı bir modeldir. Giriş, ardışık olarak farklı katmanlardan geçirilip nihai bir çıkış elde edilir.
#
# Öğrenmen gerekenler:
# Katman subclassing: tf.keras.layers.Layer kullanarak özel katmanlar nasıl oluşturulur.
# Ağırlıkların başlatılması: Ağırlıklar ya add_weight ile ya da TensorFlow değişkenleri kullanılarak tanımlanabilir.
# Katmanların tekrar kullanılabilirliği: Bir katmanı farklı bir yapıda tekrar tekrar kullanabilirsin.
# build ve call metodları: Katmanın girdi boyutunu build metodunda öğrenip, call metodunda ileri geçişi nasıl tanımlayacağını öğrenmelisin.
