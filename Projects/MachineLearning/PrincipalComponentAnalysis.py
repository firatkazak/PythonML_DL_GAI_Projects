import numpy as np  # NumPy kütüphanesini içe aktarır (dizi işlemleri için).
import matplotlib.pyplot as plt  # Matplotlib kütüphanesinden grafik çizme fonksiyonlarını içe aktarır.
from sklearn.decomposition import PCA  # Sklearn kütüphanesinden PCA (Principal Component Analysis) sınıfını içe aktarır.
from sklearn.datasets import load_digits  # Sklearn kütüphanesinden dijit veri setini yüklemek için fonksiyonu içe aktarır.

# PCA (Principal Component Analysis) uygulaması
rng = np.random.RandomState(1)  # Rastgele sayı üreteci oluşturur ve sabit bir tohum değeri kullanır.
X = np.dot(rng.rand(2, 2), rng.rand(2, 200)).T  # Rastgele veriler oluşturur ve bunları matris çarpımı ile üretir. (200 örnek, 2 özellik)
print(X.shape)  # Verinin şekli: (200, 2) (200 örnek, 2 özellik)

plt.scatter(X[:, 0], X[:, 1])  # Verinin ilk ve ikinci özelliklerini scatter plot ile görselleştirir.
plt.show()  # Grafiği gösterir.

pca = PCA(n_components=2)  # PCA nesnesi oluşturur ve 2 bileşeni koruyacak şekilde yapılandırır.
pca.fit(X)  # PCA modelini verilerle eğitir.
print(pca.components_)  # PCA bileşenlerini (yeni eksenler) ekrana yazdırır.
print(pca.explained_variance_)  # Her bir bileşenin açıkladığı varyansı ekrana yazdırır.

# Dimensionality Reduction (Boyut İndirgeme)
pca = PCA(n_components=1)  # PCA nesnesini 1 bileşen ile yeniden oluşturur.
pca.fit(X)  # PCA modelini verilerle eğitir.
X_pca = pca.transform(X)  # Veriyi 1 bileşene indirger.
print(X.shape)  # Orijinal verinin şekli: (200, 2)
print(X_pca.shape)  # Düşük boyutlu verinin şekli: (200, 1) (artık 1 özellik var)
X_yeni = pca.inverse_transform(X_pca)  # Düşük boyutlu veriyi orijinal boyutlarına geri dönüştürür.
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)  # Orijinal veriyi scatter plot ile gösterir.
plt.scatter(X_yeni[:, 0], X_yeni[:, 1], alpha=0.8)  # Yeniden dönüştürülmüş veriyi scatter plot ile gösterir.
plt.show()  # Grafiği gösterir.

# PCA for Visualization (Görselleştirme için PCA)
digits = load_digits()  # Dijit veri setini yükler.
print(digits.data.shape)  # Dijit veri setinin şekli: (1797, 64) (1797 örnek, 64 özellik)

pca = PCA(2)  # PCA nesnesini 2 bileşen ile oluşturur.
data_pca = pca.fit_transform(digits.data)  # Veri setini 2 bileşene indirger ve dönüştürür.
print(data_pca.shape)  # Düşük boyutlu verinin şekli: (1797, 2) (1797 örnek, 2 özellik)

# Choosing the number of Components (Bileşen Sayısını Seçme)
pca = PCA().fit(digits.data)  # PCA modelini veri seti ile eğitir ve tüm bileşenleri korur.
plt.plot(np.cumsum(pca.explained_variance_ratio_))  # Açıklanan varyans oranlarının kümülatif toplamını grafikle gösterir.
plt.show()  # Grafiği gösterir.

# Noise Filtering (Gürültü Filtreleme)
# Gürültü ekleme
rng = np.random.RandomState(42)  # Gürültü için rastgele sayı üreteci oluşturur ve sabit bir tohum değeri kullanır.
X_noisy = digits.data + 2.5 * rng.normal(size=digits.data.shape)  # Verilere 2.5 standart sapma ile rastgele gürültü ekler.

# Gürültülü veriyi PCA ile düşük boyutlara indirgeme
pca = PCA(0.95)  # PCA nesnesini, veri varyansının %95'ini açıklayan bileşenlerle oluşturur.
X_pca = pca.fit_transform(X_noisy)  # Gürültülü veriyi PCA ile dönüştürür.

# Gürültüsüz hale getirilmiş veriyi geri dönüştürme
X_filtered = pca.inverse_transform(X_pca)  # Gürültülü veriyi PCA ile düşük boyutlu haline getirdikten sonra orijinal boyutlarına geri döndürür.

# Orijinal, Gürültülü ve Gürültüden Arındırılmış Veriyi Görselleştirme
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))  # 1 satır, 3 sütundan oluşan bir figür oluşturur.
ax[0].imshow(digits.data[0].reshape(8, 8), cmap='gray')  # Orijinal veriyi gri tonlarında görüntüler.
ax[0].set_title("Orijinal Veri")  # İlk grafiğin başlığını ayarlar.
ax[1].imshow(X_noisy[0].reshape(8, 8), cmap='gray')  # Gürültülü veriyi gri tonlarında görüntüler.
ax[1].set_title("Gürültülü Veri")  # İkinci grafiğin başlığını ayarlar.
ax[2].imshow(X_filtered[0].reshape(8, 8), cmap='gray')  # Gürültüden arındırılmış veriyi gri tonlarında görüntüler.
ax[2].set_title("Gürültüden Arındırılmış Veri")  # Üçüncü grafiğin başlığını ayarlar.
plt.show()  # Grafikleri gösterir.
