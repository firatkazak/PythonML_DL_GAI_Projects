import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Buradaki "6" ifadesini işlemcinin çekirdek sayısına göre değiştir.

# 1 komşuluk kullanarak KNN sınıflandırma grafiği çiziliyor
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

# 3 komşuluk kullanarak KNN sınıflandırma grafiği çiziliyor
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

# mglearn.datasets.make_forge() veri setini X ve y olarak alıyoruz.
X, y = mglearn.datasets.make_forge()

# Veriyi eğitim ve test setlerine bölüyoruz, random_state=0 ile sonuçlar sabitleniyor
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, random_state=0)

# 3 komşuluk kullanarak KNN sınıflandırıcı oluşturuluyor
snf = KNeighborsClassifier(n_neighbors=3)

# Eğitim verileriyle KNN sınıflandırıcı modeli eğitiliyor
snf.fit(X_egitim, y_egitim)

# Test verileri ile tahmin yapılıyor
print(snf.predict(X_test))  # [1 0 1 0 1 0 0]

# Modelin test setindeki doğruluğu hesaplanıyor
print(snf.score(X_test, y_test))  # 0.8571428571428571

# Göğüs kanseri veri seti yükleniyor
kanser = load_breast_cancer()

# Veri setinin anahtarları (özellikler ve hedef değişken bilgisi) ekrana yazdırılıyor
print(kanser.keys())  # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# Kanser veri setini eğitim ve test setlerine bölüyoruz.
# stratify=kanser.target ile veri hedef sınıfına göre dengeli bir şekilde bölünüyor
# random_state=66 ile sonuçlar sabitleniyor
X_egitim, X_test, y_egitim, y_test = train_test_split(
    kanser.data,  # Özellikler (bağımsız değişkenler)
    kanser.target,  # Hedef değişken (bağımlı değişken)
    stratify=kanser.target,  # Hedef değişkene göre dengeli bir bölünme sağlanıyor
    random_state=66  # Sabitlenmiş bir rastgelelik sağlanıyor
)

# Eğitim ve test doğruluğunu saklayacak listeler oluşturuluyor
egitim_dogruluk = []
test_dogruluk = []

# 1'den 10'a kadar olan komşuluk sayıları için döngü başlatılıyor
komsuluk_sayisi = range(1, 11)

# Her komşuluk sayısı için KNN modeli eğitiliyor ve doğruluk değerleri hesaplanıyor
for n_komsuluk in komsuluk_sayisi:
    # KNN sınıflandırıcı, belirlenen komşu sayısıyla oluşturuluyor
    snf = KNeighborsClassifier(n_neighbors=n_komsuluk)
    # Model eğitim verisi ile eğitiliyor
    snf.fit(X_egitim, y_egitim)
    # Eğitim setindeki doğruluk hesaplanıp listeye ekleniyor
    egitim_dogruluk.append(snf.score(X_egitim, y_egitim))
    # Test setindeki doğruluk hesaplanıp listeye ekleniyor
    test_dogruluk.append(snf.score(X_test, y_test))

# Komşuluk sayısına göre eğitim ve test doğrulukları grafik üzerinde gösteriliyor
plt.plot(komsuluk_sayisi, egitim_dogruluk, label='Egitim Dogruluk')
plt.plot(komsuluk_sayisi, test_dogruluk, label='Test Dogruluk')
plt.ylabel('Dogruluk')  # Y ekseni doğruluk
plt.xlabel('n-komsuluk')  # X ekseni komşuluk sayısı
plt.legend()
plt.show()

# 1 komşuluk kullanarak KNN regresyon grafiği çiziliyor
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

# 3 komşuluk kullanarak KNN regresyon grafiği çiziliyor
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

# mglearn.datasets.make_wave() veri seti oluşturuluyor (40 örnek)
X, y = mglearn.datasets.make_wave(n_samples=40)

# Veri eğitim ve test setlerine bölünüyor, random_state=0 ile rastgele bölünme sabitleniyor
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, random_state=0)

# 3 komşuluk kullanarak KNN regresör oluşturuluyor
reg = KNeighborsRegressor(n_neighbors=3)

# Eğitim verisi ile KNN regresör modeli eğitiliyor
reg.fit(X_egitim, y_egitim)

# Modelin test setindeki performansı R-kare metriği ile hesaplanıyor
sonuc = reg.score(X_test, y_test)

# Performans sonucu yazdırılıyor
print(sonuc)  # 0.8344172446249605
