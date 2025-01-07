import matplotlib.pyplot as plt  # Grafik çizimi için kullanılır.
from sklearn.mixture import GaussianMixture  # Gaussian Mixture Model (GMM) kümeleme algoritmasını sağlar.
from sklearn.datasets import make_blobs, make_moons, load_digits  # Veri setleri oluşturmak için kullanılır.
import numpy as np  # Sayısal işlemler için kullanılır.
from sklearn.decomposition import PCA  # Temel bileşen analizi (PCA) ile boyut indirgeme için kullanılır.
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

# Gaussian Mixture Models (GMM) uygulaması
X, y_true = make_blobs(n_samples=400,  # 400 veri noktası.
                       centers=4,  # 4 farklı merkezden (cluster) oluşan bir veri seti.
                       cluster_std=0.60,  # Kümelerin standart sapması, yani yayılma oranı.
                       random_state=0  # Rastgele sonuçları aynı tutmak için kullanılır.
                       )

X = X[:, ::-1]  # Veriyi ters çevirir (sütunları tersine çevirir).
gmm = GaussianMixture(n_components=4).fit(X)  # n_components=4: 4 bileşenli bir GMM modeli oluşturur ve veriye uydurur.

labels = gmm.predict(X)  # Veriyi hangi küme bileşenine ait olduğunu tahmin eder.

plt.scatter(X[:, 0],  # X'in 0. sütunu (ilk özellik) X ekseni olarak kullanılır.
            X[:, 1],  # X'in 1. sütunu Y ekseni olarak kullanılır.
            c=labels,  # Her veri noktasını tahmin edilen küme etiketine göre renklendirir.
            s=40,  # Nokta boyutunu ayarlar.
            cmap='viridis'  # Viridis renk haritasını kullanarak renklendirme yapılır.
            )

props = gmm.predict_proba(X)  # Her veri noktasının hangi bileşene ait olduğuna dair olasılıkları döndürür.
print(props[:5].round(3))  # İlk 5 nokta için bu olasılıkları yazdırır (virgülden sonra 3 basamak gösterir).
plt.show()  # Grafiği ekranda gösterir.

# GMM ile yoğunluk tahmini (Density Estimation)
X_moons, y_moons = make_moons(n_samples=200,  # 200 veri noktası.
                              noise=0.05,  # Ay şekilli veriye eklenen rastgele gürültü miktarı.
                              random_state=0  # Aynı rastgele sonuçları elde etmek için kullanılır.
                              )

plt.scatter(X_moons[:, 0], X_moons[:, 1])  # Veriyi 2D scatter plot olarak çizer.
plt.show()

# İki bileşenli bir GMM modeli
gmm2 = GaussianMixture(n_components=2,  # 2 bileşenli bir GMM modeli oluşturur.
                       covariance_type='full',  # 'full': Bileşenler arasındaki tam kovaryans matrisini kullanır.
                       random_state=0  # Rastgele sonuçları sabitlemek için kullanılır.
                       )

# 16 bileşenli bir GMM modeli
gmm16 = GaussianMixture(n_components=16,  # 16 bileşenli bir GMM modeli oluşturur.
                        covariance_type='full',  # 'full' kovaryans tipi.
                        random_state=0  # Rastgele sonuçları sabitler.
                        )

# Optimal bileşen sayısını belirlemek için BIC ve AIC kullanımı
n_components = np.arange(1, 21)  # 1'den 20'ye kadar bileşen sayıları için bir dizi oluşturur.
models = [GaussianMixture(n_components=n,  # Farklı bileşen sayıları için GMM modelleri oluşturur.
                          covariance_type='full',  # Tam kovaryans matrisini kullanır.
                          random_state=0  # Rastgeleliği sabitler.
                          ).fit(X_moons) for n in n_components]  # Her modelin veriye uydurulmasını sağlar.

# AIC ve BIC değerlerini grafikte çizer
plt.plot(n_components, [m.bic(X_moons) for m in models], label='BIC')  # BIC değerlerini çizdirir.
plt.plot(n_components, [m.aic(X_moons) for m in models], label='AIC')  # AIC değerlerini çizdirir.
plt.legend(loc='best')  # En iyi konumda (best) bir gösterge ekler.
plt.xlabel('n_components')  # X ekseni etiketi: Bileşen sayısı.
plt.show()  # Grafiği gösterir.

# Uygulama: El yazısı rakamlar veri seti
digits = load_digits()  # Rakamlar (el yazısı) veri setini yükler.

# PCA ile boyut indirgeme
pca = PCA(n_components=.99,  # Verinin %99 varyansını koruyacak şekilde bileşen sayısını otomatik olarak belirler.
          whiten=True  # 'whiten' parametresi, her bileşeni normalize eder (ortalaması 0, standart sapması 1 olacak şekilde).
          )

# PCA ile indirgenmiş veri seti
data = pca.fit_transform(digits.data)  # PCA modelini veriye uydurur ve boyut indirgeme işlemi gerçekleştirir.

# 110 bileşenli GMM modeli
gmm = GaussianMixture(n_components=110,  # 110 bileşen içeren bir GMM modeli oluşturur.
                      covariance_type='full',  # Tam kovaryans matrisini kullanır.
                      random_state=0  # Rastgeleliği sabitler.
                      ).fit(data)  # Modeli PCA ile indirgenmiş veriye uydurur.

# Yeni veri örnekleri üretme
data_new = gmm.sample(100)  # GMM modelinden 100 yeni veri örneği üretir.
print(data_new)  # Üretilen veri örneklerini yazdırır.

# Farklı bileşen sayıları ile AIC değerlerini karşılaştırma
n_components = np.arange(50, 210, 10)  # 50'den 200'e kadar 10 aralıklarla bileşen sayıları oluşturur.

models = [GaussianMixture(n_components=n,  # Farklı bileşen sayıları için GMM modelleri oluşturur.
                          covariance_type='full',  # Tam kovaryans matrisini kullanır.
                          random_state=0  # Rastgele sonuçları sabitler.
                          ) for n in n_components]

# AIC değerlerini hesaplayıp grafikte çizer
aic = [model.fit(data).aic(data) for model in models]  # Her modelin AIC (Akaike Information Criterion) değerini hesaplar.
plt.plot(n_components, aic)  # AIC değerlerini çizer.
plt.show()  # Grafiği gösterir.
# GaussianMixture(n_components=4): Gaussian Mixture Model'de kullanılan bileşen sayısını belirler. Örneğin, 4 bileşenli bir GMM modeli.
# covariance_type='full': Kovaryans matrisinin tam (full) olarak kullanılacağını belirtir. Diğer seçenekler 'tied', 'diag', 'spherical' olabilir.
# PCA(n_components=.99, whiten=True): n_components=0.99 parametresi, verinin %99 varyansını koruyan minimum bileşen sayısını belirler.
# whiten=True, PCA bileşenlerini normalize eder.
# random_state=0: Rastgelelik durumunu kontrol eder. Sonuçların tekrar üretilebilir olması için kullanılır.
# Bu kodda GMM ile kümeleme, yoğunluk tahmini, ve AIC/BIC kriterleriyle model performansının değerlendirilmesi yapılıyor.
