from sklearn.datasets import make_moons  # Ay şeklinde veri kümeleri üretmek için kullanılır.
from sklearn.datasets import load_sample_image  # Örnek bir resim yüklemek için kullanılır (örneğin, bir Çin resmi).
from sklearn.datasets._samples_generator import make_blobs  # Veri noktalarını rastgele gruplar (blob'lar) halinde oluşturmak için kullanılır.
from sklearn.cluster import KMeans  # K-means kümeleme algoritması için kullanılır.
from sklearn.cluster import MiniBatchKMeans  # Daha büyük veri setleri için MiniBatchKMeans kullanılır, daha hızlı bir k-means versiyonu.
from sklearn.cluster import SpectralClustering  # Spektral kümeleme algoritmasını uygular.
import mglearn  # mglearn kütüphanesi, sklearn ile birlikte görselleştirme araçları sağlar.
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib kullanılır.
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

# Clustering: K-means algoritmasının nasıl çalıştığını gösterir.
mglearn.plots.plot_kmeans_algorithm()  # mglearn içindeki KMeans algoritma açıklamasını görselleştirir.
plt.show()  # Grafiği gösterir.

mglearn.plots.plot_kmeans_boundaries()  # K-means algoritmasının sınırlarını gösteren bir grafik çizer.
plt.show()  # Grafiği gösterir.

# make_blobs ile rastgele 300 veri noktası ve 4 merkez (cluster) oluşturuluyor.
X, y_gercek = make_blobs(n_samples=300,  # 300 örnek veri noktası.
                         centers=4,  # 4 adet küme merkezi (cluster center).
                         cluster_std=0.60,  # Küme içi dağılımın standart sapması (daha geniş kümeler için arttırılır).
                         random_state=0  # Sonuçları yeniden üretilebilir hale getirmek için rastgelelik kontrolü sağlar.
                         )

# Oluşturulan veri noktalarını 2D düzlemde dağılımlarını görmek için scatter plot ile çizdiriyoruz.
plt.scatter(X[:, 0],  # X'in ilk sütununu (x ekseni) alır.
            X[:, 1],  # X'in ikinci sütununu (y ekseni) alır.
            s=50  # Veri noktalarının boyutu.
            )
plt.show()  # Grafiği gösterir.

# k-Means Clustering (K-Means Kümeleme)
kmeans = KMeans(n_clusters=4)  # K-means algoritması ile 4 küme merkezine sahip olacak şekilde oluşturulur.
kmeans.fit(X)  # K-means algoritması veriye uygulanır ve kümeler hesaplanır.
y_kmeans = kmeans.predict(X)  # Her bir veri noktasının hangi kümeye ait olduğu tahmin edilir.

# Veri noktalarını kümelerine göre renklerle boyar.
plt.scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans, cmap='viridis')  # Veri noktalarını renkli olarak çizer.
centers = kmeans.cluster_centers_  # K-means algoritmasının hesapladığı küme merkezlerini alır.
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', alpha=0.5)  # Küme merkezlerini siyah renkte ve büyük boyutlu olarak çizer.
plt.show()  # Grafiği gösterir.

# make_moons: İki yarım ay şeklinde veri seti oluşturur.
X, y = make_moons(n_samples=200,  # 200 örnek veri noktası.
                  noise=.05,  # Verilere eklenen rastgele gürültü miktarı.
                  random_state=0  # Sonuçları yeniden üretilebilir hale getirir.
                  )

# KMeans ile 2 kümeye ayırır.
labels = KMeans(n_clusters=2, random_state=0).fit_predict(X)  # KMeans algoritmasıyla veriyi 2 kümeye ayırır ve etiketler.
plt.scatter(X[:, 0], X[:, 1], s=50, c=labels, cmap='viridis')  # Küme etiketlerine göre renklerle boyanmış veri noktalarını çizer.
plt.show()  # Grafiği gösterir.

# Spectral Clustering (Spektral Kümeleme)
model = SpectralClustering(n_clusters=2,  # 2 küme oluşturulacak.
                           affinity='rbf',  # Affinity parametresi, veri noktaları arasındaki yakınlık ölçüm yöntemidir. 'rbf' (Radial Basis Function) kullanır.
                           assign_labels='kmeans'  # Etiket atamak için k-means algoritmasını kullanır.
                           )
labels = model.fit_predict(X)  # Modeli veriye uygular ve kümeleri tahmin eder.
plt.scatter(X[:, 0], X[:, 1], s=50, c=labels, cmap='viridis')  # Tahmin edilen kümelere göre veri noktalarını renklerle boyar.
plt.show()  # Grafiği gösterir.

# MiniBatch k-Means (Daha hızlı bir K-means algoritması)
china = load_sample_image('china.jpg')  # Örnek bir Çin resmi yüklenir.
ax = plt.axes(xticks=[], yticks=[])  # Grafik eksenlerindeki tık işaretleri kaldırılır.
ax.imshow(china)  # Çin resmini gösterir.
print(china.shape)  # Resmin boyutlarını yazdırır.
plt.show()  # Resmi gösterir.

# Veri seti hazırlığı
veri = china / 255  # Resmi 0-1 aralığına normalleştirir.
veri = veri.reshape(427 * 640, 3)  # Resmi 2D bir formata indirger (piksel renk değerleri).
print(veri.shape)  # Yeni veri setinin boyutlarını yazdırır.

# MiniBatch K-Means ile 16 küme oluşturulur (16 renk kullanılır).
kmeans = MiniBatchKMeans(16)  # MiniBatchKMeans algoritması, 16 kümeyle uygulanır.
kmeans.fit(veri)  # Model veriye uygulanır.
china = load_sample_image('china.jpg')  # Çin resmi tekrar yüklenir.

ax = plt.axes(xticks=[], yticks=[])  # Eksen işaretleri olmadan yeni bir grafik oluşturulur.
ax.imshow(china)  # Resmi tekrar gösterir.
print(china.shape)  # Resmin boyutlarını tekrar yazdırır.
plt.show()  # Resmi gösterir.

# KMeans(n_clusters=4): K-means algoritması kullanarak veriyi 4 kümeye ayırır.
# MiniBatchKMeans(n_clusters=16): Daha büyük veri setleri için daha hızlı çalışan MiniBatchKMeans algoritması kullanılır. 16 kümeye ayırmak için kullanılır.
# SpectralClustering(n_clusters=2, affinity='rbf', assign_labels='kmeans'): Spektral kümeleme algoritmasını kullanarak veriyi 2 kümeye ayırır.
# Affinity olarak RBF çekirdek fonksiyonu (radial basis function) kullanılır.
# load_sample_image('china.jpg'): Örnek bir resim veri seti olarak Çin resmi yüklenir ve görselleştirilir.
