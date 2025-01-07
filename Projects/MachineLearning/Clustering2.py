import matplotlib.pyplot as plt  # Grafik çizimi ve görselleştirme için matplotlib kullanılır.
import mglearn  # Scikit-learn ile birlikte kullanılan görselleştirme araçları içerir.
from sklearn.cluster import AgglomerativeClustering  # Agglomerative Clustering (Hiyerarşik Kümeleme) algoritması için kullanılır.
from sklearn.datasets import make_blobs  # Rastgele veri setleri oluşturmak için kullanılır.
from sklearn.datasets import make_moons  # Ay şeklinde veri kümeleri oluşturur.
from scipy.cluster.hierarchy import dendrogram, ward  # Hiyerarşik kümeleme için kullanılan yardımcı fonksiyonlar.
from sklearn.cluster import DBSCAN  # DBSCAN kümeleme algoritması için kullanılır.
from sklearn.preprocessing import StandardScaler  # Veriyi ölçeklendirmek (normalleştirmek) için kullanılır.

# Agglomerative Clustering (Hiyerarşik Kümeleme) algoritmasını görselleştirme
mglearn.plots.plot_agglomerative_algorithm()  # Agglomerative clustering algoritmasını görselleştiren bir grafik çizer.
plt.show()  # Görseli ekranda gösterir.

# Rastgele bir veri seti oluşturuyoruz.
X, y = make_blobs(random_state=42)  # 42 numaralı random_state ile yeniden üretilebilir bir rastgele veri seti oluşturur.

# Agglomerative Clustering: Veriyi 3 küme oluşturacak şekilde bölüyor.
agg = AgglomerativeClustering(n_clusters=3)  # n_clusters=3 parametresi, veriyi 3 kümeye ayıracak şekilde hiyerarşik kümeleme yapar.
k = agg.fit_predict(X)  # Veriyi modele uydurur ve her veri noktasının hangi kümeye ait olduğunu tahmin eder.

# Kümelenmiş veri noktalarını görselleştirme.
mglearn.discrete_scatter(X[:, 0], X[:, 1], k)  # X'in iki boyutlu (0 ve 1. sütun) noktalarını kümelere göre renklendirir.
plt.show()  # Grafiği ekranda gösterir.

# Hierarchical Clustering (Hiyerarşik Kümeleme ve Dendrogram)
mglearn.plots.plot_agglomerative()  # Hiyerarşik kümelemeyi gösteren bir örnek grafik çizer.
X, y = make_blobs(random_state=0, n_samples=12)  # 12 örnek ve rastgele kümeler içeren bir veri seti oluşturur.

# ward fonksiyonu, hiyerarşik kümeleme için bağlantı matrisi oluşturur.
linkange_array = ward(X)  # Ward bağlantı yöntemi ile hiyerarşik kümeleme için bağlantı matrisini oluşturur.
dendrogram(linkange_array)  # Dendrogram (ağaç yapısı) kullanarak hiyerarşik kümelemeyi görselleştirir.
plt.show()  # Dendrogram'ı gösterir.

# DBSCAN Clustering (Yoğunluk Tabanlı Kümeleme)
X, y = make_blobs(random_state=0, n_samples=12)  # 12 veri noktası ve rastgele kümeler içeren bir veri seti oluşturur.

# DBSCAN algoritmasını veriye uygular.
dbscan = DBSCAN()  # Yoğunluk tabanlı kümeleme için DBSCAN algoritması.
kumeler = dbscan.fit_predict(X)  # Veriye uygulanır ve her veri noktasının hangi kümeye ait olduğu tahmin edilir.
print(kumeler)  # [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1] çıktı verir, çünkü DBSCAN hiçbir kümeyi bulamaz (hepsi -1 yani "gürültü" olarak sınıflandırılır).

# make_moons ile ay şeklinde bir veri seti oluşturuluyor.
X, y = make_moons(n_samples=200,  # 200 örnek veri noktası.
                  noise=0.06,  # Verilere eklenen rastgele gürültü miktarı.
                  random_state=42  # Sonuçları yeniden üretilebilir hale getirir.
                  )

# Veriyi standartlaştırmak için StandardScaler kullanılıyor.
scaler = StandardScaler()  # Veri setini ölçeklendirmek için kullanılır.
scaler.fit(X)  # Modeli veriye uydurur (ortalama ve standart sapmayı hesaplar).
X_scaled = scaler.transform(X)  # Veriyi ölçeklendirilmiş hale getirir (ortalama 0, standart sapma 1 olacak şekilde normalleştirir).

# DBSCAN algoritması ölçeklendirilmiş verilere uygulanıyor.
dbscan = DBSCAN()  # DBSCAN algoritması.
kumeler = dbscan.fit_predict(X_scaled)  # Ölçeklendirilmiş veriye DBSCAN algoritmasını uygular ve kümeleri tahmin eder.

# Kümeleri görselleştirme.
plt.scatter(X_scaled[:, 0],  # X ekseni için ölçeklendirilmiş ilk özellik.
            X_scaled[:, 1],  # Y ekseni için ölçeklendirilmiş ikinci özellik.
            c=kumeler,  # Her noktayı tahmin edilen kümelere göre renklendir.
            cmap=mglearn.cm2,  # mglearn'in özel renk haritası.
            s=60  # Noktaların boyutu.
            )
plt.show()  # Grafiği ekranda gösterir.
# AgglomerativeClustering(n_clusters=3): Hiyerarşik kümeleme algoritmasıdır ve veriyi 3 kümeye ayırır.
# ward(X): Hiyerarşik kümeleme için Ward yöntemi kullanılarak bağlantı matrisi oluşturur. Bu, veri noktalarının birleştirilme sürecini belirler.
# DBSCAN(): Yoğunluk tabanlı bir kümeleme algoritmasıdır ve belirli bir yoğunluk eşiğini aşan kümeler oluşturur. Gürültü olan noktalar -1 ile işaretlenir.
# StandardScaler(): Veriyi ortalaması 0 ve standart sapması 1 olacak şekilde ölçeklendirir. Bu, DBSCAN gibi mesafeye dayalı algoritmaların performansını artırır.
