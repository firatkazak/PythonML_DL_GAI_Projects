from sklearn.datasets import load_digits  # El yazısı rakamlar veri setini yüklemek için kullanılır.
from sklearn.decomposition import PCA  # Ana bileşen analizi (PCA) için sklearn'den PCA sınıfını içe aktarır.
from sklearn.manifold import TSNE  # t-SNE (t-Distributed Stochastic Neighbor Embedding) algoritmasını içe aktarır.
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib'i içe aktarır.
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

# Manifold Learning
digits = load_digits()  # El yazısı rakamlar veri setini yükler. 'digits.data' özellikleri, 'digits.target' hedef sınıfları içerir.
pca = PCA(n_components=2).fit(digits.data)  # PCA modelini oluşturur. Veriyi iki boyuta indirir (n_components=2).
digits_pca = pca.transform(digits.data)  # PCA modelini kullanarak veri setini iki boyuta indirger.

plt.figure(figsize=(10, 10))  # Grafiğin boyutunu 10x10 piksel olarak ayarlar.
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())  # Grafik için x ekseni sınırlarını ayarlar.
plt.ylim(digits_pca[:, 0].min(), digits_pca[:, 0].max())  # Grafik için y ekseni sınırlarını ayarlar.
renkler = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]  # Her bir rakam sınıfı için renkler belirler.

# Her bir rakamı PCA ile elde edilen iki boyutlu düzlemde görselleştirir.
for i in range(len(digits.data)):  # Veri setindeki her bir örnek için döngüye girer.
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),  # Her bir örneğin pozisyonuna hedef rakamı (digits.target[i]) yerleştirir.
             color=renkler[digits.target[i]],  # Rakam sınıfına göre renk belirler.
             fontdict={'weight': 'bold', 'size': 9})  # Yazı stilini kalın ve 9 boyutunda yapar.
plt.show()  # Grafiği gösterir.

# t-SNE ile boyut indirgeme ve görselleştirme
tsne = TSNE(random_state=42)  # t-SNE algoritmasını oluşturur. random_state=42 ile rastgeleliği kontrol eder.
digits_tsne = tsne.fit_transform(digits.data)  # Veriyi t-SNE kullanarak iki boyuta indirger.

plt.figure(figsize=(10, 10))  # Grafiğin boyutunu 10x10 piksel olarak ayarlar.
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())  # Grafik için x ekseni sınırlarını ayarlar.
plt.ylim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())  # Grafik için y ekseni sınırlarını ayarlar.
renkler = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]  # Aynı renk listesi.

# t-SNE ile indirgenmiş veriyi görselleştirir.
for i in range(len(digits.data)):  # Veri setindeki her bir örnek için döngüye girer.
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),  # t-SNE ile elde edilen x, y pozisyonlarına hedef rakamı ekler.
             color=renkler[digits.target[i]],  # Her bir hedef rakam için uygun renk atar.
             fontdict={'weight': 'bold', 'size': 9})  # Yazı stilini kalın ve 9 boyutunda yapar.
plt.show()  # Grafiği gösterir.

# Manifold Öğrenme Teknikleri: LLE, LTSA, Hessian LLE, Modified LLE, Isomap, MDS, SpectralEmbedding, t-SNE BUNLARA DA BAK.
# Bu teknikler manifold öğrenme algoritmalarıdır. Yüksek boyutlu verileri daha az boyutlu uzaylara indirgemek için kullanılırlar.
# PCA(n_components=2): PCA ile veriyi iki boyuta indirger. n_components=2, veriyi iki ana bileşene indirgeyeceğimiz anlamına gelir.
# tsne = TSNE(random_state=42): t-SNE algoritmasıyla boyut indirgeme yapılır. random_state=42 rastgeleliği kontrol eder, böylece sonuçlar tekrarlanabilir.
# plt.text(): Grafik üzerinde belirli bir noktaya metin (rakam) ekler. Burada, indirgenmiş verilerin koordinatları kullanılarak her bir veri örneğinin sınıfı gösterilir.
# Bu iki yöntem (PCA ve t-SNE) boyut indirgemede kullanılır, ancak farklı avantaj ve dezavantajları vardır.
# PCA daha basit ve hızlıdır, t-SNE ise daha karmaşık ve yüksek boyutlu verilerde daha iyi ayrım sağlar.
