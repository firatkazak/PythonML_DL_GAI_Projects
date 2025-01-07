import numpy as np  # Sayısal işlemler ve rastgele veri üretimi için kullanılıyor.
import matplotlib.pyplot as plt  # Grafikler çizmek için kullanılıyor.
import seaborn as sns  # Veri görselleştirme kütüphanesi.
from sklearn.datasets import make_blobs  # Örnek veri kümesi oluşturmak için.
from sklearn.naive_bayes import GaussianNB  # Naive Bayes sınıflandırma algoritması.
from sklearn.datasets import fetch_20newsgroups  # Metin verisi içeren bir veri seti.
from sklearn.feature_extraction.text import TfidfVectorizer  # Metin verilerinden öznitelik çıkarmak için.
from sklearn.naive_bayes import MultinomialNB  # Multinomial Naive Bayes sınıflandırma algoritması.
from sklearn.pipeline import make_pipeline  # Birden fazla adımı birleştirmek için kullanılır (pipeline).
from sklearn.metrics import confusion_matrix  # Karışıklık matrisi hesaplamak için.

# Seaborn kütüphanesini varsayılan ayarlarda çalıştırır.
sns.set()

# Yapay veri kümesi oluşturma
X, y = make_blobs(n_samples=100,  # 100 örnek veri noktası oluşturur.
                  n_features=2,  # 2 öznitelikli (x ve y) veri noktaları oluşturur.
                  centers=2,  # 2 farklı merkezden (cluster) veri oluşturur.
                  cluster_std=1.5  # Her bir merkezin standart sapması.
                  )

# Oluşturulan veri kümesini scatter plot (dağılım grafiği) ile görselleştiriyor.
plt.scatter(X[:, 0],  # X'in ilk sütununu (x ekseni) alır.
            X[:, 1],  # X'in ikinci sütununu (y ekseni) alır.
            c=y,  # Her noktayı sınıfına göre renklendirir.
            s=50,  # Noktaların boyutu (size).
            cmap='RdBu'  # Renk haritası (Red to Blue).
            )
plt.show()

# Gaussian Naive Bayes modelini oluşturuyor.
model = GaussianNB()
model.fit(X, y)  # Modeli eğitim verisiyle eğitiyor (X: özellikler, y: sınıflar).

# Rastgele yeni veri oluşturmak için bir RandomState nesnesi kullanıyor.
rng = np.random.RandomState(0)

# Yeni veri noktaları oluşturuyor. (-6, -14) ile başlayan ve [14, 18] ile çarpılmış rastgele veriler.
X_yeni = [-6, -14] + [14, 18] * rng.rand(1000, 2)

# Yeni veri noktalarının sınıflarını tahmin ediyor.
y_yeni = model.predict(X_yeni)

# İlk veri kümesini tekrar görselleştiriyor.
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')

# Şu anki eksen limitlerini kaydeder.
lim = plt.axis()

# Yeni veri kümesini de görselleştiriyor (yarı saydam olacak şekilde).
plt.scatter(X_yeni[:, 0],  # Yeni verinin ilk özelliği (x ekseni).
            X_yeni[:, 1],  # Yeni verinin ikinci özelliği (y ekseni).
            c=y_yeni,  # Yeni verilerin tahmin edilen sınıflarına göre renklendirme.
            s=20,  # Yeni veri noktalarının boyutu daha küçük.
            cmap='RdBu',  # Aynı renk haritası kullanılır.
            alpha=0.2  # Şeffaflık derecesi (0 ile 1 arası).
            )

# Kaydedilen eksen limitlerini geri yükler (ilk veri seti ile aynı görünümü sağlar).
plt.axis(lim)
plt.show()

# Çoklu sınıflandırma Naive Bayes örneği (metin sınıflandırma)
data = fetch_20newsgroups()  # Yeni haber grupları veri setini indirir.

# Kullanılacak kategorileri belirler.
kategoriler = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']

# Eğitim veri setini bu kategorilere göre indirir.
train = fetch_20newsgroups(subset='train',  # 'train' verileri alır.
                           categories=kategoriler  # Yukarıdaki belirlenen kategoriler.
                           )

# Test veri setini bu kategorilere göre indirir.
test = fetch_20newsgroups(subset='test',  # 'test' verileri alır.
                          categories=kategoriler
                          )

# Eğitim veri setinden bir örneği yazdırır.
print(train.data[5])

# Bir pipeline oluşturur: TfidfVectorizer ile öznitelikler çıkarılır, ardından MultinomialNB modeli uygulanır.
model = make_pipeline(TfidfVectorizer(),  # Metni sayısal bir forma çevirir (TF-IDF ile).
                      MultinomialNB()  # Multinomial Naive Bayes modeli.
                      )

# Modeli eğitim verileriyle eğitir.
model.fit(train.data, train.target)

# Test verilerini kullanarak sınıflandırma tahmini yapar.
etiketler = model.predict(test.data)

# Tahmin edilen etiketler ile gerçek etiketler arasındaki karışıklık matrisini hesaplar.
mat = confusion_matrix(test.target, etiketler)

# Karışıklık matrisini bir ısı haritası ile görselleştirir.
sns.heatmap(mat.T,  # Transpoze edilmiş karışıklık matrisi (satırlar ve sütunlar ters çevrilir).
            square=True,  # Hücrelerin kare şeklinde olmasını sağlar.
            annot=True,  # Hücrelerdeki sayısal değerleri gösterir.
            fmt='d',  # Hücre içindeki sayıların tam sayı formatında olmasını sağlar.
            cbar=False,  # Sağ tarafta renk barı göstermez.
            xticklabels=train.target_names,  # X eksenine kategorileri yerleştirir.
            yticklabels=train.target_names  # Y eksenine kategorileri yerleştirir.
            )

# Grafik etiketleri.
plt.xlabel('Gerçek Değerler')  # X ekseni etiketi.
plt.ylabel('Tahmin Etiketleri')  # Y ekseni etiketi.
plt.show()


# Bir metni kategorilere göre tahmin eden fonksiyon
def predict_category(s, train=train, model=model):
    # Modeli kullanarak verilen metni tahmin eder.
    pred = model.predict([s])
    return train.target_names[pred[0]]  # Tahmin edilen kategoriyi döner.


# Bir metin için kategori tahminini yazdırır.
sonuc = predict_category('determining the screen resolution')
print(sonuc)

# n_samples: Kaç adet veri noktası oluşturulacağını belirler.
# n_features: Her veri noktasının kaç özelliğe sahip olacağını belirtir.
# centers: Kaç tane merkez (cluster) olacağını belirtir.
# cluster_std: Her bir merkezdeki verilerin standart sapmasını belirler.
# c: Noktaların renklerini sınıflarına göre belirler.
# s: Noktaların boyutunu ayarlar.
# cmap: Renk haritası ayarı (hangi renklerin kullanılacağını belirler).
# alpha: Şeffaflık ayarı (0 tamamen şeffaf, 1 tamamen opak).
# square: Hücrelerin kare şeklinde olmasını sağlar.
# annot: Hücrelerin içindeki sayısal değerlerin gösterilmesini sağlar.
# fmt: Hücre içindeki sayısal değerlerin formatını belirler.
# cbar: Renk barının gösterilip gösterilmeyeceğini ayarlar.
# xticklabels, yticklabels: X ve Y eksenlerine etiket koyar.
