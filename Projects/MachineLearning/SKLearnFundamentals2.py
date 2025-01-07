import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Veriyi yükle
iris = sns.load_dataset('iris')
X_iris = iris.drop(labels='species', axis=1)  # species hariç tüm sütunları alarak özellik matrisini oluşturur.
# axis=1 parametresi, sütunları (yani veri setinin genişliğini) ifade eder.
# iris.drop(labels='species', axis=1) kodu, species sütununu veri setinden çıkartarak sadece özellik sütunlarını içeren bir veri çerçevesi oluşturur.
# Eğer axis=0 verilseydi, bu durumda satırları (yani veri setinin yüksekliğini) ifade ederdi ve belirli bir satırı çıkartmaya çalışırdı.
# Ancak burada amacımız sütunları çıkarmak olduğu için axis=1 kullanılmıştır.
y_iris = iris['species']  # species sütununu hedef değişken (yani hangi tür olduğu) olarak ayırıyor.
# Bu sütun, sınıflandırma için kullanılacak olan iris çiçek türlerini (setosa, versicolor, virginica) içerir.

# 1) PCA modelini oluştur ve veriyi dönüştür
model = PCA(n_components=2)  # PCA modelini oluşturuyorsun. n_components=2 demek, veriyi 2 bileşene (2 boyuta) indirgemek istediğin anlamına gelir.
model.fit(X_iris)  # PCA modelini, X_iris adlı veri seti üzerinde eğitiyorsun.
# Bu işlem, verinin temel bileşenlerini bulur ve veriyi bu bileşenler doğrultusunda yeniden ifade etmeye hazırlanır.
X_2D = model.transform(X_iris)  # Eğitilen PCA modelini kullanarak, X_iris verisini iki boyutlu bir uzaya dönüştürüyorsun.
# Sonuç, her veri noktasının yeni iki boyutlu uzayda yer aldığı koordinatları içerir.

# PCA bileşenlerini iris veri çerçevesine ekle
iris['PCA1'] = X_2D[:, 0]  # X_2D matrisi içindeki ilk bileşen değerlerini (PCA1 olarak adlandırılan yeni bir sütuna) iris veri çerçevesine ekliyorsun.
iris['PCA2'] = X_2D[:, 1]  # Aynı şekilde, X_2D matrisi içindeki ikinci bileşen değerlerini (PCA2 olarak adlandırılan) iris veri çerçevesine ekliyorsun.

print(iris.head())  # Veri çerçevesinin ilk birkaç satırını yazdır.

# lmplot: Bu fonksiyon genelde iki değişken arasındaki ilişkiyi gösteren doğrusal bir regresyon çizgisi çizer.
# ANCAK burada fit_reg=False olduğu için regresyon çizgisi olmadan sadece veri noktalarının dağılımını göstereceksin.
sns.lmplot(
    x='PCA1',  # Grafikte x ekseni için PCA1, y ekseni için PCA2 değişkenleri kullanılır. Bunlar, PCA ile elde ettiğin bileşenlerdir.
    y='PCA2',  # Grafikte x ekseni için PCA1, y ekseni için PCA2 değişkenleri kullanılır. Bunlar, PCA ile elde ettiğin bileşenlerdir.
    hue='species',  # Veri noktalarını species (yani iris veri setindeki çiçek türü) değişkenine göre renklendirir.
    data=iris,  # Grafikte kullanılacak veri seti olarak iris veri çerçevesini belirtirsin. Bu veri seti, PCA bileşenlerini ve çiçek türlerini içerir.
    fit_reg=False)  # Regresyon çizgisi çizilmemesini sağlar. Sadece veri noktaları gösterilecektir.
plt.show()  # Grafiği ekranda gösterir.

# 2) Gaussian Mixture Model kullanarak iris veri setini 3 kümeye ayırıyoruz.
model = GaussianMixture(
    n_components=3,  # Gaussian Mixture Model oluşturuluyor. n_components=3 parametresi, veriyi 3 kümeye ayırmak istediğini belirtir.
    covariance_type='full'  # Kümelerin tam kovaryans matrislerini kullanacağını belirtir, yani her kümenin farklı şekillerde eliptik olmasına izin verir.
)
model.fit(X_iris)  # GMM modelini X_iris adlı veri setine uygular ve modelin kümeleri öğrenmesini sağlar.
y_gmm = model.predict(X_iris)  # Modeli kullanarak her veri noktasının hangi kümeye ait olduğunu tahmin eder.
# Sonuçta her bir veri noktası için bir küme etiketi (0, 1, 2 gibi) atanır ve y_gmm değişkenine kaydedilir.
iris['kumeleme'] = y_gmm  # Kümeleme sonuçları (y_gmm), iris veri setine yeni bir sütun olarak eklenir.
# Bu sütun her veri noktasının ait olduğu kümeyi (kumeleme adıyla) içerir.
print(iris.head())  # Veri çerçevesinin ilk birkaç satırını yazdır.
sns.lmplot(
    x='PCA1',  # PCA ile elde edilen iki bileşen, x ve y eksenlerinde kullanılır.
    y='PCA2',  # PCA ile elde edilen iki bileşen, x ve y eksenlerinde kullanılır.
    hue='species',  # Çiçek türlerini (setosa, versicolor, virginica) renklerle ayırır.
    col='kumeleme',  # Kümeleme sütunu ile grafiği farklı sütunlara böler. Her sütun, farklı bir kümenin (0, 1, 2) görselleştirilmesini sağlar.
    data=iris,  # Görselleştirilecek veri seti.
    fit_reg=False  # Regresyon çizgisi olmadan sadece dağılım grafiği oluşturur.
)
plt.show()  # Grafiği ekranda gösterir.

#  3) Sayıların olduğu bölüm;
digits = load_digits()  # Sklearn kütüphanesinden digits veri setini yüklersin. Bu veri seti, 0-9 arasındaki el yazısı rakamların 8x8 piksellik görüntülerini içerir.
print(digits.images.shape)  # (1797, 8, 8) 8x8 pixelden oluşan 1797 tane örnek varmış.

fig, axes = plt.subplots(  # 10x10 boyutunda bir grafik düzeni oluşturursun, yani toplamda 100 küçük grafik (subplot) olacak.
    nrows=10,  # 10 satır ve 10 sütundan oluşan 100 tane alt grafik olacak.
    ncols=10,  # 10 satır ve 10 sütundan oluşan 100 tane alt grafik olacak.
    figsize=(8, 8),  # Grafiklerin boyutunu ayarlar (8x8 inçlik bir figür).
    subplot_kw={'xticks': [], 'yticks': []},  # Her küçük grafikte x ve y eksenlerinde tik işaretleri gösterilmez.
    gridspec_kw=dict(hspace=0.1, wspace=0.1)  # Grafikler arasındaki boşlukları (yükseklik ve genişlik) ayarlarsın. hspace dikey boşluk, wspace yatay boşluktur.
)

for i, ax in enumerate(axes.flat):  # 10x10'luk alt grafikleri tek boyutlu bir liste gibi düzleştirip, her grafikte bir görüntü çizmek için sırayla erişirsin.
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    # Her bir alt grafikte, digits.images[i] (yani i. görüntü) 8x8 piksellik bir resim olarak çizilir.
    # cmap='binary': Siyah-beyaz renk haritası (grayscale) kullanılır.
    # interpolation='nearest': Görüntüyü çizmek için en yakın komşu interpolasyon yöntemi kullanılır.
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
    # Her görüntünün sol üst köşesine, o görüntüdeki rakamı(etiketini) küçük bir yazı olarak ekler.
    # str(digits.target[i]): Görüntüdeki rakamın etiketini (0-9) elde edersin.
    # transform=ax.transAxes: Yazıyı eksen koordinatlarına göre konumlandırırsın.
    # color='green': Yazı rengini yeşil yaparsın.
plt.show()  # Grafiği ekranda gösterir.

# 4)
X = digits.data
# digits veri setindeki her 8x8 piksellik görüntü, 64 özellikli bir vektör olarak temsil edilir. Bu nedenle X, 1797 örneği ve her bir örnek için 64 piksel değeri içerir.
y = digits.target  # digits.target ise her örneğin hangi rakam olduğunu (0-9 arası) içerir.
print(X.shape)  # X'in boyutları yazdırılır. Çıktı (1797, 64) olacaktır, yani 1797 adet örnek ve her biri 64 boyutlu bir veri vektörü.
print(y.shape)  # y'nin boyutları yazdırılır. Çıktı (1797,) olacaktır, yani her örnek için bir rakam etiketi.
iso = Isomap(  # Isomap (Isometric Mapping) algoritmasını kullanarak veriyi 2 boyuta indirgiyoruz.
    n_components=2,  # Veriyi iki boyuta indirgeyeceğini belirtir.
    n_neighbors=10  # Her veri noktası için en yakın 10 komşuyu dikkate alarak veri yapısını çıkarır.
)  # Isomap, komşuluk ilişkilerini koruyarak düşük boyutlu bir temsil oluşturur.
iso.fit(X)  # X (64 boyutlu veri) üzerinde Isomap algoritmasını eğitirsin.
data2 = iso.transform(X)  # Veriyi 2 boyuta indirgersin. data2, her örneğin 2 boyutlu temsillerini içerir.
print(data2.shape)  # data2'nin boyutlarını yazdırır. Çıktı (1797, 2) olacaktır, yani 1797 örnek ve her biri için 2 boyutlu bir veri noktası.

plt.scatter(  # İki boyutlu veri noktalarını bir scatter plot (dağılım grafiği) ile çizer.
    x=data2[:, 0],  # data2 veri setindeki ilk boyut (x ekseni) ve ikinci boyut (y ekseni) kullanılarak noktalar grafikte yerleştirilir.
    y=data2[:, 1],  # data2 veri setindeki ilk boyut (x ekseni) ve ikinci boyut (y ekseni) kullanılarak noktalar grafikte yerleştirilir.
    c=digits.target,  # Renkler, her veri noktasının hangi rakam olduğunu (digits.target) gösterir. Her rakam için farklı renkler kullanılır.
    alpha=0.5,  # Noktaların yarı saydam olmasını sağlar (şeffaflık), böylece üst üste binen noktalar daha net görünür.
    cmap=plt.get_cmap(name='tab10', lut=10)  # 10 farklı renkten oluşan bir renk haritası (colormap) kullanılır, bu da 0-9 arası rakamları farklı renklerle gösterir.
)
plt.colorbar(  # Renk çubuğu eklenir. Bu çubuk, her renk ile hangi rakamın temsil edildiğini gösterir.
    label='digit etiket',  # Renk çubuğuna "digit etiket" başlığı eklenir.
    ticks=range(10)  # 0-9 arasındaki rakamlar için tik işaretleri eklenir.
)
plt.show()  # Grafiği ekranda gösterir.

# 5)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, random_state=0)  # Veriyi eğitim ve test olarak ikiye bölersin.
model = GaussianNB()  # Naive Bayes algoritmasının Gaussian versiyonu kullanılır.
model.fit(X_egitim, y_egitim)  # Modeli, eğitim verisi X_egitim ve etiketler y_egitim ile eğitirsin. Model, her sınıf için özelliklerin dağılımlarını öğrenir.
y_model = model.predict(X_test)  # Test verisi X_test kullanılarak modelin tahmin ettiği değerleri elde edersin. y_model, modelin tahmin ettiği rakamları içerir.
dogrulukOrani = accuracy_score(y_test, y_model)
# Test verisinin gerçek etiketleri y_test ile modelin tahmin ettiği etiketleri y_model karşılaştırarak doğruluk oranını hesaplar.
# Bu oran, modelin doğru tahminlerinin toplam tahminlere oranını verir.
print(dogrulukOrani)  # Modelin doğruluk oranını

mat = confusion_matrix(y_test, y_model)  # Karışıklık matrisi, modelin hangi sınıfları doğru ve hangi sınıfları yanlış tahmin ettiğini gösteren bir tabloyu oluşturur.
# Matrisin satırları gerçek sınıfları, sütunları ise tahmin edilen sınıfları temsil eder.
# Örneğin, (2, 3) hücresindeki değer, modelin 2 yerine 3'ü tahmin ettiği örneklerin sayısını gösterir.
sns.heatmap(
    data=mat,  # # Karışıklık matrisi
    square=True,  # Hücrelerin kare şeklinde olmasını sağlar.
    annot=True,  # Her hücredeki sayısal değeri (örneğin, kaç tane doğru/yanlış tahmin yapıldığı) gösterir.
    cbar=False  # Yan tarafta bir renk barı göstermeyi devre dışı bırakır.
)
plt.xlabel('Tahmin Deger')  # X ekseni (sütunlar), modelin tahmin ettiği değerleri temsil eder.
plt.ylabel('Gercek Deger')  # Y ekseni (satırlar), gerçek sınıfları temsil eder.
plt.show()  # Görselleştirilen karışıklık matrisini ekranda gösterir.
# Burada Unsupervised öğrenme algoritmalarından sınıflama ve kümeleme algoritmalarını öğrendik.
# Resimdeki yakamları fark eden basit bir model oluşturduk.
