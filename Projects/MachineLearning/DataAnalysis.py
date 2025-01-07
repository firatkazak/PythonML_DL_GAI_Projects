import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Iris çiçeğinin ölçümşerini bildiğimiz için etiketler var, haliyle bu bir denetimli öğrenme.
# Iris çiçeğinin 3 türünden birisini tahmin edeceği için bu bir Sınıflandırma örneği.

# Iris veri setini yükleme
iris = load_iris()

# Ekran Çıktıları
print(iris.keys())  # Veri setindeki Key'leri gösterir.
print(iris)  # Veri setini verir.
print(iris['DESCR'])  # Veri setinin özetini verir.
print(iris['target_names'])  # Tahmin edilecek çiçeğin türlerini gösterir.
print(iris['feature_names'])  # Niteliklerin isimlerini verir.
print(type(iris['data']))  # Verinin tipini verir.
print(iris['data'].shape)  # Verinin yapısını verir. (150, 4) 150 farklı çiçek olduğunu ve bu çiçeklerin 4 özelliği olduğunu söyler.
print(iris['data'][:5])  # Veri setindeki ilk 5 örneklemin nitelik değerlerini gösterir.
print(iris['target'])  # 3 tür var. 0 setosa, 1 versicolor 2 virginica olacak şekilde 150 tane veri verir.

# Eğitim ve test setlerine ayırma
X_egitim, X_test, y_egitim, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
# iris['data']: Girdi (features) verileri. Bu veri, modelin öğrenmesi için kullanılacak.
# iris['target']: Çıktı (target) verileri. Bu veri, modelin tahmin etmesini istediğimiz hedef değerleri içeriyor.
# random_state=0: Verilerin rastgele ayrılmasını kontrol eden bir değerdir. Aynı sonuçları tekrar alabilmek için bu sabit bir sayı olarak verilmiştir.

print(X_egitim.shape)  # X_egitim verisinin yapısı verir. (112, 4) 2 boyutlu dizi.
print(y_egitim.shape)  # y_egitim verisinin yapısını verir. (112,) 1 boyutlu dizi.
print(X_test.shape)  # X_test verisinin yapısı verir. (38, 4) 2 boyutlu dizi.
print(y_test.shape)  # y_test verisinin yapısı verir. (38,)   1 boyutlu dizi.

# Eğitim verilerini DataFrame'e dönüştürme
iris_df = pd.DataFrame(data=X_egitim, columns=iris.feature_names)
iris_df['target'] = y_egitim  # Hedef değişkeni de ekleyelim

# Scatter matrix görselleştirme
scatter_matrix(frame=iris_df,  # Gözlem kümesi, scatter plot oluşturulacak veri setidir.
               c=iris_df['target'],  # Verileri renklendirmek için kullanılacak sütunu belirtir. iris_df['target'], farklı kategoriler için farklı renkler sağlar.
               figsize=(15, 15),  # Grafik boyutlarını ayarlar.
               marker='o',  # Verilerin grafikte hangi şekille gösterileceğini belirler.
               hist_kwds={'bins': 20},  # Histogram için ek parametrelerdir.
               s=80,  # Nokta boyutlarını belirler.
               alpha=0.8,  # Saydamlık seviyesini ayarlar.
               cmap=plt.get_cmap('viridis'))  # Renk haritası seçimini sağlar.

# Grafiği gösterme
plt.show()

# Model Oluşturma
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X=X_egitim, y=y_egitim)
X_yeni = np.array([[5, 2.9, 1, 0.2]])  # sepal uzunluğu 5, sepal genişliği 2.9, petal uzunluğu 1, petal genişliği 0.2 iris çiçeği
print(X_yeni.shape)  # 1 tane veri bu 4 özelliği sağlıyor. Onu da aşağıda bulduk, 0 index'e sahip setosa çiçeğiymiş.

tahmin = knn.predict(X_yeni)
print('Tahmin sınıfı: ', tahmin)  # [0]
print('Tahmin türü: ', iris['target_names'][tahmin])  # ['setosa']

y_tahmin = knn.predict(X_test)  # X_test verilerine göre tahminler;
print(y_tahmin)  # [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]

ortalama = np.mean(y_tahmin == y_test)  # Tahmin edilen türün, gerçek tür ile ne kadar eşit olduğunun ortalamasını alıyoruz.
print(ortalama)  # 0.9736842105263158 Bu değere göre %97 doğru tahmin ediyor.

skor = knn.score(X_test, y_test)  # Yukarıdaki tahmini score metodu ile de yapabiliyoruz.
print(skor)  # 0.9736842105263158 Sonuç yine aynı çıktı.

# Sonuç: Artık botanikçi bulduğu Iris çiçeğinin türünü %97 oranında doğru tahmin edebilecektir.
