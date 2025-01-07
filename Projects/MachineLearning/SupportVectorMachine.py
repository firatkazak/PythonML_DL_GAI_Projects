import matplotlib.pyplot as plt  # Grafik çizmek için matplotlib kütüphanesi.
from sklearn.svm import SVC  # Destek vektör makineleri (SVM) sınıflandırıcısı.
from sklearn.datasets import load_breast_cancer  # Göğüs kanseri veri setini yüklemek için.
from sklearn.model_selection import train_test_split  # Eğitim ve test setlerine bölmek için kullanılır.
from sklearn.datasets._samples_generator import make_blobs, make_circles  # Veri noktaları oluşturmak için make_blobs, Daire şeklinde veri noktaları oluşturmak için make_circles fonksiyonunu kullanıyoruz.

# make_blobs fonksiyonu ile veri noktaları oluşturuyoruz.
X, y = make_blobs(n_samples=50,  # n_samples: Oluşturulacak veri örneği sayısı.
                  centers=2,  # centers: Kaç sınıf/merkez olduğunu belirler (bu örnekte 2 sınıf var).
                  random_state=0,  # random_state: Rastgele sayı üreticisinin başlangıç noktası (tekrarlanabilir sonuçlar için).
                  cluster_std=0.60)  # cluster_std: Grupların yayılımı (standart sapma).

# X ve y verilerini kullanarak bir scatter plot (dağılım grafiği) çiziyoruz.
plt.scatter(X[:, 0],  # İlk özelliği (x ekseni) kullanarak dağılım çiziyoruz.
            X[:, 1],  # İkinci özelliği (y ekseni) kullanarak dağılım çiziyoruz.
            c=y,  # Sınıf etiketlerine göre renklendiriyoruz.
            s=50,  # Veri noktalarının boyutu.
            cmap='autumn')  # Renk haritası olarak sonbahar renklerini kullanıyoruz.

plt.show()  # Grafiği görüntülemek için.

# Lineer bir SVM modeli oluşturuyoruz.
model = SVC(kernel='linear',  # kernel: Lineer bir karar sınırı kullanarak verileri ayırmak için.
            C=1E10  # C: Hata toleransı parametresi (büyük C değeri hatayı azaltmaya odaklanır).
            )

model.fit(X, y)  # SVM modelini X ve y verileri ile eğitiyoruz.

# make_circles fonksiyonu ile dairesel veri noktaları oluşturuyoruz.
X, y = make_circles(n_samples=100,  # n_samples: Oluşturulacak veri sayısı.
                    factor=.1,  # factor: Dairenin iç ve dış çapları arasındaki oran.
                    noise=.1)  # noise: Veriye rastgele gürültü ekler.

# RBF kernel ile başka bir SVM modeli oluşturuyoruz.
clf = SVC(kernel='rbf',  # kernel: RBF (radial basis function) çekirdeği.
          C=1E6,  # C: Büyük değer hatayı azaltmaya çalışır.
          gamma='auto'  # gamma: Çekirdek fonksiyonun etkisini kontrol eder, 'auto' ile veri boyutuna göre ayarlanır.
          ).fit(X, y)  # Modeli X ve y verileri ile eğitiyoruz.

# make_blobs fonksiyonu ile farklı bir veri seti oluşturuyoruz.
X, y = make_blobs(n_samples=100,  # 100 örnek oluşturuyoruz.
                  centers=2,  # 2 sınıf/merkez belirliyoruz.
                  random_state=0,  # Rastgele sayı üretimi sabitleniyor.
                  cluster_std=1.2)  # Grupların yayılımını (standart sapma) ayarlıyoruz.

# Yeni veri seti için scatter plot çiziyoruz.
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()

# Göğüs kanseri veri setini yüklüyoruz.
kanser = load_breast_cancer()

# Veriyi eğitim ve test setlerine ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(kanser.data,  # Özellik vektörü.
                                                    kanser.target,  # Hedef değişken (sınıf etiketleri).
                                                    random_state=0)  # Veri bölünmesini tekrarlanabilir hale getirmek için rastgele durumu ayarlıyoruz.

# SVC modeli oluşturup eğitiyoruz (başlangıçta verileri ölçeklendirmeden).
svc = SVC(gamma='auto').fit(X_train, y_train)

# Eğitim ve test seti üzerindeki başarıyı yazdırıyoruz.
print(svc.score(X_train, y_train))  # Eğitim seti doğruluğu.
print(svc.score(X_test, y_test))  # Test seti doğruluğu.

# Veriyi ölçeklendiriyoruz (normalizasyon işlemi).
min_on_training = X_train.min(axis=0)  # Eğitim setindeki her bir özelliğin minimum değerini alıyoruz.
range_on_training = (X_train - min_on_training).max(axis=0)  # Her bir özelliğin aralığını (maksimum - minimum) hesaplıyoruz.
X_train_scaled = (X_train - min_on_training) / range_on_training  # Eğitim verisini ölçeklendiriyoruz.
X_test_scaled = (X_test - min_on_training) / range_on_training  # Test verisini de aynı şekilde ölçeklendiriyoruz.

# Veriyi ölçeklendirdikten sonra modeli tekrar eğitiyoruz.
svc = SVC(gamma='auto').fit(X_train_scaled, y_train)

# Ölçeklendirilmiş veriyle eğitim ve test doğruluklarını yazdırıyoruz.
print(svc.score(X_train_scaled, y_train))  # Ölçeklendirilmiş eğitim doğruluğu.
print(svc.score(X_test_scaled, y_test))  # Ölçeklendirilmiş test doğruluğu.

# Modeli C parametresiyle yeniden eğitiyoruz (C: Modelin hata toleransını kontrol eder).
svc = SVC(C=1000, gamma='auto').fit(X_train_scaled, y_train)

# Yeni modelin doğruluklarını yazdırıyoruz.
print(svc.score(X_train_scaled, y_train))  # Eğitim doğruluğu (C=1000).
print(svc.score(X_test_scaled, y_test))  # Test doğruluğu (C=1000).

# Parametrelerin Açıklamaları:
# make_blobs:
#
# n_samples: Veri setindeki örnek sayısı.
# centers: Oluşturulacak merkezlerin (sınıfların) sayısı.
# random_state: Sonuçların tekrar elde edilmesi için rastgele sayı üreticisinin başlangıç noktası.
# cluster_std: Grupların yayılımı, standart sapma.
# plt.scatter:
#
# X[:, 0], X[:, 1]: Dağılım grafiğindeki x ve y eksenindeki özellikler.
# c=y: Veri noktalarının sınıflarına göre renklendirilmesi.
# s: Veri noktalarının boyutu.
# cmap: Renk haritası.
# SVC (Destek Vektör Makinesi):
#
# kernel: Kullanılan çekirdek fonksiyonu (lineer veya RBF).
# C: Modelin hata toleransını kontrol eden parametre. Büyük C değerleri daha az hataya izin verir.
# gamma: Çekirdek fonksiyonun etkisini belirler.
# train_test_split:
#
# X: Özellik vektörü.
# y: Hedef değişken (sınıf etiketleri).
# random_state: Bölünmenin tekrarlanabilir olması için rastgele durum.
