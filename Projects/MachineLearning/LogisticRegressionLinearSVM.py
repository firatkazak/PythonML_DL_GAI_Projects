from sklearn.datasets import load_breast_cancer  # Meme kanseri veri setini yüklüyor
from sklearn.datasets import make_blobs  # Örnek veri kümeleri oluşturmak için kullanılır
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon modeli
import mglearn  # Görselleştirme araçları sağlayan bir kütüphane
import matplotlib.pyplot as plt  # Grafikler çizmek için
from sklearn.svm import LinearSVC  # Lineer destek vektör makineleri (SVM) sınıflandırıcısı
import numpy as np  # Sayısal işlemler için

# Meme kanseri veri seti yükleniyor
kanser = load_breast_cancer()

# Eğitim ve test setlerine ayrılıyor, hedef sınıflar arasında stratifikasyon yapılıyor
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data,
                                                      kanser.target,
                                                      stratify=kanser.target,  # Sınıf dağılımını korur
                                                      random_state=42)

# Binary classification: Lojistik regresyon modeli
lr = LogisticRegression(solver='liblinear').fit(X_egitim, y_egitim)
print(lr.score(X_egitim, y_egitim))  # Eğitim setindeki doğruluk oranı: 0.953
print(lr.score(X_test, y_test))  # Test setindeki doğruluk oranı: 0.958

# Farklı C değeriyle lojistik regresyon (C, düzenleme gücünü kontrol eder)
lr100 = LogisticRegression(C=100, solver='liblinear').fit(X_egitim, y_egitim)
print(lr100.score(X_egitim, y_egitim))  # Eğitim doğruluğu: 0.967
print(lr100.score(X_test, y_test))  # Test doğruluğu: 0.965

# C=0.01 ile lojistik regresyon (daha güçlü düzenleme)
lr001 = LogisticRegression(C=0.01, solver='liblinear').fit(X_egitim, y_egitim)
print(lr001.score(X_egitim, y_egitim))  # Eğitim doğruluğu: 0.934
print(lr001.score(X_test, y_test))  # Test doğruluğu: 0.930

# L1 cezasıyla Lojistik regresyon ve farklı C değerleri için
for C, market in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear', C=C).fit(X_egitim, y_egitim)
    print('C={:.3f} için eğitim doğruluk {:.2f}'.format(C, lr_l1.score(X_egitim, y_egitim)))
    print('C={:.3f} için test doğruluk {:.2f}'.format(C, lr_l1.score(X_test, y_test)))

# Multiclass classification (3 sınıflı örnek veri kümesi oluşturuluyor)
X, y = make_blobs(random_state=42)  # Örnek veri noktaları ve sınıf etiketleri oluşturuluyor

# Özellikleri ve sınıfları görselleştir
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Öznitelik 0')  # X ekseni etiketi
plt.ylabel('Öznitelik 1')  # Y ekseni etiketi
plt.legend(['sınıf 0', 'sınıf 1', 'sınıf 2'])  # Sınıf etiketleri
plt.show()

# Linear SVM (Destek Vektör Makinesi)
linear_svm = LinearSVC(dual=False).fit(X, y)

# Destek vektör makinesi katsayılarını yazdırıyor
print(linear_svm.coef_)  # Alt satırda;
# [[-0.17492412  0.23140766]
#  [ 0.47622012 -0.06936786]
#  [-0.18914207 -0.20400079]]

# Veri noktalarını tekrar çiziyor
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(start=-15, stop=15)  # Grafikte kullanılacak bir çizgi oluşturuluyor

# Her sınıf için ayırma doğrularını çiziyor
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)  # Doğru çizimi
plt.ylim(-10, 15)  # Y ekseni sınırları
plt.xlim(-10, 8)  # X ekseni sınırları
plt.xlabel('Öznitelik 0')  # X ekseni etiketi
plt.ylabel('Öznitelik 1')  # Y ekseni etiketi
plt.legend(['sınıf 0', 'sınıf 1', 'sınıf 2', 'doğru sınıf 0', 'doğru sınıf 1', 'doğru sınıf 2'], loc=(1.02, 0.4))
plt.show()  # Grafik gösteriliyor

# Lojistik Regresyon (Binary Classification): Meme kanseri veri seti ile, düzenleme parametresi C farklı değerlerde ayarlanarak modelin performansı test ediliyor.
# C parametresi modelin düzenleme seviyesini belirler.
# Küçük C daha fazla düzenleme (daha basit model), büyük C ise daha az düzenleme (daha karmaşık model) anlamına gelir.

# L1 Cezası: L1 düzenlemesi bazı özelliklerin katsayılarını sıfırlayarak özellik seçiminde kullanılır.

# Multiclass Classification: Yapay bir veri kümesi oluşturulup 3 sınıfı ayırmak için görselleştiriliyor. Bu sınıflar arasında ayırma sınırları çiziliyor.

# Linear SVM: Destek vektör makineleri ile sınıflandırma yapılırken, ayırma sınırları (decision boundaries) görselleştiriliyor.

# dual Parametresi Nedir?
# dual=True: Çözümü çift formda hesaplar, genellikle çok sayıda özellik olduğunda kullanılır.
# dual=False: Doğrudan primal formda çözüm arar, genellikle çok sayıda örnek ve az sayıda özellik olduğunda kullanılması önerilir.
