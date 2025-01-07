import matplotlib.pyplot as plt  # Grafik çizmek için kullanılan kütüphane.
from sklearn.datasets import load_iris, make_blobs  # Veri seti yüklemek ve sentetik veri oluşturmak için.
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, GroupKFold  # Model doğrulama işlemleri için.
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon modeli oluşturmak için.
import mglearn  # Mglearn, makine öğrenmesi için çeşitli grafik ve araçlar sağlayan bir kütüphane.

# Modeli Kurma (Lojistik Regresyon ile Iris veri seti üzerinde model oluşturma)
iris = load_iris()  # Iris çiçeği veri setini yükler.
X, y = iris.data, iris.target  # Özellikler (X) ve hedef değişken (y) olarak veri setini ayırır.
X_train, X_test, y_train, y_test = train_test_split(X,  # Veriyi eğitim ve test setine böler.
                                                    y,  # Hedef değişkenler.
                                                    random_state=0  # Rastgeleliği sabitler (aynı sonuçları almak için).
                                                    )

print(X.shape)  # (150, 4)  # Iris veri setinde 150 gözlem ve 4 özellik vardır.
print(X_train.shape)  # (112, 4)  # Eğitim veri seti: 112 gözlem, 4 özellik.
print(X_test.shape)  # (38, 4)  # Test veri seti: 38 gözlem, 4 özellik.

# Lojistik Regresyon modeli
logreg = LogisticRegression(solver='lbfgs',  # 'lbfgs' optimizasyon algoritması kullanılır.
                            max_iter=1000  # Maksimum iterasyon sayısı 1000 olarak belirlenir (modelin çözüm bulana kadar kaç kez deneyeceği).
                            )

logreg.fit(X_train, y_train)  # Modeli eğitim verisine (X_train, y_train) uydurur.
sonuc = logreg.score(X_test, y_test)  # Modelin test verisi üzerindeki başarısını ölçer (doğruluk skoru).
print(sonuc)  # 0.9736842105263158 -> Modelin doğruluk skoru.

# Cross Validation (Çapraz Doğrulama)
scores = cross_val_score(logreg, X, y, cv=5)  # 5 katlı çapraz doğrulama (5-fold cross-validation) ile modelin doğruluğunu ölçer.
print(scores)  # Her bir katlama için doğruluk skorları.
print(scores.mean())  # 5 katlama sonucunda ortalama doğruluk skoru.
print(iris.target)  # Iris veri setindeki hedef değişkenler dizisi (class label).

# Çapraz doğrulama stratejisinin grafiği
mglearn.plots.plot_stratified_cross_validation()  # Dengeli (stratified) çapraz doğrulama stratejisinin nasıl çalıştığını gösterir.
plt.show()  # Grafiği ekranda gösterir.

# KFold ile K Katlamalı Doğrulama
kfold = KFold(n_splits=3,  # Veri setini 3 katmana böler.
              shuffle=True,  # Veriyi karıştırır (rastgele şekilde katmanlar oluşturur).
              random_state=0  # Rastgele karıştırma işlemi için sabit rastgelelik.
              )
cross_val_score(logreg, iris.data, iris.target, cv=kfold)  # KFold ile 3 katlamalı çapraz doğrulama.

# LeaveOneOut Doğrulaması
loo = LeaveOneOut()  # LeaveOneOut, her seferinde bir veri noktasını test seti olarak ayırır, geri kalanı eğitim verisi olur.
scores = cross_val_score(estimator=logreg,  # Model (lojistik regresyon).
                         X=iris.data,  # Özellikler.
                         y=iris.target,  # Hedef değişken.
                         cv=loo  # LeaveOneOut çapraz doğrulama stratejisi.
                         )
print(scores.mean())  # 0.9666666666666667 -> LeaveOneOut doğrulama ortalama doğruluk skoru.

# Sentetik Veri Üretme
X, y = make_blobs(n_samples=12,  # 12 örnekten oluşan bir sentetik veri seti oluşturur.
                  random_state=0  # Rastgelelik sabitlenir.
                  )

# Gruplu K-Katlamalı Çapraz Doğrulama (Group K-Fold)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]  # Grupları belirtir, her örnek bir gruba ait.
scores = cross_val_score(estimator=logreg,  # Lojistik regresyon modeli.
                         X=X,  # Özellikler.
                         y=y,  # Hedef değişken.
                         cv=GroupKFold(n_splits=3),  # Grupları dikkate alan 3 katlamalı çapraz doğrulama stratejisi.
                         groups=groups  # Gruplar: Her katlamada aynı gruptaki örnekler bir arada tutulur.
                         )
print(scores)  # Her bir katlamanın doğruluk skoru. Örnek: [0.75, 0.6, 0.66666667]
# train_test_split(X, y, random_state=0): Veriyi eğitim ve test setlerine ayırır. random_state=0, bölme işleminin her çalıştırmada aynı olmasını sağlar.
# LogisticRegression(solver='lbfgs', max_iter=1000): Lojistik regresyon modeli oluşturur.
# solver='lbfgs' optimizasyon yöntemi ve max_iter=1000 modelin maksimum 1000 iterasyon denemesi gerektiğini belirtir.
# cross_val_score(logreg, X, y, cv=5): 5 katlamalı çapraz doğrulama uygular, model her bir katmanda eğitilir ve test edilir.
# KFold(n_splits=3, shuffle=True, random_state=0): Veriyi 3 parçaya böler, her defasında birini test seti olarak kullanır.
# shuffle=True, veri karıştırılır, random_state=0 karıştırma işlemi tekrarlanabilir olur.
# LeaveOneOut(): Leave-One-Out çapraz doğrulama stratejisini uygular, her defasında bir örnek test seti olarak seçilir.
# GroupKFold(n_splits=3): Verileri belirli gruplara göre böler ve çapraz doğrulama yapar. Aynı gruptaki veriler aynı katmanda olur.
