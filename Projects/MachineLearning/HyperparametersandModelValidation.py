from sklearn.datasets import load_iris  # İris veri setini yüklemek için
from sklearn.neighbors import KNeighborsClassifier  # K-NN sınıflandırıcısı (K-en yakın komşu algoritması)
from sklearn.metrics import accuracy_score  # Doğruluk skorunu hesaplamak için
from sklearn.model_selection import train_test_split, cross_val_score  # Eğitim/test seti bölme ve çapraz doğrulama

# İris veri setini yüklüyoruz
iris = load_iris()  # sklearn'den hazır veri seti (İris çiçeği türleri için)
X = iris.data  # X değişkenine bağımsız değişkenler (özellikler) atanır
y = iris.target  # y değişkenine bağımlı değişken (hedef etiketler) atanır

# K-Nearest Neighbors (KNN) modeli oluşturuyoruz
model = KNeighborsClassifier(n_neighbors=1)  # 1 en yakın komşuyu baz alarak sınıflandırma yapacak

# Modeli eğitiyoruz (fit metodu modeli eğitmek için kullanılır)
model.fit(X, y)  # X: özellikler (girdiler), y: sınıflar (hedefler)

# Eğitim setiyle modelin tahminlerini yapıyoruz
y_model = model.predict(X)  # X üzerinde tahmin yap, sonuçları y_model'e ata

# Modelin doğruluğunu hesaplıyoruz
acscore1 = accuracy_score(y, y_model)  # y: gerçek değerler, y_model: tahmin edilen değerler
print(acscore1)  # 1.0, modelin eğitim setindeki doğruluk oranı (1.0 mükemmel demek)

# Veriyi eğitim ve test setine bölüyoruz
X1, X2, y1, y2 = train_test_split(X,  # X: özellik matrisi
                                  y,  # y: hedef etiketler
                                  random_state=0,  # Sabit rastgelelik (sonuçların tekrar üretilebilir olması için)
                                  train_size=0.5  # Verinin %50'si eğitim, %50'si test seti olacak
                                  )

# Modeli eğitim setiyle tekrar eğitiyoruz
model.fit(X1, y1)  # X1: eğitim özellikleri, y1: eğitim hedefleri

# Test seti üzerinde tahmin yapıyoruz
y2_model = model.predict(X2)  # X2: test özellikleri üzerinde tahmin yap, y2_model'e ata

# Modelin test setindeki doğruluğunu hesaplıyoruz
acscore2 = accuracy_score(y2, y2_model)  # y2: gerçek test etiketleri, y2_model: tahmin edilen test etiketleri
print(acscore2)  # 0.9066666666666666, test seti doğruluk oranı

# Test setiyle modeli eğitip, eğitim seti üzerinde tahmin yapıyoruz (tersine bölme)
y1_model = model.fit(X2, y2).predict(X1)  # X2 ve y2 ile eğit, X1 üzerinde tahmin yap

# Tersine bölme ile doğruluk oranını hesaplıyoruz
acscore3 = accuracy_score(y1, y1_model)  # y1: gerçek eğitim etiketleri, y1_model: tahmin edilen değerler
print(acscore3)  # 0.96, tersine bölme ile doğruluk

# Cross Validation (Çapraz Doğrulama) yapıyoruz
crosvalscore = cross_val_score(estimator=model,  # Model (sınıflandırıcı)
                               X=X,  # Özellik matrisi
                               y=y,  # Hedef etiketler
                               cv=5  # 5 katlı çapraz doğrulama
                               )

# Her katmanda doğruluk skorlarını yazdırıyoruz
print(crosvalscore)  # [0.96666667 0.96666667 0.93333333 0.93333333 1.        ], 5 katlı çapraz doğrulama sonuçları

# load_iris(): İris veri setini yüklüyor.
# KNeighborsClassifier(n_neighbors=1): 1 en yakın komşulu KNN sınıflandırıcısı oluşturur.
# fit(X, y): Modeli, X (özellikler) ve y (etiketler) ile eğitir.
# predict(X): Verilen X veri seti için sınıf tahminleri yapar.
# accuracy_score(y_true, y_pred): Gerçek ve tahmin edilen değerler arasındaki doğruluğu hesaplar.
# train_test_split(X, y, train_size, random_state): Veriyi eğitim ve test setlerine böler.
# cross_val_score(estimator, X, y, cv): Veriyi cv katlı çapraz doğrulama yaparak değerlendirir.
