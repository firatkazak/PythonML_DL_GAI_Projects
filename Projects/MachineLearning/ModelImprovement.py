import pandas as pd  # Veri manipülasyonu ve analiz için kullanılan kütüphane.
from sklearn.datasets import load_digits  # Rakamların görüntü verisini yüklemek için.
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score  # Model doğrulama ve parametre arama yöntemleri için.
from sklearn.svm import SVC  # SVM (Destek Vektör Makineleri) sınıflandırma algoritması için.

# Grid Search (Parametre Arama) - SVC (Destek Vektör Makineleri) üzerinde hiperparametre optimizasyonu.
digits = load_digits()  # Rakamların görüntü verilerini yükler (8x8 boyutunda el yazısı rakamlar).
param_grid = [{'kernel': ['rbf'],  # İlk grid: 'rbf' kernel (radial basis function) için parametreler aranacak.
               'C': [1, 10, 100, 1000],  # C: Düzenleme parametresi (modelin ne kadar hatayı tolere edeceği).
               'gamma': [1e-3, 1e-4]},  # Gamma: RBF kernel için etki alanı parametresi.
              {'kernel': ['linear'],  # İkinci grid: 'linear' kernel için parametreler aranacak.
               'C': [1, 10, 100, 1000]}]  # C: Düzenleme parametresi (modelin ne kadar hatayı tolere edeceği).

# GridSearchCV ile SVC modeli için en iyi parametrelerin aranması.
grid_search = GridSearchCV(SVC(),  # SVC modelini kullanarak GridSearchCV sınıfını oluşturur.
                           param_grid,  # Aranacak parametre ızgarası (grid).
                           cv=5  # 5 katlamalı çapraz doğrulama kullanılarak model performansı ölçülür.
                           )

# Veri setini eğitim ve test olarak ayırma.
X_train, X_test, y_train, y_test = train_test_split(digits.data,  # Özellikler (8x8 piksel değerleri).
                                                    digits.target,  # Hedef değişken (rakam etiketleri).
                                                    random_state=0  # Rastgele bölme işlemi için sabitlenmiş rastgelelik.
                                                    )

# Grid Search ile eğitim verisi üzerinde modeli eğit.
grid_search.fit(X_train, y_train)  # Modeli eğitim setinde en iyi parametrelerle eğitir.

# Grid Search sonuçları (isteğe bağlı satırlar)
# print(grid_search.score(X_test, y_test))  # Test seti üzerindeki doğruluk skoru. Örnek: 0.9933333333333333
# print(grid_search.best_params_)  # En iyi hiperparametre kombinasyonu. Örnek: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
# print(grid_search.best_score_)  # Eğitim seti üzerindeki en iyi doğruluk skoru (cross-validation ile). Örnek: 0.9918353297535452
# print(grid_search.best_estimator_)  # En iyi tahminleyici model (SVC). Örnek: SVC(C=10, gamma=0.001)
# print(grid_search.cv_results_)  # Cross-validation sonuçları detaylı bir şekilde döndürülür (fit süreleri, skorlar vb.).

# Grid Search sonuçlarını bir pandas DataFrame olarak görüntüleme.
sonuclar = pd.DataFrame(grid_search.cv_results_)  # Cross-validation sonuçlarını DataFrame'e dönüştürür.
# print(sonuclar)  # Sonuçlar tablosu olarak çıktı verir.
# print(sonuclar.T)  # Transpoze edilmiş (satır ve sütunları yer değiştirmiş) sonuçlar tablosu.

# Nested Cross-Validation (İç İçe Çapraz Doğrulama)
skor = cross_val_score(GridSearchCV(SVC(),  # GridSearchCV ile her katlama için model optimizasyonu yapılır.
                                    param_grid,  # Parametre arama grid'i.
                                    cv=5  # 5 katlamalı çapraz doğrulama.
                                    ),
                       digits.data,  # Özellikler.
                       digits.target,  # Hedef değişken (rakam etiketleri).
                       cv=5  # Dış çapraz doğrulama (5 katlamalı).
                       )

# Nested Cross Validation sonuçları
print(skor)  # Her katlama için doğruluk skorları. Örnek: [0.97777778 0.95       0.98328691 0.98607242 0.9637883]
print(skor.mean())  # Doğruluk skorlarının ortalaması. Örnek: 0.972185082017951

# GridSearchCV(SVC(), param_grid, cv=5):
#
# SVC(): Destek Vektör Makineleri sınıflandırma modeli.
# param_grid: Modelin hiperparametrelerini optimize etmek için arama yapılacak parametrelerin ızgarası. Burada, SVC için kernel, C, ve gamma değerleri aranıyor.
# cv=5: 5 katlamalı çapraz doğrulama yapılacak, yani veri seti 5 parçaya ayrılır, her biri bir kez test seti olarak kullanılır.
# train_test_split(X, y, random_state=0):
#
# Veri setini eğitim ve test setlerine böler.
# X: Özellikler (girdiler).
# y: Hedef değişken (çıktılar).
# random_state=0: Rastgeleliği sabitlemek için kullanılır, böylece sonuçlar her çalıştırmada aynı olur.
# grid_search.fit(X_train, y_train):
#
# Eğitim verisi üzerinde Grid Search yaparak modeli optimize eder ve eğitir.
# cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), digits.data, digits.target, cv=5):
#
# Nested Cross-Validation gerçekleştirir. Bu yöntem, model optimizasyonunun doğruluğunu ölçmek için çapraz doğrulamanın içine başka bir çapraz doğrulama yerleştirir.
# İlk GridSearchCV: İçteki çapraz doğrulamada her defasında hiperparametre arama yapılır.
# İkinci cv=5: Dıştaki çapraz doğrulama, optimizasyonu bağımsız bir test setinde doğrular.
# Bu kod parçası, modelin hiperparametrelerini optimize etmek ve genel performansını doğrulamak için yaygın olarak kullanılan bir yaklaşımdır.
# GridSearchCV ve Nested Cross-Validation, modelin en iyi hiperparametrelerini bulma ve modelin genel başarısını değerlendirme konusunda oldukça etkili yöntemlerdir.
